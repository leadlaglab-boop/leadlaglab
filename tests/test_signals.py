"""
Tests for M3 signal collectors.

All network calls are mocked. We test:
  - Record structure and schema compliance
  - No look-ahead: observed_at < market_close(decision_date)
  - Idempotency: _all_fetched short-circuits correctly
  - NLP scoring: VADER works without FinBERT
  - EDGAR: hash-based security_ids are skipped
  - GDELT: missing alias returns empty
  - Trends: anchor normalization structure
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from pipeline.sources.base import DateRange
from pipeline.utils.calendar import market_close_utc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_date_range(n_days: int = 3) -> DateRange:
    start = date(2024, 1, 2)
    end = start + timedelta(days=n_days - 1)
    return DateRange(start=start, end=end)


def assert_schema_valid(records: list[dict[str, Any]]) -> None:
    """All required signal_raw fields must be present and typed correctly."""
    required = {
        "source",
        "security_id",
        "effective_date",
        "observed_at",
        "payload_json",
        "n_items",
        "source_version",
        "schema_version",
    }
    for rec in records:
        missing = required - set(rec.keys())
        assert not missing, f"Record missing fields: {missing}"
        assert isinstance(rec["payload_json"], str), "payload_json must be str"
        assert json.loads(rec["payload_json"]) is not None
        assert isinstance(rec["n_items"], int)
        # observed_at must be a timezone-aware datetime
        obs = rec["observed_at"]
        assert isinstance(obs, datetime), f"observed_at not datetime: {type(obs)}"
        assert obs.tzinfo is not None, "observed_at must be timezone-aware"


# ---------------------------------------------------------------------------
# Wikipedia pageviews
# ---------------------------------------------------------------------------


class TestWikipediaPageviewsSource:
    def _make_source(self, tmp_path: Path) -> Any:
        from pipeline.sources.wikipedia_pageviews import WikipediaPageviewsSource

        map_path = tmp_path / "wikipedia_article_map.yaml"
        map_path.write_text(
            "overrides:\n  AAPL:\n    articles:\n      - Apple Inc.\n    ambiguous: false\n"
        )
        return WikipediaPageviewsSource(map_path, lookup_cache_path=None)

    def test_fetch_ticker_returns_records(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)
        dr = make_date_range(3)

        fake_response = {
            "items": [
                {"timestamp": "20240102", "views": 10000},
                {"timestamp": "20240103", "views": 12000},
                {"timestamp": "20240104", "views": 9000},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response

        with patch.object(source, "_get", return_value=mock_resp):
            records = source.fetch_ticker("AAPL", "0000320193", "Apple Inc.", dr)

        assert len(records) == 3
        assert_schema_valid(records)

    def test_views_aggregated_per_day(self, tmp_path: Path) -> None:
        """Multiple articles for same ticker should be summed."""
        map_path = tmp_path / "map.yaml"
        map_path.write_text(
            "overrides:\n"
            "  GOOGL:\n"
            "    articles:\n"
            "      - Alphabet Inc.\n"
            "      - Google\n"
            "    ambiguous: false\n"
        )
        from pipeline.sources.wikipedia_pageviews import WikipediaPageviewsSource

        source = WikipediaPageviewsSource(map_path)
        dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 2))

        fake = {"items": [{"timestamp": "20240102", "views": 5000}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake

        with patch.object(source, "_get", return_value=mock_resp):
            records = source.fetch_ticker("GOOGL", "test-sid", "Alphabet Inc.", dr)

        assert len(records) == 1
        payload = json.loads(records[0]["payload_json"])
        # 5000 + 5000 from two articles
        assert payload["views"] == 10000

    def test_observed_at_backfilled(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)
        dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 2))

        fake = {"items": [{"timestamp": "20240102", "views": 100}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake

        with patch.object(source, "_get", return_value=mock_resp):
            records = source.fetch_ticker("AAPL", "0000320193", "Apple Inc.", dr, "backfilled")

        assert len(records) == 1
        expected = market_close_utc("2024-01-02")
        assert records[0]["observed_at"] == expected

    def test_404_returns_empty(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)
        dr = make_date_range(3)

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch.object(source, "_get", return_value=mock_resp):
            records = source.fetch_ticker("AAPL", "0000320193", "Apple Inc.", dr)

        assert records == []


# ---------------------------------------------------------------------------
# EDGAR
# ---------------------------------------------------------------------------


class TestEdgarSource:
    def _make_source(self) -> Any:
        from pipeline.sources.edgar import EdgarSource

        return EdgarSource()

    def test_hash_based_id_skipped(self) -> None:
        source = self._make_source()
        dr = make_date_range(3)
        records = source.fetch_ticker("HOOD", "h595fc32c3", "h595fc32c3", dr)
        assert records == []

    def test_cik_based_returns_records(self) -> None:
        source = self._make_source()
        dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 5))

        fake_submissions = {
            "filings": {
                "recent": {
                    "form": ["4", "8-K", "10-K"],
                    "filingDate": ["2024-01-02", "2024-01-03", "2024-01-04"],
                    "acceptanceDateTime": [
                        "2024-01-02T16:30:00.000Z",
                        "2024-01-03T10:00:00.000Z",
                        "2024-01-04T09:00:00.000Z",
                    ],
                    "accessionNumber": ["0001-01", "0001-02", "0001-03"],
                }
            }
        }

        with patch.object(source, "_get_submissions", return_value=fake_submissions):
            records = source.fetch_ticker("AAPL", "0000320193", "0000320193", dr)

        # 10-K is not in FORM_TYPES, so only Form 4 + 8-K = 2 filing days
        assert len(records) == 2
        assert_schema_valid(records)

    def test_form_counts_in_payload(self) -> None:
        source = self._make_source()
        dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 2))

        fake_submissions = {
            "filings": {
                "recent": {
                    "form": ["4", "4", "8-K"],
                    "filingDate": ["2024-01-02", "2024-01-02", "2024-01-02"],
                    "acceptanceDateTime": [
                        "2024-01-02T14:00:00.000Z",
                        "2024-01-02T15:00:00.000Z",
                        "2024-01-02T16:00:00.000Z",
                    ],
                    "accessionNumber": ["001", "002", "003"],
                }
            }
        }

        with patch.object(source, "_get_submissions", return_value=fake_submissions):
            records = source.fetch_ticker("AAPL", "0000320193", "0000320193", dr)

        assert len(records) == 1
        payload = json.loads(records[0]["payload_json"])
        assert payload["form_4_count"] == 2
        assert payload["form_8k_count"] == 1
        assert payload["total_filings"] == 3

    def test_no_submissions_returns_empty(self) -> None:
        source = self._make_source()
        dr = make_date_range(3)

        with patch.object(source, "_get_submissions", return_value=None):
            records = source.fetch_ticker("MSFT", "0000789019", "0000789019", dr)

        assert records == []


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------


class TestGdeltSource:
    def _make_source(self, tmp_path: Path) -> Any:
        from pipeline.sources.gdelt import GdeltSource

        aliases_path = tmp_path / "gdelt_entity_aliases.yaml"
        aliases_path.write_text(
            "aliases:\n"
            "  AAPL:\n"
            '    primary: \'"Apple" OR "AAPL"\'\n'
            "    ambiguous: false\n"
            "  UNKNOWN:\n"
            "    primary: '\"Unknown Corp\"'\n"
            "    ambiguous: true\n"
        )
        return GdeltSource(aliases_path)

    def test_no_alias_returns_none(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)
        result = source.fetch_ticker_day("GOOGL", "sid", date(2024, 1, 2))
        assert result is None

    def test_fetch_ticker_day_structure(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)

        fake_artlist = {
            "articles": [
                {
                    "url": "http://example.com/1",
                    "title": "Apple reports strong earnings",
                    "domain": "example.com",
                    "seendate": "20240102T120000Z",
                    "language": "English",
                },
            ]
        }
        fake_tone = {"timeline": [{"data": [{"value": 2.5, "norm": 10}]}]}

        responses = [fake_artlist, fake_tone]
        call_idx = [0]

        def mock_get(params: dict) -> dict | None:
            result = responses[call_idx[0]] if call_idx[0] < len(responses) else None
            call_idx[0] += 1
            return result

        with patch.object(source, "_get", side_effect=mock_get):
            record = source.fetch_ticker_day("AAPL", "0000320193", date(2024, 1, 2))

        assert record is not None
        assert_schema_valid([record])
        payload = json.loads(record["payload_json"])
        assert "article_count" in payload
        assert "avg_tone" in payload
        assert "sample_articles" in payload

    def test_no_coverage_returns_none(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)

        with patch.object(source, "_get", return_value=None):
            result = source.fetch_ticker_day("AAPL", "sid", date(2024, 1, 2))

        assert result is None

    def test_observed_at_backfilled(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)
        d = date(2024, 1, 2)

        fake_artlist = {
            "articles": [
                {"url": "u", "title": "t", "domain": "d", "seendate": "s", "language": "English"}
            ]
        }

        responses = [fake_artlist, None]
        call_idx = [0]

        def mock_get(params: dict) -> dict | None:
            result = responses[call_idx[0]] if call_idx[0] < len(responses) else None
            call_idx[0] += 1
            return result

        with patch.object(source, "_get", side_effect=mock_get):
            record = source.fetch_ticker_day("AAPL", "sid", d, data_type="backfilled")

        assert record is not None
        expected = market_close_utc("2024-01-02")
        assert record["observed_at"] == expected


# ---------------------------------------------------------------------------
# Google Trends
# ---------------------------------------------------------------------------


class TestTrendsSource:
    def _make_source(self, tmp_path: Path) -> Any:
        from pipeline.sources.trends import TrendsSource

        cache_dir = tmp_path / "trends_cache"
        return TrendsSource(cache_dir)

    def test_fetch_returns_records(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)
        dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 4))

        fake_result = {
            "AAPL": {
                "2024-01-02": 45.0,
                "2024-01-03": 50.0,
                "2024-01-04": 47.5,
            }
        }

        with patch.object(source, "_fetch_batch_raw", return_value=fake_result):
            records = source.fetch_tickers_batch(
                [("AAPL", "0000320193")], dr, data_type="backfilled"
            )

        assert len(records) == 3
        assert_schema_valid(records)

    def test_payload_has_anchor_normalized_interest(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)
        dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 2))

        fake_result = {"AAPL": {"2024-01-02": 75.0}}

        with patch.object(source, "_fetch_batch_raw", return_value=fake_result):
            records = source.fetch_tickers_batch([("AAPL", "sid")], dr)

        assert len(records) == 1
        payload = json.loads(records[0]["payload_json"])
        assert "anchor_normalized_interest" in payload
        assert payload["anchor_normalized_interest"] == 75.0
        assert "anchor_term" in payload

    def test_graceful_degradation_on_failure(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)
        dr = make_date_range(3)

        with patch.object(source, "_fetch_batch_raw", return_value=None):
            records = source.fetch_tickers_batch([("AAPL", "sid"), ("MSFT", "sid2")], dr)

        assert records == []

    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        source = self._make_source(tmp_path)
        key = source._cache_key(["AAPL"], date(2024, 1, 2), date(2024, 1, 4))
        payload = {"AAPL": {"2024-01-02": 33.0}}
        source._cache_set(key, payload)
        cached = source._cache_get(key)
        assert cached == payload


# ---------------------------------------------------------------------------
# NLP scorer
# ---------------------------------------------------------------------------


class TestNLPScorer:
    def test_vader_only_mode(self) -> None:
        from pipeline.sources.nlp import NLPScorer

        scorer = NLPScorer(use_finbert=False)
        result = scorer.score_text("Apple stock surges on record earnings")

        # VADER scores
        assert -1.0 <= result.vader_compound <= 1.0
        # FinBERT should be uniform prior
        assert abs(result.finbert_positive - 1 / 3) < 0.01

    def test_batch_consistency(self) -> None:
        from pipeline.sources.nlp import NLPScorer

        scorer = NLPScorer(use_finbert=False)
        texts = ["Apple stock surges", "Market crashes on Fed fears", "Flat trading day"]
        batch = scorer.score_batch(texts)
        singles = [scorer.score_text(t) for t in texts]

        assert len(batch) == 3
        for b, s in zip(batch, singles, strict=False):
            assert abs(b.vader_compound - s.vader_compound) < 1e-4

    def test_score_gdelt_records_adds_nlp(self) -> None:
        from pipeline.sources.nlp import NLPScorer

        scorer = NLPScorer(use_finbert=False)
        records = [
            {
                "source": "gdelt",
                "security_id": "0000320193",
                "effective_date": date(2024, 1, 2),
                "observed_at": datetime(2024, 1, 2, 21, 0, 0, tzinfo=UTC),
                "payload_json": json.dumps(
                    {
                        "article_count": 2,
                        "avg_tone": 1.5,
                        "ambiguous": False,
                        "query": '"Apple"',
                        "sample_articles": [
                            {"title": "Apple beats earnings estimates", "url": "http://x.com/1"},
                            {"title": "iPhone sales soar", "url": "http://x.com/2"},
                        ],
                    }
                ),
                "n_items": 2,
                "source_version": "v1",
                "schema_version": "1.0.0",
            }
        ]

        scored = scorer.score_gdelt_records(records)
        assert len(scored) == 1
        payload = json.loads(scored[0]["payload_json"])
        assert "nlp" in payload
        assert "vader_avg_compound" in payload["nlp"]
        assert "n_articles_scored" in payload["nlp"]
        assert payload["nlp"]["n_articles_scored"] == 2

    def test_score_gdelt_no_articles_passthrough(self) -> None:
        from pipeline.sources.nlp import NLPScorer

        scorer = NLPScorer(use_finbert=False)
        record = {
            "source": "gdelt",
            "security_id": "test",
            "payload_json": json.dumps({"article_count": 0, "sample_articles": []}),
        }
        scored = scorer.score_gdelt_records([record])
        assert len(scored) == 1
        # No nlp key added when no articles
        payload = json.loads(scored[0]["payload_json"])
        assert "nlp" not in payload


# ---------------------------------------------------------------------------
# Signal ingestor (integration-level with mocked source methods)
# ---------------------------------------------------------------------------


class TestSignalIngestor:
    def _make_ingestor(self, tmp_path: Path) -> Any:
        from pipeline.signals.ingestor import SignalIngestor

        # Minimal config files
        article_map = tmp_path / "config" / "wikipedia_article_map.yaml"
        article_map.parent.mkdir()
        article_map.write_text(
            "overrides:\n  AAPL:\n    articles: [Apple Inc.]\n    ambiguous: false\n"
        )
        aliases = tmp_path / "config" / "gdelt_entity_aliases.yaml"
        aliases.write_text("aliases:\n  AAPL:\n    primary: '\"Apple\"'\n    ambiguous: false\n")

        data_repo = tmp_path / "data"
        data_repo.mkdir()
        (data_repo / "processed" / "security_master").mkdir(parents=True)

        # Write a minimal security_master parquet
        import pyarrow as pa
        import pyarrow.parquet as pq

        from pipeline.validate.schemas import SECURITY_MASTER_SCHEMA

        sm_data = {
            "security_id": ["0000320193"],
            "ticker": ["AAPL"],
            "valid_from": [date(2020, 1, 1)],
            "valid_to": [None],
            "name": ["Apple Inc."],
            "gics_sector": ["Information Technology"],
            "cohort": ["sp500"],
            "schema_version": ["1.0.0"],
        }
        import pandas as pd

        df = pd.DataFrame(sm_data)
        df["valid_from"] = pd.to_datetime(df["valid_from"]).dt.date
        df["valid_to"] = None
        table = pa.Table.from_pandas(df, schema=SECURITY_MASTER_SCHEMA, preserve_index=False)
        pq.write_table(
            table, data_repo / "processed" / "security_master" / "security_master_latest.parquet"
        )

        return SignalIngestor(
            data_repo_path=data_repo,
            config_dir=tmp_path / "config",
            use_finbert=False,
        )

    def test_all_fetched_false_when_missing(self, tmp_path: Path) -> None:
        ingestor = self._make_ingestor(tmp_path)
        dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 3))
        assert not ingestor._all_fetched("wikipedia_pageviews", "AAPL", dr)

    def test_all_fetched_true_when_present(self, tmp_path: Path) -> None:
        ingestor = self._make_ingestor(tmp_path)
        dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 2))

        # Create the expected file
        from pipeline.signals.ingestor import _archive_path

        path = _archive_path(ingestor.data_repo, "wikipedia_pageviews", date(2024, 1, 2), "AAPL")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dummy")

        assert ingestor._all_fetched("wikipedia_pageviews", "AAPL", dr)

    def test_run_wikipedia_only(self, tmp_path: Path) -> None:
        """Smoke test: run with mocked wikipedia fetch, check files written."""
        ingestor = self._make_ingestor(tmp_path)
        dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 2))

        fake_records = [
            {
                "source": "wikipedia_pageviews",
                "security_id": "0000320193",
                "effective_date": date(2024, 1, 2),
                "observed_at": datetime(2024, 1, 2, 21, 0, 0, tzinfo=UTC),
                "payload_json": json.dumps(
                    {"views": 5000, "articles": ["Apple Inc."], "ambiguous": False}
                ),
                "n_items": 1,
                "source_version": "v1",
                "schema_version": "1.0.0",
            }
        ]

        with patch.object(ingestor.wikipedia, "fetch_ticker", return_value=fake_records):
            summary = ingestor.run(dr, data_type="backfilled", sources=["wikipedia"])

        assert summary["sources"]["wikipedia"]["succeeded"] == 1
        assert summary["sources"]["wikipedia"]["rows"] == 1

        # Check the file was actually written
        from pipeline.signals.ingestor import _archive_path

        path = _archive_path(ingestor.data_repo, "wikipedia_pageviews", date(2024, 1, 2), "AAPL")
        assert path.exists()

"""
Signal ingestor: orchestrates all signal source collectors.

For each source:
  1. Check which (ticker, date) pairs already have data (idempotency)
  2. Fetch missing data
  3. Write to archive as Parquet, partitioned by source/year/month/day

All raw signal records follow the SIGNAL_RAW_SCHEMA. Each source plugin
writes its own payload_json with source-specific fields.

NLP scoring of GDELT text runs at the end of the ingest pass (batch efficiency).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from pipeline.sources.base import DateRange
from pipeline.sources.edgar import EdgarSource
from pipeline.sources.gdelt import GdeltSource
from pipeline.sources.nlp import NLPScorer
from pipeline.sources.trends import BATCH_SIZE, TrendsSource
from pipeline.sources.wikipedia_pageviews import WikipediaPageviewsSource
from pipeline.universe.security_master import load_security_master
from pipeline.validate.schemas import SIGNAL_RAW_SCHEMA

log = structlog.get_logger()


def _archive_path(
    data_repo: Path,
    source: str,
    d: date,
    ticker: str,
) -> Path:
    return (
        data_repo
        / "raw"
        / source
        / f"year={d.year}"
        / f"month={d.month:02d}"
        / f"day={d.day:02d}"
        / f"{ticker.lower()}.parquet"
    )


def _already_fetched(data_repo: Path, source: str, d: date, ticker: str) -> bool:
    return _archive_path(data_repo, source, d, ticker).exists()


def _write_records(
    data_repo: Path,
    source: str,
    ticker: str,
    records: list[dict[str, Any]],
) -> int:
    """Write signal_raw records to the archive. Groups by effective_date for partitioning."""
    if not records:
        return 0

    df = pd.DataFrame(records)
    df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date

    rows_written = 0
    for d, group in df.groupby("effective_date"):
        assert isinstance(d, date)
        path = _archive_path(data_repo, source, d, ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = _df_to_arrow(group)
        pq.write_table(table, path, compression="zstd")
        rows_written += len(group)

    return rows_written


def _df_to_arrow(df: pd.DataFrame) -> pa.Table:
    df = df.copy()
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    return pa.Table.from_pandas(
        df[
            [
                "source",
                "security_id",
                "effective_date",
                "observed_at",
                "payload_json",
                "n_items",
                "source_version",
                "schema_version",
            ]
        ],
        schema=SIGNAL_RAW_SCHEMA,
        preserve_index=False,
    )


class SignalIngestor:
    def __init__(
        self,
        data_repo_path: Path,
        config_dir: Path,
        use_finbert: bool = True,
    ) -> None:
        self.data_repo = data_repo_path
        self.config_dir = config_dir

        article_map = config_dir / "wikipedia_article_map.yaml"
        lookup_cache = data_repo_path / "processed" / "wikipedia_article_lookup_cache.json"
        aliases = config_dir / "gdelt_entity_aliases.yaml"
        trends_cache = data_repo_path / "processed" / "trends_cache"

        self.wikipedia = WikipediaPageviewsSource(article_map, lookup_cache)
        self.edgar = EdgarSource()
        self.gdelt = GdeltSource(aliases)
        self.trends = TrendsSource(trends_cache)
        self.nlp = NLPScorer(use_finbert=use_finbert)

    def run(
        self,
        date_range: DateRange,
        data_type: str = "backfilled",
        sources: list[str] | None = None,
        max_tickers: int | None = None,
    ) -> dict[str, Any]:
        """
        Run all signal collectors for the given date range and universe.
        Returns a summary dict with counts per source.
        """
        # trends is opt-in: excluded from default run due to ToS uncertainty (unofficial scraper)
        enabled = set(sources or ["wikipedia", "edgar", "gdelt"])
        sm = load_security_master(self.data_repo)
        active = sm[sm["valid_to"].isna()].copy()
        if max_tickers:
            active = active.head(max_tickers)

        summary: dict[str, Any] = {
            "date_range": f"{date_range.start} – {date_range.end}",
            "tickers": len(active),
            "sources": {},
        }

        # 1. Wikipedia pageviews
        if "wikipedia" in enabled:
            summary["sources"]["wikipedia"] = self._run_wikipedia(active, date_range, data_type)

        # 2. EDGAR filings
        if "edgar" in enabled:
            summary["sources"]["edgar"] = self._run_edgar(active, date_range, data_type)

        # 3. GDELT (with NLP scoring at the end)
        if "gdelt" in enabled:
            summary["sources"]["gdelt"] = self._run_gdelt(active, date_range, data_type)

        # 4. Google Trends (batched)
        if "trends" in enabled:
            summary["sources"]["trends"] = self._run_trends(active, date_range, data_type)

        log.info("signal ingest complete", summary=summary)
        return summary

    def _run_wikipedia(
        self, active: pd.DataFrame, date_range: DateRange, data_type: str
    ) -> dict[str, int]:
        succeeded = failed = rows = 0
        for _, row in active.iterrows():
            ticker = row["ticker"]
            sid = row["security_id"]
            name = row["name"]

            # Skip if today's data already exists for all days in range
            if self._all_fetched("wikipedia_pageviews", ticker, date_range):
                log.debug("wikipedia: already fetched", ticker=ticker)
                continue

            try:
                records = self.wikipedia.fetch_ticker(ticker, sid, name, date_range, data_type)
                n = _write_records(self.data_repo, "wikipedia_pageviews", ticker, records)
                rows += n
                succeeded += 1
            except Exception as e:
                log.error("wikipedia: ticker failed", ticker=ticker, error=str(e))
                failed += 1

        return {"succeeded": succeeded, "failed": failed, "rows": rows}

    def _run_edgar(
        self, active: pd.DataFrame, date_range: DateRange, data_type: str
    ) -> dict[str, int]:
        succeeded = failed = rows = 0
        for _, row in active.iterrows():
            ticker = row["ticker"]
            sid = row["security_id"]
            cik = str(row["security_id"])  # security_id IS the CIK for CIK-based IDs

            if self._all_fetched("edgar", ticker, date_range):
                continue

            try:
                records = self.edgar.fetch_ticker(ticker, sid, cik, date_range, data_type)
                if records:
                    n = _write_records(self.data_repo, "edgar", ticker, records)
                    rows += n
                succeeded += 1
            except Exception as e:
                log.error("edgar: ticker failed", ticker=ticker, error=str(e))
                failed += 1

        return {"succeeded": succeeded, "failed": failed, "rows": rows}

    def _run_gdelt(
        self, active: pd.DataFrame, date_range: DateRange, data_type: str
    ) -> dict[str, int]:
        succeeded = failed = rows = 0
        gdelt_records_for_nlp: list[dict[str, Any]] = []

        for _, row in active.iterrows():
            ticker = row["ticker"]
            sid = row["security_id"]

            if self._all_fetched("gdelt", ticker, date_range):
                continue

            try:
                records = self.gdelt.fetch_ticker(ticker, sid, date_range, data_type)
                if records:
                    gdelt_records_for_nlp.extend(records)
                succeeded += 1
            except Exception as e:
                log.error("gdelt: ticker failed", ticker=ticker, error=str(e))
                failed += 1

        # NLP scoring (batch over all collected GDELT records)
        if gdelt_records_for_nlp:
            log.info("running NLP scoring", n_records=len(gdelt_records_for_nlp))
            try:
                scored = self.nlp.score_gdelt_records(gdelt_records_for_nlp)
            except Exception as e:
                log.error("NLP scoring failed; writing records without NLP scores", error=str(e))
                scored = gdelt_records_for_nlp  # preserve records rather than drop them
            # Write scored records to archive
            # Group by ticker (from security_id lookup is complex; use source + security_id path)
            for record in scored:
                # Find ticker for this security_id
                sm_row = active[active["security_id"] == record["security_id"]]
                if sm_row.empty:
                    continue
                ticker = sm_row.iloc[0]["ticker"]
                n = _write_records(self.data_repo, "gdelt", ticker, [record])
                rows += n

        return {"succeeded": succeeded, "failed": failed, "rows": rows}

    def _run_trends(
        self, active: pd.DataFrame, date_range: DateRange, data_type: str
    ) -> dict[str, int]:
        rows = batches_ok = batches_failed = 0
        tickers = list(active["ticker"])
        sids = dict(zip(active["ticker"], active["security_id"], strict=False))

        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i : i + BATCH_SIZE]
            pairs = [(t, sids[t]) for t in batch]

            # Skip if all already fetched
            if all(self._all_fetched("google_trends", t, date_range) for t in batch):
                continue

            try:
                records = self.trends.fetch_tickers_batch(pairs, date_range, data_type)
                if records:
                    # Group back by ticker
                    ticker_records: dict[str, list[dict[str, Any]]] = {}
                    for r in records:
                        sm_row = active[active["security_id"] == r["security_id"]]
                        if sm_row.empty:
                            continue
                        t = sm_row.iloc[0]["ticker"]
                        ticker_records.setdefault(t, []).append(r)
                    for t, trecs in ticker_records.items():
                        n = _write_records(self.data_repo, "google_trends", t, trecs)
                        rows += n
                batches_ok += 1
            except Exception as e:
                log.error("trends: batch failed", batch=batch, error=str(e))
                batches_failed += 1

        return {"batches_ok": batches_ok, "batches_failed": batches_failed, "rows": rows}

    def _all_fetched(self, source: str, ticker: str, date_range: DateRange) -> bool:
        """Return True if data already exists for every day in the range."""
        current = date_range.start
        while current <= date_range.end:
            if not _already_fetched(self.data_repo, source, current, ticker):
                return False
            current += timedelta(days=1)
        return True

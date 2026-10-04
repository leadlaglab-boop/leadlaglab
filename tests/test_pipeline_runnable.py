"""Tests for the pieces that make the daily pipeline runnable end to end.

Covers:
  - lll-* entrypoints forward their CLI args
  - PriceIngestor uses Tiingo first and Stooq only as a fallback
  - load_prices attaches the ticker from the filename
  - prices_sample.json publishes a rebased index, never raw prices
  - evaluate skips cleanly when there is no data yet
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline import cli
from pipeline.build.site_data import build_prices_sample_json
from pipeline.evaluation.engine import NoEvalDataError
from pipeline.sources.base import DateRange
from pipeline.sources.prices import PriceIngestor, load_prices
from pipeline.validate.schemas import SCHEMA_VERSION


def _price_record(sid: str, d: date, close: float, source: str) -> dict[str, Any]:
    return {
        "security_id": sid,
        "date": d,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adj_close": close,
        "volume": 100,
        "observed_at": datetime(d.year, d.month, d.day, 21, tzinfo=UTC),
        "source": source,
        "data_type": "backfilled",
        "schema_version": SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# CLI entrypoints
# ---------------------------------------------------------------------------


class TestEntrypoints:
    @pytest.mark.parametrize(
        ("func", "command"),
        [
            (cli.universe, "universe"),
            (cli.prices, "prices"),
            (cli.ingest, "ingest"),
            (cli.features, "features"),
            (cli.evaluate, "evaluate"),
            (cli.predict, "predict"),
            (cli.build_site, "build-site"),
            (cli.monthly_report, "monthly-report"),
        ],
    )
    def test_forwards_argv(self, func: Any, command: str) -> None:
        with (
            patch.object(cli, "app") as app,
            patch("sys.argv", ["lll-x", "--start", "2026-10-01", "--live"]),
        ):
            func()
        app.assert_called_once_with(
            [command, "--start", "2026-10-01", "--live"], standalone_mode=True
        )


# ---------------------------------------------------------------------------
# PriceIngestor source order
# ---------------------------------------------------------------------------


class TestPriceIngestor:
    def _ingestor(self, tmp_path: Path) -> PriceIngestor:
        ing = PriceIngestor(data_repo_path=tmp_path, tiingo_api_key="test-key")
        ing.tiingo = MagicMock()
        ing.stooq = MagicMock()
        ing.tiingo.available = True
        return ing

    def test_tiingo_is_primary(self, tmp_path: Path) -> None:
        ing = self._ingestor(tmp_path)
        d = date(2026, 9, 30)
        ing.tiingo.fetch_ticker.return_value = [_price_record("sid", d, 10.0, "tiingo")]

        n = ing.fetch_and_store("AAPL", "sid", DateRange(d, d))

        assert n == 1
        ing.stooq.fetch_ticker.assert_not_called()
        assert (tmp_path / "raw/prices/year=2026/month=09/day=30/aapl.parquet").exists()

    def test_falls_back_to_stooq(self, tmp_path: Path) -> None:
        ing = self._ingestor(tmp_path)
        d = date(2026, 9, 30)
        ing.tiingo.fetch_ticker.return_value = []
        ing.stooq.fetch_ticker.return_value = [_price_record("sid", d, 10.0, "stooq")]

        assert ing.fetch_and_store("AAPL", "sid", DateRange(d, d)) == 1
        ing.stooq.fetch_ticker.assert_called_once()

    def test_no_source_returns_zero(self, tmp_path: Path) -> None:
        ing = self._ingestor(tmp_path)
        ing.tiingo.fetch_ticker.return_value = []
        ing.stooq.fetch_ticker.return_value = []
        d = date(2026, 9, 30)
        assert ing.fetch_and_store("AAPL", "sid", DateRange(d, d)) == 0


# ---------------------------------------------------------------------------
# Archive round-trip and published sample
# ---------------------------------------------------------------------------


def _write_prices(tmp_path: Path, ticker: str, closes: list[float], start: date) -> None:
    ing = PriceIngestor(data_repo_path=tmp_path, tiingo_api_key="test-key")
    ing.tiingo = MagicMock()
    ing.tiingo.available = True
    ing.tiingo.fetch_ticker.return_value = [
        _price_record(f"sid_{ticker}", start + timedelta(days=i), c, "tiingo")
        for i, c in enumerate(closes)
    ]
    ing.fetch_and_store(ticker, f"sid_{ticker}", DateRange(start, start))


def test_load_prices_attaches_ticker(tmp_path: Path) -> None:
    start = date.today() - timedelta(days=5)
    _write_prices(tmp_path, "AAPL", [10.0, 11.0], start)
    _write_prices(tmp_path, "BRK-B", [20.0], start)

    df = load_prices(tmp_path)

    assert sorted(df["ticker"].unique()) == ["AAPL", "BRK-B"]
    assert len(df[df["ticker"] == "AAPL"]) == 2


def test_prices_sample_is_rebased_index(tmp_path: Path) -> None:
    start = date.today() - timedelta(days=5)
    _write_prices(tmp_path, "AAPL", [50.0, 55.0, 45.0], start)
    site = tmp_path / "site"
    site.mkdir()

    out = build_prices_sample_json(tmp_path, site)

    series = out["series"]["AAPL"]
    assert [p["index"] for p in series] == [100.0, 110.0, 90.0]
    # No raw vendor price fields may be published
    assert all(set(p) == {"date", "index"} for p in series)
    written = json.loads((site / "prices_sample.json").read_text())
    assert written["series"]["AAPL"] == series


# ---------------------------------------------------------------------------
# Evaluate with no data
# ---------------------------------------------------------------------------


def test_evaluate_skips_without_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "pipeline.evaluation.engine.run_evaluation",
        side_effect=NoEvalDataError("No feature+return data assembled."),
    ):
        cli._evaluate(start="2020-01-01", end=None, data_repo=tmp_path, fdr_q=0.05)
    assert "Skipped" in capsys.readouterr().out


def test_load_prices_empty_archive(tmp_path: Path) -> None:
    assert load_prices(tmp_path).empty
    assert isinstance(load_prices(tmp_path), pd.DataFrame)

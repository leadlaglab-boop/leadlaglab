"""Correctness tests for the prediction ledger, built on a small synthetic archive.

Covers:
  - every (model, horizon) gets its own immutable ledger file
  - target windows span exactly `horizon` trading days
  - outcomes are scored against the window the prediction was about (no off-by-one)
  - training never sees returns realized after as_of
  - the momentum baseline gets a real trailing-return input
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pyarrow.parquet as pq
import pytest

from pipeline.evaluation.returns import (
    compute_excess_returns,
    compute_trailing_excess_returns,
)
from pipeline.features.builder import _write_features
from pipeline.ledger.scorer import score_predictions
from pipeline.ledger.writer import (
    _append_to_ledger,
    _target_window,
    load_ledger,
    write_daily_predictions,
)
from pipeline.sources.prices import _df_to_arrow as prices_to_arrow
from pipeline.universe.security_master import _df_to_arrow as sm_to_arrow
from pipeline.validate.schemas import SCHEMA_VERSION

TICKERS = [f"T{i:02d}" for i in range(12)]
NYSE = mcal.get_calendar("NYSE")


def _trading_days(start: str, end: str) -> list[date]:
    return [d.date() for d in NYSE.schedule(start_date=start, end_date=end).index]


def _write_security_master(repo: Path) -> None:
    df = pd.DataFrame(
        {
            "security_id": [f"sid_{t}" for t in TICKERS],
            "ticker": TICKERS,
            "valid_from": [date(2020, 1, 1)] * len(TICKERS),
            "valid_to": [None] * len(TICKERS),
            "name": TICKERS,
            "gics_sector": ["Tech"] * len(TICKERS),
            "cohort": ["sp500"] * len(TICKERS),
            "schema_version": [SCHEMA_VERSION] * len(TICKERS),
        }
    )
    out = repo / "processed" / "security_master"
    out.mkdir(parents=True)
    pq.write_table(sm_to_arrow(df), out / "security_master_2026-01-01.parquet")


def _write_price(repo: Path, ticker: str, d: date, price: float) -> None:
    df = pd.DataFrame(
        [
            {
                "security_id": f"sid_{ticker}",
                "date": d,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "adj_close": price,
                "volume": 1000,
                "observed_at": datetime(d.year, d.month, d.day, 21, tzinfo=UTC),
                "source": "test",
                "data_type": "backfilled",
                "schema_version": SCHEMA_VERSION,
            }
        ]
    )
    out = repo / "raw/prices" / f"year={d.year}" / f"month={d.month:02d}" / f"day={d.day:02d}"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(prices_to_arrow(df), out / f"{ticker.lower()}.parquet")


@pytest.fixture
def archive(tmp_path: Path) -> tuple[Path, list[date], dict[str, list[float]]]:
    """Security master, random-walk prices (+SPY) and features over ~3 months."""
    rng = np.random.default_rng(7)
    days = _trading_days("2026-05-01", "2026-08-31")
    _write_security_master(tmp_path)

    prices: dict[str, list[float]] = {}
    for t in [*TICKERS, "SPY"]:
        path = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        prices[t] = path.tolist()
        for d, p in zip(days, path, strict=True):
            _write_price(tmp_path, t, d, float(p))

    for d in days:
        _write_features(
            tmp_path,
            d,
            [
                {
                    "security_id": f"sid_{t}",
                    "date": d,
                    "feature_name": "wiki_views_z60d",
                    "value": float(v),
                    "rank_cs": 0.5,
                    "z_cs": float(v),
                    "n_obs": 40,
                    "shrunk_value": float(v),
                    "feature_version": "v1",
                    "schema_version": SCHEMA_VERSION,
                }
                for t, v in zip(TICKERS, rng.standard_normal(len(TICKERS)), strict=True)
            ],
        )
    return tmp_path, days, prices


# ---------------------------------------------------------------------------
# Target window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_target_window_spans_horizon_trading_days(horizon: int) -> None:
    as_of = date(2026, 11, 20)  # window crosses Thanksgiving
    start, end = _target_window(as_of, horizon)
    span = _trading_days(start.isoformat(), end.isoformat())
    assert len(span) == horizon
    assert start == _trading_days("2026-11-21", "2026-12-31")[0]


# ---------------------------------------------------------------------------
# Ledger partitioning
# ---------------------------------------------------------------------------


def _rows(model_id: str, horizon: int) -> list[dict[str, object]]:
    return [
        {
            "prediction_id": f"{model_id}-{horizon}",
            "made_at": datetime.now(UTC),
            "security_id": "sid_T00",
            "target_date_start": date(2026, 9, 1),
            "target_date_end": date(2026, 9, 1),
            "horizon_days": horizon,
            "model_id": model_id,
            "model_version": "v1",
            "data_hash": "h",
            "predicted_excess_return": 0.0,
            "predicted_direction": 0,
            "ci_lower": -0.01,
            "ci_upper": 0.01,
            "features_used_json": "[]",
            "schema_version": SCHEMA_VERSION,
        }
    ]


def test_each_model_horizon_gets_its_own_file(tmp_path: Path) -> None:
    as_of = date(2026, 8, 31)
    for model_id in ["zero", "momentum"]:
        for h in [1, 5]:
            _append_to_ledger(_rows(model_id, h), tmp_path, as_of)
    assert len(load_ledger(tmp_path)) == 4


def test_mixed_rows_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one model and horizon"):
        _append_to_ledger(_rows("zero", 1) + _rows("zero", 5), tmp_path, date(2026, 8, 31))


# ---------------------------------------------------------------------------
# Returns helpers
# ---------------------------------------------------------------------------


def test_prices_until_drops_unrealized_windows(archive: tuple[Path, list[date], dict]) -> None:
    repo, days, _ = archive
    cutoff = days[-10]
    rets = compute_excess_returns(repo, start=days[0], end=days[-1], prices_until=cutoff)
    ends = {
        (row.date, row.horizon): days[days.index(row.date) + row.horizon]
        for row in rets.itertuples()
    }
    assert max(ends.values()) <= cutoff


def test_trailing_excess_return_value(archive: tuple[Path, list[date], dict]) -> None:
    repo, days, prices = archive
    d = days[40]
    mom = compute_trailing_excess_returns(repo, start=d, end=d, window=21)
    got = mom[mom["ticker"] == "T03"]["trailing_excess_return"].iloc[0]
    i = days.index(d)
    expected = math.log(prices["T03"][i] / prices["T03"][i - 21]) - math.log(
        prices["SPY"][i] / prices["SPY"][i - 21]
    )
    assert got == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# End to end: write, then score
# ---------------------------------------------------------------------------


def test_writes_all_nine_model_horizon_files(archive: tuple[Path, list[date], dict]) -> None:
    repo, days, _ = archive
    as_of = days[-25]
    written = write_daily_predictions(repo, as_of=as_of)
    assert len(written) == 9
    ledger = load_ledger(repo)
    assert set(ledger["model_id"]) == {"zero", "momentum", "ridge_linear"}
    assert set(ledger["horizon_days"]) == {1, 5, 21}
    # Momentum is no longer the zero model in disguise
    mom = ledger[ledger["model_id"] == "momentum"]["predicted_excess_return"]
    assert mom.abs().sum() > 0
    # Re-running is a no-op, not an error
    assert write_daily_predictions(repo, as_of=as_of) == []


def test_scoring_uses_the_predicted_window(archive: tuple[Path, list[date], dict]) -> None:
    repo, days, prices = archive
    as_of = days[-25]
    write_daily_predictions(repo, as_of=as_of)

    outcomes = score_predictions(repo, as_of=days[-1])
    ledger = load_ledger(repo).merge(outcomes, on="prediction_id")
    row = ledger[
        (ledger["security_id"] == "sid_T05")
        & (ledger["horizon_days"] == 5)
        & (ledger["model_id"] == "zero")
    ].iloc[0]

    i = days.index(as_of)
    expected = math.log(prices["T05"][i + 5] / prices["T05"][i]) - math.log(
        prices["SPY"][i + 5] / prices["SPY"][i]
    )
    assert row["realized_excess_return"] == pytest.approx(expected, rel=1e-9)
    assert len(outcomes) == len(load_ledger(repo))  # every prediction scored

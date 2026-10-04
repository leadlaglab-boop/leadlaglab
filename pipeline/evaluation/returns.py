"""
Compute forward excess returns for the evaluation engine.

Excess return for security i over horizon h starting on day t+1:
  R(i, t, h) = log_cumret(i, t+1, t+h) - log_cumret(SPY, t+1, t+h)

Uses adj_close for all computations.  SPY is treated as the benchmark; it must
be present in the price data.  If adj_close is missing for a given row the
close is used as a fallback (flagged in the returned DataFrame).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import structlog

from pipeline.sources.prices import load_prices

log = structlog.get_logger()

HORIZONS = (1, 5, 21)  # trading days
BENCHMARK_TICKER = "SPY"


def _load_wide_adj_close(data_repo: Path, start: date, end: date) -> pd.DataFrame:
    """
    Return a (date × ticker) DataFrame of daily adj_close prices.
    Falls back to close if adj_close is null.
    end is extended by 25 calendar days to have enough forward data.
    """
    from datetime import timedelta

    extended_end = end + timedelta(days=35)  # enough for a 21d forward horizon
    raw = load_prices(data_repo, start=start, end=extended_end)
    if raw.empty:
        return pd.DataFrame()

    raw = raw.copy()
    raw["price"] = raw["adj_close"].fillna(raw["close"])
    raw["date"] = pd.to_datetime(raw["date"]).dt.date

    # Keep most-recent observation per (ticker, date) in case of duplicates
    raw = raw.sort_values("observed_at").drop_duplicates(["ticker", "date"], keep="last")

    pivot = raw.pivot(index="date", columns="ticker", values="price")
    pivot = pivot.sort_index()
    return pivot


def compute_excess_returns(
    data_repo: Path,
    start: date,
    end: date,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """
    Compute forward excess returns for every (ticker, date) pair in [start, end].

    Returns a DataFrame with columns:
        ticker, date, horizon, excess_return, benchmark_return, adj_close_used

    Look-ahead invariant: the return over days t+1..t+h is computed using
    close prices on those future days.  Features for date t are not touched here;
    the caller's join on feature_date ensures no leakage.
    """
    wide = _load_wide_adj_close(data_repo, start, end)
    if wide.empty:
        log.warning("returns.no_prices", start=str(start), end=str(end))
        return pd.DataFrame()

    if BENCHMARK_TICKER not in wide.columns:
        log.error(
            "returns.benchmark_missing",
            benchmark=BENCHMARK_TICKER,
            available=list(wide.columns)[:10],
        )
        raise ValueError(
            f"Benchmark ticker {BENCHMARK_TICKER} not found in price archive. "
            "Ensure SPY is ingested before running the evaluation."
        )

    # Daily log-returns for all tickers
    log_ret = cast("pd.DataFrame", np.log(wide / wide.shift(1)))  # shape (date, ticker)

    # Trading-day index (NYSE calendar)
    try:
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=wide.index.min().isoformat(),
            end_date=wide.index.max().isoformat(),
        )
        trading_days = sorted(d.date() if hasattr(d, "date") else d for d in schedule.index)
    except Exception:
        trading_days = sorted(wide.index.tolist())

    trading_day_pos = {d: i for i, d in enumerate(trading_days)}

    records: list[dict[str, Any]] = []
    for d in sorted(d for d in wide.index if start <= d <= end):
        pos = trading_day_pos.get(d)
        if pos is None:
            continue
        for h in horizons:
            end_pos = pos + h
            if end_pos >= len(trading_days):
                continue
            end_day = trading_days[end_pos]

            # Cumulative log-return from d+1 through end_day (inclusive)
            window = log_ret.loc[(log_ret.index > d) & (log_ret.index <= end_day)]
            if window.shape[0] < h:
                continue  # insufficient forward data

            cum_ret = window.sum()  # log-cumulative per ticker
            bench = cum_ret.get(BENCHMARK_TICKER)
            if bench is None or np.isnan(bench):
                continue

            for ticker in wide.columns:
                if ticker == BENCHMARK_TICKER:
                    continue
                r = cum_ret.get(ticker)
                if r is None or np.isnan(r):
                    continue
                records.append(
                    {
                        "ticker": ticker,
                        "date": d,
                        "horizon": h,
                        "excess_return": float(r - bench),
                        "benchmark_return": float(bench),
                    }
                )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    log.info(
        "returns.computed",
        rows=len(df),
        tickers=df["ticker"].nunique(),
        horizons=list(horizons),
        date_range=f"{df['date'].min()} – {df['date'].max()}",
    )
    return df

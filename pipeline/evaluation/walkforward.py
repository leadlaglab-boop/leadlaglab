"""
Walk-forward validation with embargo (purged cross-validation).

Configuration (from preregistration §6.3):
  - Training window expands from 2020-01-01
  - First test period starts 2021-01-01 (252 trading days min training)
  - Embargo = max(horizon, 5) trading days between train-end and test-start
    to prevent leakage from overlapping return windows

This module produces fold definitions only; callers run metrics per fold.
The engine uses these to build IC and Fama-MacBeth time series that represent
out-of-sample (OOS) performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import pandas_market_calendars as mcal
import structlog

log = structlog.get_logger()

_NYSE = mcal.get_calendar("NYSE")

TRAIN_START = date(2020, 1, 1)
TEST_START = date(2021, 1, 1)
MIN_TRAIN_DAYS = 252  # trading days


def _get_trading_days(start: date, end: date) -> list[date]:
    schedule = _NYSE.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    return sorted(d.date() if hasattr(d, "date") else d for d in schedule.index)


@dataclass
class Fold:
    fold_id: int
    train_start: date
    train_end: date
    test_start: date  # first test date (after embargo)
    test_end: date  # last test date
    embargo_days: int  # trading days between train_end and test_start (exclusive)


def generate_folds(
    all_dates: list[date],
    horizon: int,
    test_window_days: int = 63,  # ~1 quarter per fold
) -> list[Fold]:
    """
    Generate expanding-window folds with a per-horizon embargo.

    all_dates: sorted list of dates that have feature + return data.
    """
    embargo = max(horizon, 5)
    trading_days = _get_trading_days(min(all_dates), max(all_dates))

    folds: list[Fold] = []
    fold_id = 0

    # First test fold: test starts at TEST_START
    test_cursor = TEST_START

    while True:
        # Find train_end: embargo before test_cursor
        test_cursor_pos = next((i for i, d in enumerate(trading_days) if d >= test_cursor), None)
        if test_cursor_pos is None:
            break

        # train_end is embargo trading days before test_cursor
        train_end_pos = test_cursor_pos - embargo - 1
        if train_end_pos < 0:
            break
        train_end = trading_days[train_end_pos]

        # Check minimum training days
        train_dates_in_data = [d for d in all_dates if TRAIN_START <= d <= train_end]
        if len(train_dates_in_data) < MIN_TRAIN_DAYS:
            # Advance test cursor by one trading day
            test_cursor_pos += 1
            if test_cursor_pos >= len(trading_days):
                break
            test_cursor = trading_days[test_cursor_pos]
            continue

        # test window
        test_end_pos = min(test_cursor_pos + test_window_days - 1, len(trading_days) - 1)
        test_end = trading_days[test_end_pos]

        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=TRAIN_START,
                train_end=train_end,
                test_start=test_cursor,
                test_end=test_end,
                embargo_days=embargo,
            )
        )
        fold_id += 1

        # Advance to next test window
        next_test_pos = test_end_pos + 1
        if next_test_pos >= len(trading_days):
            break
        test_cursor = trading_days[next_test_pos]

    log.info(
        "walkforward.folds_generated",
        n_folds=len(folds),
        horizon=horizon,
        embargo=embargo,
    )
    return folds


def split_df(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a features+returns DataFrame into train and test sets for this fold."""
    df_date = pd.to_datetime(df["date"]).dt.date
    train_mask = (df_date >= fold.train_start) & (df_date <= fold.train_end)
    test_mask = (df_date >= fold.test_start) & (df_date <= fold.test_end)
    return df[train_mask].copy(), df[test_mask].copy()

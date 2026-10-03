"""
Leakage test suite.

Core invariant: features for date t must only use data with
observed_at < market_close(t)  (16:00 ET = cutoff).

These tests use Hypothesis to generate random date ranges and verify
that no future data ever influences past feature values.
"""

from __future__ import annotations

import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.utils.calendar import market_close_utc


# ---------------------------------------------------------------------------
# Invariant helpers
# ---------------------------------------------------------------------------


def assert_no_future_observed_at(
    feature_date: datetime.date,
    observed_at_values: list[datetime.datetime],
) -> None:
    """Assert that all observed_at timestamps precede market close on feature_date."""
    cutoff = market_close_utc(feature_date.isoformat())
    for ts in observed_at_values:
        # Normalize to UTC for comparison
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        assert ts < cutoff, (
            f"Look-ahead detected: observed_at={ts.isoformat()} is not before "
            f"market_close({feature_date.isoformat()})={cutoff.isoformat()}"
        )


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    feature_date=st.dates(
        min_value=datetime.date(2020, 1, 2),
        max_value=datetime.date(2030, 12, 31),
    ),
    hours_before_close=st.floats(min_value=0.001, max_value=72.0),
)
@settings(max_examples=500)
def test_data_before_cutoff_passes(
    feature_date: datetime.date, hours_before_close: float
) -> None:
    """Data collected strictly before market close must not trigger leakage detection."""
    cutoff = market_close_utc(feature_date.isoformat())
    observed = cutoff - datetime.timedelta(hours=hours_before_close)
    assert_no_future_observed_at(feature_date, [observed])  # must not raise


@given(
    feature_date=st.dates(
        min_value=datetime.date(2020, 1, 2),
        max_value=datetime.date(2030, 12, 31),
    ),
    hours_after_close=st.floats(min_value=0.001, max_value=72.0),
)
@settings(max_examples=500)
def test_data_after_cutoff_fails(
    feature_date: datetime.date, hours_after_close: float
) -> None:
    """Data collected after market close must be detected as look-ahead."""
    cutoff = market_close_utc(feature_date.isoformat())
    observed = cutoff + datetime.timedelta(hours=hours_after_close)
    with pytest.raises(AssertionError, match="Look-ahead detected"):
        assert_no_future_observed_at(feature_date, [observed])


# ---------------------------------------------------------------------------
# Unit tests for the calendar utility
# ---------------------------------------------------------------------------


def test_market_close_utc_is_aware() -> None:
    """market_close_utc must return a timezone-aware datetime."""
    close = market_close_utc("2024-01-02")
    assert close.tzinfo is not None


def test_market_close_utc_respects_dst() -> None:
    """Market close should be 21:00 UTC in summer (EDT) and 21:00 UTC in winter (EST).

    NYSE closes at 16:00 ET always:
      16:00 EST = 21:00 UTC  (UTC-5, winter)
      16:00 EDT = 20:00 UTC  (UTC-4, summer)
    """
    # Winter: 2024-01-02 (EST, UTC-5) → 21:00 UTC
    winter_close = market_close_utc("2024-01-02")
    assert winter_close.hour == 21, f"Expected 21:00 UTC in winter, got {winter_close.hour}:00"

    # Summer: 2024-07-01 (EDT, UTC-4) → 20:00 UTC
    summer_close = market_close_utc("2024-07-01")
    assert summer_close.hour == 20, f"Expected 20:00 UTC in summer, got {summer_close.hour}:00"

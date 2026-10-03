"""US market calendar utilities."""

from __future__ import annotations

import datetime

import pandas_market_calendars as mcal


_NYSE = mcal.get_calendar("NYSE")


def is_market_holiday(date_str: str) -> bool:
    """Return True if date_str (YYYY-MM-DD) is a US market holiday or weekend."""
    d = datetime.date.fromisoformat(date_str)
    if d.weekday() >= 5:
        return True
    schedule = _NYSE.schedule(
        start_date=date_str,
        end_date=date_str,
    )
    return schedule.empty


def market_close_utc(date_str: str) -> datetime.datetime:
    """Return 16:00 ET on the given date as a UTC-aware datetime."""
    d = datetime.date.fromisoformat(date_str)
    # NYSE closes at 16:00 ET; ET is UTC-5 (EST) or UTC-4 (EDT)
    # Use zoneinfo for correct DST handling
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    close_et = datetime.datetime(d.year, d.month, d.day, 16, 0, 0, tzinfo=et)
    return close_et.astimezone(datetime.timezone.utc)


def prev_trading_day(date_str: str) -> str:
    """Return the most recent trading day before date_str."""
    d = datetime.date.fromisoformat(date_str)
    candidate = d - datetime.timedelta(days=1)
    for _ in range(10):  # max 10 calendar days back (covers any holiday run)
        if not is_market_holiday(candidate.isoformat()):
            return candidate.isoformat()
        candidate -= datetime.timedelta(days=1)
    raise RuntimeError(f"Could not find a trading day in the 10 days before {date_str}")

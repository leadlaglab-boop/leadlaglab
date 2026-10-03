"""
yfinance — DEV-ONLY price source.

NEVER use this for the production archive or for any published data.
yfinance scrapes Yahoo Finance, which prohibits automated bulk access and
redistribution. See docs/DATA_SOURCES.md.

This module exists solely to:
  1. Verify the price pipeline end-to-end before Tiingo credentials are set up.
  2. Produce .tmp/ test fixtures for unit tests.

Records produced by this source are tagged data_type="dev_yfinance" and
are rejected by the archive writer's allowlist check.
"""

from __future__ import annotations

import warnings
from datetime import date

import structlog

log = structlog.get_logger()

_ALLOWED_DATA_TYPES = {"live", "backfilled"}  # archive write only allows these


def fetch_ticker_dev(
    ticker: str,
    security_id: str,
    start: date,
    end: date,
) -> list[dict]:
    """Fetch prices via yfinance for development verification ONLY."""
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except ImportError:
        log.error("yfinance not installed. Run: uv add --dev yfinance")
        return []

    warnings.filterwarnings("ignore")
    log.warning(
        "DEV-ONLY yfinance fetch — NOT for production archive",
        ticker=ticker,
    )

    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        log.warning("yfinance fetch failed", ticker=ticker, error=str(e))
        return []

    if df.empty:
        return []

    # yfinance returns MultiIndex columns when auto_adjust=True
    # Flatten them
    if hasattr(df.columns, "get_level_values"):
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    records = []
    for idx, row in df.iterrows():
        row_date = idx.date() if hasattr(idx, "date") else idx
        close = float(row.get("close", 0))
        if close <= 0:
            continue

        from pipeline.utils.calendar import market_close_utc

        observed_at = market_close_utc(row_date.isoformat())

        records.append(
            {
                "security_id": security_id,
                "date": row_date,
                "open": float(row.get("open") or 0) or None,
                "high": float(row.get("high") or 0) or None,
                "low": float(row.get("low") or 0) or None,
                "close": close,
                "adj_close": close,  # yfinance auto_adjust=True gives adj prices as "close"
                "volume": int(row.get("volume") or 0) or None,
                "observed_at": observed_at,
                "source": "yfinance_dev",
                "data_type": "dev_yfinance",  # intentionally not in _ALLOWED_DATA_TYPES
                "schema_version": "1.0.0",
            }
        )

    log.debug("yfinance_dev: fetched", ticker=ticker, rows=len(records))
    return records

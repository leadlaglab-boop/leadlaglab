"""
Tiingo price source.

Provides OHLCV + adjusted close via the Tiingo REST API.
Requires TIINGO_API_KEY in .env / GitHub Actions secrets.
Free tier: 500 requests/day, 50/hour.

Terms: https://www.tiingo.com/legal/terms-of-service
       Derived statistics only; raw price data may not be redistributed.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, date, datetime
from typing import Any

import pyarrow as pa
import requests
import structlog

from pipeline.sources.base import DateRange, SourcePlugin
from pipeline.utils.calendar import market_close_utc
from pipeline.validate.schemas import PRICES_DAILY_SCHEMA, SCHEMA_VERSION

log = structlog.get_logger()

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
DEFAULT_REQUESTS_PER_HOUR = 50


class TiingoPriceSource(SourcePlugin):
    name = "tiingo"
    terms_url = "https://www.tiingo.com/legal/terms-of-service"
    # Requests/hour comes from TIINGO_REQUESTS_PER_HOUR (default: free tier's 50/hour).
    # Raise it once on a paid tier; at 50/hour a full-universe run takes ~10 hours.
    rate_limit_per_second = DEFAULT_REQUESTS_PER_HOUR / 3600

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("TIINGO_API_KEY")
        per_hour = float(os.getenv("TIINGO_REQUESTS_PER_HOUR") or DEFAULT_REQUESTS_PER_HOUR)
        self.rate_limit_per_second = per_hour / 3600
        self._last_request_time: float = 0.0
        if not self._api_key:
            log.warning(
                "TIINGO_API_KEY not set; Tiingo source unavailable. "
                "Set it in .env or GitHub Actions secrets."
            )

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def _throttle(self) -> None:
        min_interval = 1.0 / self.rate_limit_per_second
        elapsed = time.monotonic() - self._last_request_time
        if self._last_request_time and elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def fetch_ticker(
        self,
        ticker: str,
        security_id: str,
        date_range: DateRange,
        data_type: str = "backfilled",
    ) -> list[dict[str, Any]]:
        if not self.available:
            log.debug("tiingo: skipped (no API key)", ticker=ticker)
            return []

        url = TIINGO_BASE.format(ticker=ticker.lower())
        params = {
            "startDate": date_range.start.isoformat(),
            "endDate": date_range.end.isoformat(),
            "resampleFreq": "daily",
            "token": self._api_key,
        }

        self._throttle()
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            log.warning("tiingo request failed", ticker=ticker, error=str(e))
            return []

        if resp.status_code == 404:
            log.debug("tiingo: ticker not found", ticker=ticker)
            return []

        if resp.status_code == 429:
            log.warning("tiingo: rate limited", ticker=ticker)
            return []

        if resp.status_code != 200:
            log.warning("tiingo: unexpected status", ticker=ticker, status=resp.status_code)
            return []

        try:
            data = resp.json()
        except Exception as e:
            log.warning("tiingo: json parse failed", ticker=ticker, error=str(e))
            return []

        if not isinstance(data, list):
            log.debug("tiingo: no data", ticker=ticker, response=str(data)[:200])
            return []

        fetch_time = datetime.now(tz=UTC)
        records = []
        for item in data:
            try:
                row_date = date.fromisoformat(item["date"][:10])
                close = float(item["close"])
                adj_close_raw = item.get("adjClose")
                adj_close = float(adj_close_raw) if adj_close_raw is not None else None

                if close <= 0:
                    continue

                observed_at = (
                    market_close_utc(row_date.isoformat())
                    if data_type == "backfilled"
                    else fetch_time
                )

                records.append(
                    {
                        "security_id": security_id,
                        "date": row_date,
                        "open": float(item.get("open") or 0) or None,
                        "high": float(item.get("high") or 0) or None,
                        "low": float(item.get("low") or 0) or None,
                        "close": close,
                        "adj_close": adj_close,
                        "volume": int(item["volume"]) if item.get("volume") else None,
                        "observed_at": observed_at,
                        "source": self.name,
                        "data_type": data_type,
                        "schema_version": SCHEMA_VERSION,
                    }
                )
            except (KeyError, ValueError, TypeError) as e:
                log.warning("tiingo: row parse error", ticker=ticker, error=str(e))
                continue

        log.debug("tiingo: fetched", ticker=ticker, rows=len(records), data_type=data_type)
        return records

    def fetch(self, date_range: DateRange) -> int:
        raise NotImplementedError("Use fetch_ticker() directly for price sources.")

    def health_check(self) -> bool:
        if not self.available:
            return False
        import datetime

        today = datetime.date.today()
        records = self.fetch_ticker(
            "AAPL",
            "test",
            DateRange(today - datetime.timedelta(days=7), today),
        )
        return len(records) > 0

    @property
    def schema(self) -> pa.Schema:
        return PRICES_DAILY_SCHEMA

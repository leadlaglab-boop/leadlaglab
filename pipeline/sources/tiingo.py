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
from datetime import date, datetime, timezone

import requests
import structlog

from pipeline.sources.base import DateRange, SourcePlugin
from pipeline.utils.calendar import market_close_utc
from pipeline.validate.schemas import PRICES_DAILY_SCHEMA, SCHEMA_VERSION

log = structlog.get_logger()

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"


class TiingoPriceSource(SourcePlugin):
    name = "tiingo"
    terms_url = "https://www.tiingo.com/legal/terms-of-service"
    rate_limit_per_second = 0.8  # stay under 50/hour = ~0.014/s; 0.8/s is fine within daily cap

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("TIINGO_API_KEY")
        if not self._api_key:
            log.warning(
                "TIINGO_API_KEY not set; Tiingo source unavailable. "
                "Set it in .env or GitHub Actions secrets."
            )

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def fetch_ticker(
        self,
        ticker: str,
        security_id: str,
        date_range: DateRange,
        data_type: str = "backfilled",
    ) -> list[dict]:
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

        fetch_time = datetime.now(tz=timezone.utc)
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
    def schema(self):  # type: ignore[override]
        return PRICES_DAILY_SCHEMA

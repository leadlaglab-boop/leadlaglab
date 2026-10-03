"""
Stooq price source.

Provides unadjusted OHLCV via CSV download. No API key required.
adj_close will be None for all records (Stooq does not publish adjusted prices).

Terms: https://stooq.com/ — derived statistics may be published; raw price files
       may not be redistributed. See docs/DATA_SOURCES.md.
"""

from __future__ import annotations

import io
import time
from datetime import UTC, date, datetime

import pandas as pd
import requests
import structlog

from pipeline.sources.base import DateRange, SourcePlugin
from pipeline.utils.calendar import market_close_utc
from pipeline.validate.schemas import PRICES_DAILY_SCHEMA, SCHEMA_VERSION

log = structlog.get_logger()

STOOQ_CSV_URL = "https://stooq.com/q/d/l/?s={ticker_lower}.us&d1={start}&d2={end}&i=d"
USER_AGENT = "LeadLagLab/1.0 (leadlaglab@gmail.com)"


class StooqPriceSource(SourcePlugin):
    name = "stooq"
    terms_url = "https://stooq.com/"
    rate_limit_per_second = 0.4  # 1 request every 2.5 seconds; conservative

    def __init__(self) -> None:
        self._last_request_time: float = 0.0

    def _throttle(self) -> None:
        min_interval = 1.0 / self.rate_limit_per_second
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def fetch_ticker(
        self,
        ticker: str,
        security_id: str,
        date_range: DateRange,
        data_type: str = "backfilled",
    ) -> list[dict]:
        """
        Fetch OHLCV for one ticker. Returns list of price record dicts.
        Returns empty list if ticker is not found on Stooq or on error.
        """
        self._throttle()

        url = STOOQ_CSV_URL.format(
            ticker_lower=ticker.lower().replace("-", "."),
            start=date_range.start.strftime("%Y%m%d"),
            end=date_range.end.strftime("%Y%m%d"),
        )

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=20,
            )
        except requests.RequestException as e:
            log.warning("stooq request failed", ticker=ticker, error=str(e))
            return []

        if resp.status_code == 404 or "No data" in resp.text or len(resp.content) < 50:
            log.debug("stooq: no data", ticker=ticker)
            return []

        if resp.status_code != 200:
            log.warning("stooq: unexpected status", ticker=ticker, status=resp.status_code)
            return []

        try:
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as e:
            log.warning("stooq: csv parse failed", ticker=ticker, error=str(e))
            return []

        # Stooq column names: Date, Open, High, Low, Close, Volume
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"date", "open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            log.warning("stooq: unexpected columns", ticker=ticker, cols=list(df.columns))
            return []

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["date", "close"])
        df = df[df["close"] > 0]

        fetch_time = datetime.now(tz=UTC)
        records = []
        for _, row in df.iterrows():
            row_date: date = row["date"]
            # For backfilled data, observed_at is market close of the data date
            # (reflecting when the data would have been available, not when we fetched it)
            if data_type == "backfilled":
                observed_at = market_close_utc(row_date.isoformat())
            else:
                observed_at = fetch_time

            records.append(
                {
                    "security_id": security_id,
                    "date": row_date,
                    "open": float(row.get("open") or 0) or None,
                    "high": float(row.get("high") or 0) or None,
                    "low": float(row.get("low") or 0) or None,
                    "close": float(row["close"]),
                    "adj_close": None,  # Stooq does not provide adjusted prices
                    "volume": int(float(row["volume"]))
                    if "volume" in row and pd.notna(row["volume"])
                    else None,
                    "observed_at": observed_at,
                    "source": self.name,
                    "data_type": data_type,
                    "schema_version": SCHEMA_VERSION,
                }
            )

        log.debug(
            "stooq: fetched",
            ticker=ticker,
            rows=len(records),
            data_type=data_type,
        )
        return records

    def fetch(self, date_range: DateRange) -> int:
        raise NotImplementedError("Use fetch_ticker() directly for price sources.")

    def health_check(self) -> bool:
        """Spot-check AAPL to verify Stooq is reachable and returning expected data."""
        import datetime

        from pipeline.sources.base import DateRange

        today = datetime.date.today()
        records = self.fetch_ticker(
            "AAPL",
            "test",
            DateRange(today - datetime.timedelta(days=7), today),
            data_type="backfilled",
        )
        return len(records) > 0

    @property
    def schema(self):  # type: ignore[override]
        return PRICES_DAILY_SCHEMA

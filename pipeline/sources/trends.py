"""
Google Trends source via pytrends.

UNOFFICIAL: pytrends scrapes the Google Trends UI. Google can change
its behavior at any time. Rate limiting is aggressive.

Terms: No public API; Google ToS prohibits automated scraping. We use this
       source only because it is widely used in academic research and the
       data is not redistributed. See docs/DATA_SOURCES.md.

Normalization: every batch includes the anchor term "stock market" so that
               indices are comparable across time and batch compositions.
               Without an anchor, Google Trends values are relative within
               the batch and change when the batch composition changes.

Caching: results are cached to disk (24h TTL) to reduce hammering Google.

Graceful degradation: if Google blocks us, the source logs a warning and
                      returns empty records. The pipeline continues without
                      failing the run.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import structlog

from pipeline.sources.base import DateRange, SourcePlugin
from pipeline.utils.calendar import market_close_utc
from pipeline.validate.schemas import SCHEMA_VERSION, SIGNAL_RAW_SCHEMA

log = structlog.get_logger()

ANCHOR_TERM = "stock market"
BATCH_SIZE = 4  # terms per request (1 slot reserved for anchor)
CACHE_TTL_HOURS = 24
BACKOFF_BASE_SECONDS = 60
MAX_RETRIES = 3


class TrendsSource(SourcePlugin):
    name = "google_trends"
    terms_url = "https://policies.google.com/terms"
    rate_limit_per_second = 0.1  # very conservative; ~1 req per 10s

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._last_req: float = 0.0

    def _cache_key(self, tickers: list[str], start: date, end: date) -> str:
        payload = f"{','.join(sorted(tickers))}|{start}|{end}"
        return hashlib.md5(payload.encode()).hexdigest()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if not self._cache_dir:
            return None
        path = self._cache_dir / f"trends_{key}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        age_hours = (time.time() - data["cached_at"]) / 3600
        if age_hours > CACHE_TTL_HOURS:
            path.unlink(missing_ok=True)
            return None
        return cast("dict[str, Any]", data["payload"])

    def _cache_set(self, key: str, payload: dict[str, Any]) -> None:
        if not self._cache_dir:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"trends_{key}.json"
        path.write_text(json.dumps({"cached_at": time.time(), "payload": payload}))

    def _throttle(self) -> None:
        gap = 1.0 / self.rate_limit_per_second
        elapsed = time.monotonic() - self._last_req
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_req = time.monotonic()

    def _fetch_batch_raw(
        self, tickers: list[str], start: date, end: date
    ) -> dict[str, dict[str, float]] | None:
        """
        Fetch Google Trends for a batch of tickers with anchor normalization.
        Returns {ticker: {date_str: relative_interest}} or None on failure.
        """
        # Check cache first
        key = self._cache_key(tickers, start, end)
        cached = self._cache_get(key)
        if cached:
            return cached

        try:
            from pytrends.request import TrendReq
        except ImportError:
            log.error("pytrends not installed")
            return None

        self._throttle()
        kw_list = [ANCHOR_TERM] + tickers[:BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                pt = TrendReq(hl="en-US", tz=-300, timeout=(10, 25))
                pt.build_payload(
                    kw_list,
                    cat=0,
                    timeframe=f"{start.isoformat()} {end.isoformat()}",
                    geo="US",
                )
                df = pt.interest_over_time()
                if df is None or df.empty:
                    return None

                anchor_values = df[ANCHOR_TERM].values
                result: dict[str, dict[str, float]] = {}

                for ticker in tickers:
                    if ticker not in df.columns:
                        continue
                    ticker_values = df[ticker].values
                    # Normalize: divide by anchor, scale to 0-100
                    normalized: dict[str, float] = {}
                    for i in range(len(df)):
                        anchor_v = float(anchor_values[i])
                        ticker_v = float(ticker_values[i])
                        norm_v = ticker_v / anchor_v * 100.0 if anchor_v > 0 else 0.0
                        date_str = df.index[i].strftime("%Y-%m-%d")
                        normalized[date_str] = round(norm_v, 4)
                    result[ticker] = normalized

                self._cache_set(key, result)
                return result

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "blocked" in err_str:
                    wait = BACKOFF_BASE_SECONDS * (2**attempt)
                    log.warning(
                        "google trends: rate limited",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                else:
                    log.warning("google trends: fetch failed", error=str(e)[:200])
                    return None

        log.warning("google trends: max retries exceeded")
        return None

    def fetch_tickers_batch(
        self,
        ticker_pairs: list[tuple[str, str]],  # [(ticker, security_id), ...]
        date_range: DateRange,
        data_type: str = "backfilled",
    ) -> list[dict[str, Any]]:
        """
        Fetch Google Trends for a batch of tickers.
        Batches into groups of BATCH_SIZE, retrying on rate limit.
        """
        tickers = [t for t, _ in ticker_pairs]
        security_ids = dict(ticker_pairs)

        batch_data = self._fetch_batch_raw(tickers, date_range.start, date_range.end)
        if not batch_data:
            log.warning(
                "google trends: no data for batch",
                tickers=tickers,
                note="this is logged but does not fail the pipeline",
            )
            return []

        records = []
        for ticker, daily in batch_data.items():
            security_id = security_ids.get(ticker, ticker)
            for date_str, value in sorted(daily.items()):
                try:
                    d = date.fromisoformat(date_str)
                except ValueError:
                    continue

                observed_at = (
                    market_close_utc(d.isoformat())
                    if data_type == "backfilled"
                    else datetime.now(tz=UTC)
                )

                records.append(
                    {
                        "source": self.name,
                        "security_id": security_id,
                        "effective_date": d,
                        "observed_at": observed_at,
                        "payload_json": json.dumps(
                            {
                                "anchor_normalized_interest": value,
                                "anchor_term": ANCHOR_TERM,
                                "note": "relative index 0–100; normalized by anchor",
                            }
                        ),
                        "n_items": 1,
                        "source_version": "v1",
                        "schema_version": SCHEMA_VERSION,
                    }
                )

        log.debug(
            "google trends: fetched",
            tickers=tickers,
            records=len(records),
            data_type=data_type,
        )
        return records

    def fetch(self, date_range: DateRange) -> int:
        raise NotImplementedError("Use fetch_tickers_batch() via the signal ingestor.")

    def health_check(self) -> bool:
        from datetime import date as d

        today = d.today()
        start = today - timedelta(days=7)
        result = self._fetch_batch_raw(["AAPL"], start, today)
        return result is not None and "AAPL" in result

    @property
    def schema(self) -> pa.Schema:
        return SIGNAL_RAW_SCHEMA

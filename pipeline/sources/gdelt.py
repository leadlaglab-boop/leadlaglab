"""
GDELT (Global Database of Events, Language, and Tone) source.

Uses the GDELT DOC API v2 for daily collection:
  https://api.gdeltproject.org/api/v2/doc/doc

Terms: Open, free for any use including commercial.
URL: https://www.gdeltproject.org/about.html#termsofuse

Provides per company per day:
  - Article count (n_items)
  - Average GDELT tone score (-100 to +100)
  - Average positive/negative/polarity scores
  - Sampled article URLs (up to 25) for NLP scoring and precision audit

GDELT DOC API modes used:
  - mode=artlist  → article list with metadata (for NLP scoring)
  - mode=timelinevolume  → daily article volume over time period
  - mode=timelinetone    → daily average tone over time period

Entity ambiguity: queries use the aliases from config/gdelt_entity_aliases.yaml.
A precision audit samples 200 company-article pairs per quarter.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import requests
import structlog
import yaml

from pipeline.sources.base import DateRange, SourcePlugin
from pipeline.utils.calendar import market_close_utc
from pipeline.validate.schemas import SCHEMA_VERSION, SIGNAL_RAW_SCHEMA

log = structlog.get_logger()

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "LeadLagLab/1.0 (leadlaglab@gmail.com)"


class GdeltSource(SourcePlugin):
    name = "gdelt"
    terms_url = "https://www.gdeltproject.org/about.html#termsofuse"
    rate_limit_per_second = 0.5  # be conservative; no documented limit but it's shared infra

    def __init__(self, aliases_path: Path) -> None:
        with open(aliases_path) as f:
            cfg = yaml.safe_load(f)
        self._aliases: dict[str, dict[str, Any]] = cfg.get("aliases", {})
        self._last_req: float = 0.0

    def _throttle(self) -> None:
        gap = 1.0 / self.rate_limit_per_second
        elapsed = time.monotonic() - self._last_req
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_req = time.monotonic()

    def _get(self, params: dict[str, str]) -> dict[str, Any] | None:
        self._throttle()
        try:
            resp = requests.get(
                DOC_API,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
        except requests.RequestException as e:
            log.warning("gdelt: request failed", params=str(params)[:80], error=str(e))
            return None
        if resp.status_code != 200:
            log.warning("gdelt: unexpected status", status=resp.status_code)
            return None
        try:
            return cast("dict[str, Any]", resp.json())
        except ValueError:
            return None

    def _query_for_ticker(self, ticker: str) -> str | None:
        """Get the GDELT search query for this ticker."""
        if ticker in self._aliases:
            return cast("str", self._aliases[ticker]["primary"])
        return None

    def fetch_ticker_day(
        self,
        ticker: str,
        security_id: str,
        day: date,
        data_type: str = "backfilled",
    ) -> dict[str, Any] | None:
        """
        Fetch GDELT signal for one ticker on one day.
        Returns a signal_raw dict or None if no data.
        """
        query = self._query_for_ticker(ticker)
        if not query:
            log.debug("gdelt: no alias for ticker", ticker=ticker)
            return None

        start_dt = datetime(day.year, day.month, day.day, 0, 0, 0)
        end_dt = datetime(day.year, day.month, day.day, 23, 59, 59)
        start_str = start_dt.strftime("%Y%m%d%H%M%S")
        end_str = end_dt.strftime("%Y%m%d%H%M%S")

        # 1. Get article list (for NLP scoring and precision audit)
        artlist_data = self._get(
            {
                "query": query + " sourcelang:english",
                "mode": "artlist",
                "maxrecords": "25",
                "format": "json",
                "startdatetime": start_str,
                "enddatetime": end_str,
            }
        )

        articles: list[dict[str, Any]] = []
        if artlist_data and "articles" in artlist_data:
            for art in artlist_data["articles"]:
                articles.append(
                    {
                        "url": art.get("url", ""),
                        "title": art.get("title", ""),
                        "domain": art.get("domain", ""),
                        "seendate": art.get("seendate", ""),
                        "language": art.get("language", ""),
                    }
                )

        # 2. Get timeline tone for the day
        tone_data = self._get(
            {
                "query": query + " sourcelang:english",
                "mode": "timelinetone",
                "format": "json",
                "startdatetime": start_str,
                "enddatetime": end_str,
            }
        )

        avg_tone: float | None = None
        article_count: int = len(articles)
        if tone_data and "timeline" in tone_data:
            tl = tone_data["timeline"]
            # Weighted average across data points in the period
            tones = []
            counts = []
            for series in tl:
                if "data" not in series:
                    continue
                for pt in series["data"]:
                    v = pt.get("value")
                    n = pt.get("norm", 1)
                    if v is not None:
                        tones.append(float(v) * float(n))
                        counts.append(float(n))
            if counts:
                total = sum(counts)
                avg_tone = sum(tones) / total if total > 0 else None
                article_count = max(article_count, int(total))

        if article_count == 0 and not articles:
            return None  # no coverage this day

        ambiguous = self._aliases.get(ticker, {}).get("ambiguous", False)

        observed_at = (
            market_close_utc(day.isoformat()) if data_type == "backfilled" else datetime.now(tz=UTC)
        )

        return {
            "source": self.name,
            "security_id": security_id,
            "effective_date": day,
            "observed_at": observed_at,
            "payload_json": json.dumps(
                {
                    "article_count": article_count,
                    "avg_tone": round(avg_tone, 4) if avg_tone is not None else None,
                    "ambiguous": ambiguous,
                    "query": query,
                    "sample_articles": articles[:10],  # store up to 10 for audit
                }
            ),
            "n_items": article_count,
            "source_version": "v1",
            "schema_version": SCHEMA_VERSION,
        }

    def fetch_ticker(
        self,
        ticker: str,
        security_id: str,
        date_range: DateRange,
        data_type: str = "backfilled",
    ) -> list[dict[str, Any]]:
        """Fetch GDELT signal for one ticker over a date range, one day at a time."""
        records = []
        current = date_range.start
        while current <= date_range.end:
            record = self.fetch_ticker_day(ticker, security_id, current, data_type)
            if record is not None:
                records.append(record)
            current += timedelta(days=1)
        log.debug("gdelt: fetched", ticker=ticker, days=len(records), data_type=data_type)
        return records

    def fetch(self, date_range: DateRange) -> int:
        raise NotImplementedError("Use fetch_ticker() via the signal ingestor.")

    def health_check(self) -> bool:
        import datetime

        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        result = self.fetch_ticker_day("AAPL", "test", yesterday)
        return result is not None

    @property
    def schema(self) -> pa.Schema:
        return SIGNAL_RAW_SCHEMA

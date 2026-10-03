"""
Wikipedia Pageviews source.

Official Wikimedia REST API — free, open access.
Terms: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
Attribution: "Wikipedia Pageviews API / Wikimedia Foundation"

Provides daily pageview counts per article. Company→article mapping is
maintained in config/wikipedia_article_map.yaml; unmapped tickers are
auto-resolved via the Wikipedia opensearch API and the result is cached.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
import structlog
import yaml

from pipeline.sources.base import DateRange, SourcePlugin
from pipeline.utils.calendar import market_close_utc
from pipeline.validate.schemas import SCHEMA_VERSION, SIGNAL_RAW_SCHEMA

log = structlog.get_logger()

PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/daily/{start}/{end}"
)
OPENSEARCH_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=opensearch&search={query}&limit=1&namespace=0&format=json"
)
USER_AGENT = "LeadLagLab/1.0 (leadlaglab@gmail.com)"


class WikipediaPageviewsSource(SourcePlugin):
    name = "wikipedia_pageviews"
    terms_url = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"
    rate_limit_per_second = 5.0

    def __init__(self, article_map_path: Path, lookup_cache_path: Path | None = None) -> None:
        with open(article_map_path) as f:
            cfg = yaml.safe_load(f)
        self._overrides: dict[str, dict] = cfg.get("overrides", {})
        self._lookup_cache_path = lookup_cache_path
        self._lookup_cache: dict[str, str | None] = {}
        if lookup_cache_path and lookup_cache_path.exists():
            self._lookup_cache = json.loads(lookup_cache_path.read_text())
        self._last_req: float = 0.0

    def _throttle(self) -> None:
        min_gap = 1.0 / self.rate_limit_per_second
        elapsed = time.monotonic() - self._last_req
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        self._last_req = time.monotonic()

    def _get(self, url: str) -> requests.Response | None:
        self._throttle()
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            return resp
        except requests.RequestException as e:
            log.warning("wikipedia: request failed", url=url[:80], error=str(e))
            return None

    def get_articles(self, ticker: str, name: str) -> list[str]:
        """Return list of Wikipedia article titles to sum for this ticker."""
        if ticker in self._overrides:
            return self._overrides[ticker].get("articles", [name])
        # Auto-lookup via opensearch
        if ticker not in self._lookup_cache:
            self._lookup_cache[ticker] = self._opensearch(name)
            self._save_cache()
        found = self._lookup_cache[ticker]
        return [found] if found else []

    def _opensearch(self, query: str) -> str | None:
        resp = self._get(OPENSEARCH_URL.format(query=quote(query)))
        if resp is None or resp.status_code != 200:
            return None
        try:
            data = resp.json()
            titles = data[1]
            return titles[0] if titles else None
        except (IndexError, ValueError):
            return None

    def _save_cache(self) -> None:
        if self._lookup_cache_path:
            self._lookup_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._lookup_cache_path.write_text(json.dumps(self._lookup_cache, indent=2))

    def fetch_ticker(
        self,
        ticker: str,
        security_id: str,
        name: str,
        date_range: DateRange,
        data_type: str = "backfilled",
    ) -> list[dict[str, Any]]:
        """Fetch daily pageviews for one ticker. Returns list of signal_raw dicts."""
        articles = self.get_articles(ticker, name)
        if not articles:
            log.debug("wikipedia: no article mapping", ticker=ticker)
            return []

        # Aggregate pageviews across all articles for this company
        daily_views: dict[date, int] = {}
        for article in articles:
            url = PAGEVIEWS_URL.format(
                article=quote(article, safe=""),
                start=date_range.start.strftime("%Y%m%d"),
                end=date_range.end.strftime("%Y%m%d"),
            )
            resp = self._get(url)
            if resp is None or resp.status_code == 404:
                log.debug("wikipedia: article not found", article=article)
                continue
            if resp.status_code != 200:
                log.warning(
                    "wikipedia: unexpected status", article=article, status=resp.status_code
                )
                continue
            try:
                data = resp.json()
            except ValueError:
                continue

            for item in data.get("items", []):
                try:
                    d = date(
                        int(item["timestamp"][:4]),
                        int(item["timestamp"][4:6]),
                        int(item["timestamp"][6:8]),
                    )
                    daily_views[d] = daily_views.get(d, 0) + int(item["views"])
                except (KeyError, ValueError, IndexError):
                    continue

        records = []
        for d, views in sorted(daily_views.items()):
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
                            "views": views,
                            "articles": articles,
                            "ambiguous": self._overrides.get(ticker, {}).get("ambiguous", False),
                        }
                    ),
                    "n_items": len(articles),
                    "source_version": "v1",
                    "schema_version": SCHEMA_VERSION,
                }
            )

        log.debug("wikipedia: fetched", ticker=ticker, days=len(records), data_type=data_type)
        return records

    def fetch(self, date_range: DateRange) -> int:
        raise NotImplementedError("Use fetch_ticker() via the signal ingestor.")

    def health_check(self) -> bool:
        import datetime

        end = datetime.date.today()
        start = end - datetime.timedelta(days=3)
        records = self.fetch_ticker("AAPL", "test", "Apple Inc.", DateRange(start, end))
        return len(records) > 0

    @property
    def schema(self):  # type: ignore[override]
        return SIGNAL_RAW_SCHEMA

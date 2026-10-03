# Data Sources

_Last updated: 2026-10-03_

This document records every data source used by Lead/Lag Lab: its URL, terms/license link, what we store, what we publish, rate limits, and verification status. It is updated whenever a source is added, removed, or when terms change.

---

## Clearance key

| Status | Meaning |
|---|---|
| ✅ CLEARED | Terms verified; use is permitted as described |
| ⚠️ RESTRICTED | Permitted with constraints (noted below) |
| ❌ BLOCKED | Terms prohibit our use; source omitted |
| 🔍 PENDING | Terms unclear or not yet fully reviewed |

---

## Price data

### Stooq (primary)
- **URL:** https://stooq.com
- **Terms:** https://stooq.com/  _(free data downloads; no explicit redistribution prohibition for derived statistics)_
- **Status:** ⚠️ RESTRICTED
- **What we store:** Daily OHLCV (adj. + raw close). Stored in private archive only.
- **What we publish:** Derived statistics (returns, volatility) and charts. **Not** raw price files.
- **Rate limit:** Download via CSV; no formal API. Throttle to 1 req/3s and cache aggressively.
- **Notes:** Terms do not explicitly permit redistribution of raw price data, so we publish only derived statistics and interactive charts. Raw files stay in the private archive.

### Tiingo (fallback)
- **URL:** https://www.tiingo.com
- **Terms:** https://www.tiingo.com/legal/terms-of-service
- **Status:** ⚠️ RESTRICTED
- **What we store:** OHLCV + adjusted factors. Private archive only.
- **What we publish:** Derived statistics and charts only; Tiingo terms prohibit redistribution of raw data.
- **Rate limit:** Free tier: 500 requests/day, 50 req/hour.
- **Notes:** Requires `TIINGO_API_KEY` in `.env`. Free tier sufficient for daily runs.

### Alpha Vantage (last resort)
- **URL:** https://www.alphavantage.co
- **Terms:** https://www.alphavantage.co/terms_of_service/
- **Status:** ⚠️ RESTRICTED
- **What we store:** Daily OHLCV. Private archive only.
- **What we publish:** Derived statistics only. Terms prohibit bulk redistribution of raw data.
- **Rate limit:** Free tier: 25 req/day, 5 req/minute.
- **Notes:** Only used if Stooq and Tiingo both fail. Too rate-limited for production use.

### yfinance
- **Status:** ❌ BLOCKED for published data
- **Notes:** yfinance scrapes Yahoo Finance, which explicitly prohibits automated access and redistribution in its terms. Allowed only as a private fallback for personal verification. Never used as the basis for published data or bulk archiving.

---

## News tone and volume

### GDELT (Global Database of Events, Language, and Tone)
- **URL:** https://www.gdeltproject.org
- **Terms:** https://www.gdeltproject.org/about.html#termsofuse _(open, free for any use including commercial)_
- **Status:** ✅ CLEARED
- **What we store:** GKG records: source URLs, article counts, tone scores, themes. Stored partitioned by date.
- **What we publish:** Aggregated company-level tone scores and article counts. No raw article text.
- **Rate limit:** No formal rate limit; be a good citizen: batch downloads, don't hammer continuously.
- **Notes:**
  - Company entity matching uses name/alias dictionaries. Ambiguous names (e.g., "Target", "Block", "Apple") are handled with documented disambiguation rules (see `pipeline/sources/gdelt/entity_map.yaml`).
  - A precision audit of 200 randomly sampled company-article pairs per quarter is published on the site.

---

## Search interest

### Google Trends (via pytrends)
- **URL:** https://trends.google.com
- **Terms:** https://policies.google.com/terms _(unofficial client; no public API terms)_
- **Status:** ⚠️ RESTRICTED
- **What we store:** Relative search interest indices (0–100), batched with anchor term `"stock market"` for normalization. Cached with 24h TTL.
- **What we publish:** Relative indices and z-scores. Values are Google's "relative interest" numbers, not absolute queries.
- **Rate limit:** Unofficial; Google blocks aggressive scrapers. We retry with exponential backoff (base 60s) and degrade gracefully when blocked. A failed fetch is logged and skipped, not retried in a loop.
- **Notes:**
  - pytrends is an unofficial library. Google can change its behavior at any time.
  - All published values are labeled "Source: Google Trends (unofficial)" with the caveat that sampling noise affects comparability across time.
  - Historical data is labeled "backfilled" and the known limitations (sampling variability, index rescaling) are documented on the Methodology page.

---

## Public attention

### Wikipedia Pageviews API
- **URL:** https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews
- **Terms:** https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use _(free, open access; attribution required)_
- **Status:** ✅ CLEARED
- **What we store:** Daily pageview counts per article, with `observed_at` timestamp.
- **What we publish:** Pageview counts and derived z-scores. Attribution: "Wikipedia Pageviews API / Wikimedia Foundation."
- **Rate limit:** 100 req/s per the API docs; we stay well below at <5 req/s.
- **Notes:**
  - Company → article mapping maintained in `pipeline/sources/wikipedia/article_map.yaml`.
  - One company may map to multiple articles (e.g., "Tesla, Inc." + "Tesla Autopilot"); these are summed and documented.

---

## Retail social sentiment

### StockTwits
- **URL:** https://stocktwits.com
- **Terms:** https://stocktwits.com/developers/docs/developer_policy _(Public API deprecated; terms changed in 2023)_
- **Status:** 🔍 PENDING — **not ingesting until terms are verified**
- **Notes:** Current API access requires application approval and terms review. As of 2026-10-03, bulk automated ingestion is not clearly permitted. Flagged for manual review. If terms prohibit it, this source will remain in the BLOCKED state and is noted as omitted.

### Reddit (r/wallstreetbets and similar)
- **URL:** https://reddit.com
- **Terms:** https://www.reddit.com/wiki/api/ _(OAuth API required; commercial data use requires Reddit Data API license agreement)_
- **Status:** ❌ BLOCKED
- **Notes:** Reddit's 2023 API policy changes require a paid Data API license for systematic data collection beyond personal use. We do not have this license. Source is omitted. Documented here for transparency.

---

## SEC EDGAR

### Form 4, 8-K, 13F-HR
- **URL:** https://www.sec.gov/developer
- **Terms:** https://www.sec.gov/privacy.htm _(public government data; free for any use)_
- **Status:** ✅ CLEARED
- **What we store:** Filing acceptance timestamps, form type, filer CIK, issuer CIK, transaction details (Form 4). Raw XML stored in archive.
- **What we publish:** Aggregated filing counts, insider buy/sell ratios, and 8-K frequency. Filing acceptance timestamps give clean `observed_at` values.
- **Rate limit:** 10 requests/second per SEC policy. We use 8 req/s to be safe.
- **User-Agent:** `"LeadLagLab leadlaglab@gmail.com"` — required by SEC; set in all requests.
- **Notes:** EDGAR filing dates are the canonical `observed_at` for this source.

---

## Macro controls

### FRED / ALFRED (Federal Reserve Economic Data)
- **URL:** https://fred.stlouisfed.org / https://alfred.stlouisfed.org
- **Terms:** https://fred.stlouisfed.org/legal _(free; attribution required; redistribution of data permitted for non-commercial research)_
- **Status:** ✅ CLEARED
- **What we store:** Vintages via ALFRED API — each observation stored with `vintage_date` so we always reconstruct what was known at any past date. Controls only (not signals).
- **What we publish:** Derived control variable values used in regression tables. Attribution: "Federal Reserve Bank of St. Louis (FRED/ALFRED)."
- **Rate limit:** 120 requests/minute with a free API key; public endpoint rate-limited without key.

---

## Revision history

| Date | Change |
|---|---|
| 2026-10-03 | Initial document; terms checks completed for Stooq, GDELT, Wikipedia, EDGAR, FRED. StockTwits PENDING. Reddit BLOCKED. yfinance BLOCKED for published data. |

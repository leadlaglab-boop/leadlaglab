# Lead/Lag Lab — Architecture & Build Plan

_Last updated: 2026-10-03_

---

## 1. What this is

A public, static research site that archives daily alternative-data signals across the S&P 500 and a retail-attention basket, runs honest out-of-sample tests of whether those signals predict subsequent excess returns, and maintains a frozen, timestamped prediction ledger scored as outcomes arrive. It is a portfolio piece and the seed of a real study — not investment advice.

---

## 2. Repository layout

Two GitHub repos:

| Repo | Purpose |
|---|---|
| `leadlaglab` | Pipeline code, site, CI, docs |
| `leadlaglab-data` | Append-only raw archive (Parquet); separate so it can be moved to R2 without touching the code repo) |

```
leadlaglab/
  pipeline/
    sources/          # one plugin per data source
    features/         # feature construction
    models/           # baselines, LightGBM
    evaluation/       # IC, quintiles, Fama-MacBeth, walk-forward
    ledger/           # daily prediction + scoring
    validate/         # data contracts, leakage tests
  config/
    universe_sp500.yaml
    universe_retail.yaml
    sources.yaml
    site.yaml         # SITE_NAME and other site-level config
  study/
    PREREGISTRATION.md
    AMENDMENTS.md
    notebooks/
  site/               # Astro static site
    public/
      data/           # generated JSON/Parquet + manifest.json
      fonts/          # self-hosted
    src/
      components/
      layouts/
      pages/
  docs/
    PLAN.md           (this file)
    DATA_SOURCES.md
    METHODOLOGY.md
    SECURITY.md
  tests/
  .github/
    workflows/
```

---

## 3. Technology choices (with rationale)

### Python pipeline
- **Python 3.12** — required for pattern matching, better typing. Install via `uv` (see §6).
- **uv** — faster than pip/poetry; lockfile reproducibility; manages Python versions too.
- **polars** — columnar, fast, lazy evaluation. Better than pandas for large Parquet files.
- **pyarrow** — Parquet read/write and schema enforcement.
- **statsmodels** — Fama-MacBeth, Newey-West standard errors.
- **scikit-learn** — cross-validation utilities, calibration.
- **lightgbm** — gradient boosting (Phase 2, only if it earns its place over baselines).
- **pydantic v2** — schema validation for data contracts.
- **pandera** — runtime DataFrame validation (schema, ranges, freshness checks).
- **transformers + ProsusAI/finbert** — FinBERT for news tone; vaderSentiment as baseline.
- **pytrends** — Google Trends (unofficial; aggressive caching + graceful degradation).
- **pandas_market_calendars** — US market holiday handling.
- **ruff** — linter (replaces flake8 + isort + more).
- **mypy** — static type checking.
- **pytest** — unit + property tests.
- **hypothesis** — property-based testing (leakage checks).
- **gitleaks** — secret scanning pre-commit hook.

### Frontend
- **Astro 5** — static site generator; zero JS by default; excellent for content-heavy sites.
- **TypeScript** — all site code.
- **Observable Plot** — chosen over Apache ECharts because:
  - ~50 KB vs ~1 MB minified; critical for Lighthouse ≥ 95 goal
  - Designed specifically for analytical/statistical data; better small multiples
  - Composable grammar-of-graphics approach fits the "editorial" aesthetic
  - Better TypeScript types
  - Authored by Mike Bostock (D3 creator); financial data is its home turf
  - ECharts would be chosen instead only if we need its specific interactivity patterns
- **Inter + Source Serif 4 + JetBrains Mono** — self-hosted via `fontsource` or downloaded static subsets
- **CSS custom properties** — full light/dark theme, no CSS framework (keeps bundle tiny)
- **Fuse.js** — lightweight client-side fuzzy search for the Stocks index page

### Infrastructure
- **GitHub Actions** — daily cron after US close (~18:30 ET) + morning catch-up; idempotent steps.
- **Cloudflare Pages** — static hosting; auto-deploy on push to `main`.
- **Cloudflare Web Analytics** — privacy-friendly, no cookies, no banner required.
- **R2** (upgrade path) — if raw archive exceeds ~800 MB in git; monitor and alert at 600 MB.

---

## 4. Data contracts (schemas)

All defined in `pipeline/validate/schemas.py` using Pydantic + pyarrow. Schema version is embedded in every file and in `manifest.json`.

| Schema | Key fields |
|---|---|
| `SecurityMaster` | `security_id, ticker, valid_from, valid_to, name, gics_sector, cohort` |
| `PricesDaily` | `security_id, date, open, high, low, close, adj_close, volume, observed_at, source` |
| `SignalRaw` | `source, security_id, effective_date, observed_at, payload, n_items, source_version` |
| `SignalFeature` | `security_id, date, feature_name, value, rank_cs, z_cs, n_obs, shrunk_value, feature_version` |
| `PredictionLedger` | `prediction_id, made_at, security_id, target_date_start/end, horizon_days, model_id, model_version, data_hash, predicted_excess_return, predicted_direction, confidence_interval, features_used` |
| `Outcome` | `prediction_id, realized_excess_return, scored_at` |

Targets: excess return vs SPY and vs sector ETF, at 1, 5, 21 trading days. Not price levels.

---

## 5. No-look-ahead enforcement

Every `SignalFeature` row is built with this invariant:

> Feature for date `t` may only use data where `observed_at < market_close(t)` (16:00 ET).

The leakage test suite (`tests/test_leakage.py`) uses Hypothesis to generate random date ranges and asserts that shifting future `observed_at` values never changes past features. This runs in CI and blocks the build on failure.

FRED series: fetched via ALFRED vintage API, storing `vintage_date` so we always reconstruct what was known at the time.

---

## 6. Milestones

### M1 — Foundations (current)
- [ ] Environment setup (Python 3.12, uv, gh, wrangler, gitleaks via Homebrew or curl)
- [ ] `leadlaglab` repo: scaffold directories, `pyproject.toml`, `package.json`, `.gitignore`
- [ ] `leadlaglab-data` repo: scaffold
- [ ] `config/site.yaml` with `SITE_NAME: "Lead/Lag Lab"`
- [ ] CI skeleton: lint + type check + test on every push
- [ ] Secret scanner: gitleaks pre-commit hook
- [ ] Security baseline: Dependabot, pip-audit, npm audit in CI
- [ ] `docs/DATA_SOURCES.md`: terms check for every candidate source
- [ ] `docs/SECURITY.md`: headers policy, CSP plan
- [ ] Commit, tag `v0.1.0-foundations`

### M2 — Universe + prices
- [ ] S&P 500 point-in-time membership from Wikipedia "selected changes"
- [ ] Retail-attention basket from `config/universe_retail.yaml`
- [ ] SecurityMaster table with stable `security_id`, ticker history, GICS sector
- [ ] Ticker change handling (e.g., SQ → XYZ for Block)
- [ ] Price ingestion plugin (Stooq first; Tiingo as fallback if terms allow)
- [ ] Adjusted close + raw close; splits/dividends handled
- [ ] First published JSON: `site/public/data/universe.json`, `prices_sample.json`
- [ ] Pandera validation: schema, ranges, freshness

### M3 — Signal collectors v1
- [ ] Wikipedia pageviews (official API; company→article mapping)
- [ ] GDELT GKG news tone + volume (name/alias dictionary; ambiguity audit)
- [ ] SEC EDGAR Form 4, 8-K counts, 13F (User-Agent with contact email)
- [ ] Google Trends via pytrends (anchor-term normalization, caching, graceful degradation)
- [ ] StockTwits / Reddit — only if terms permit (see DATA_SOURCES.md; otherwise document as "omitted")
- [ ] FinBERT + VADER scoring of GDELT text
- [ ] Raw archive running daily; backfills labeled
- [ ] Precision audit scaffold (200-sample entity matching)

### M4 — Features + pre-registration
- [ ] Feature pipeline: 60-day z-score, cross-sectional rank, minimum-count rules, shrinkage
- [ ] Leakage test suite passes
- [ ] Draft `study/PREREGISTRATION.md` for review
- [ ] **STOP: user reviews and approves pre-registration before it is committed**
- [ ] Pre-registration committed and locked; any future changes go to AMENDMENTS.md

### M5 — Evaluation engine
- [ ] IC (Spearman) per date with Newey-West SEs
- [ ] Quintile long-short spreads with transaction-cost assumption
- [ ] Hit rate with binomial CI
- [ ] Fama-MacBeth regressions with controls
- [ ] Walk-forward with embargo (purged CV); no random splits
- [ ] BH FDR correction; trial count documented
- [ ] Results written to `study/results/` and `site/public/data/`
- [ ] "Time until adequately powered" estimate

### M6 — Models + prediction ledger
- [ ] Baseline models: naive zero, momentum, AR(p), sector-mean
- [ ] Daily frozen predictions (committed before outcomes are known)
- [ ] Automatic outcome scoring
- [ ] LightGBM — only if it beats baselines out-of-sample; documented decision

### M7 — Site build
- [ ] Design system: CSS variables, typography, color (light + dark)
- [ ] All 8 pages built
- [ ] Observable Plot charts: IC over time, quintile returns, correlation by lag, calibration
- [ ] Accessibility: WCAG AA, keyboard nav, data tables, reduced-motion
- [ ] Performance: Lighthouse ≥ 95 mobile; no CLS; lazy chart loading
- [ ] Open-graph images at build time
- [ ] Sitemap, RSS/Atom changelog feed
- [ ] Downloadable study data page

### M8 — Domain + launch
- [ ] Confirm domain registrar; provide DNS/nameserver steps (preserving MX/SPF/DKIM)
- [ ] Cloudflare Pages setup + auto-deploy
- [ ] Security headers in `_headers`: HSTS, CSP, Referrer-Policy, X-Content-Type-Options, Permissions-Policy
- [ ] Full launch checklist

### M9 — Operate
- [ ] Failed-run → GitHub Issue automation
- [ ] Archive size monitor (alert at 600 MB)
- [ ] Monthly "state of the study" auto-generated report
- [ ] Dependabot + pip-audit running weekly

---

## 7. Open questions (pending your answers)

| # | Question | Needed for |
|---|---|---|
| Q1 | Contact email at leadlaglab.com? (e.g. `hello@leadlaglab.com`) | SEC EDGAR User-Agent, About page, data-provider signups |
| Q2 | Where is leadlaglab.com registered? (Cloudflare, Namecheap, Google Domains, etc.) | M8 DNS steps; don't touch MX/SPF/DKIM |
| Q3 | Confirm you want raw archive in a separate GitHub repo `leadlaglab-data` (not R2 to start) | M1 repo setup |
| Q4 | GitHub username / org to create repos under? | M1 repo setup |
| Q5 | Do you have a Cloudflare account, or should I walk you through creating one? | M8 |
| Q6 | Preferred Tiingo API key tier (free = 500 req/day)? | M2 price plugin |

---

## 8. Cost estimate (pre-launch, recurring)

| Item | Cost |
|---|---|
| Cloudflare Pages | Free |
| Cloudflare Web Analytics | Free |
| Cloudflare R2 (if needed) | ~$0.015/GB/month + egress; $0 up to 10 GB |
| GitHub | Free (public repos) |
| All data sources in M3 | Free (terms permitting) |
| FinBERT inference | Free (CPU in CI; ~5–10 min/day) |
| Domain renewal | ~$10–15/year |

Total operating cost before R2 upgrade: **~$0/month**.

---

## 9. Key risks and mitigations

| Risk | Mitigation |
|---|---|
| pytrends blocked / rate-limited | Cache aggressively; exponential backoff; degrade gracefully; daily run frequency is low |
| StockTwits/Reddit terms change or prohibit | Terms-checked before any ingestion; omitted with clear documentation if unclear |
| S&P 500 point-in-time membership incomplete | Wikipedia "selected changes" table; document gaps; label as best-effort |
| Raw archive outgrows git | Monitor size in CI; migrate to R2 with one-time migration script at 800 MB |
| FinBERT too slow in CI | Batch inference; cache embeddings; only re-run on new text |
| Entity matching precision | 200-sample audit per source; precision published on site; ambiguous names documented in DATA_SOURCES.md |
| No signal found | Explicitly planned for; null results are the intended content |

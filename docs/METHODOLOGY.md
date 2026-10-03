# Lead/Lag Lab — Methodology

_This document describes the study design, data construction, statistical methods,
and known limitations of Lead/Lag Lab. It is the human-readable companion to
`study/PREREGISTRATION.md`, which is the binding methodological record._

---

## 1. Research Question

Does publicly available alternative data — Wikipedia pageviews, news tone from
GDELT, Google Trends search interest, and SEC filing activity — lead short-term
excess stock returns for S&P 500 companies and a "retail-attention" basket?

"Lead" means: does observing signal *s* for company *c* at time *t* improve our
ability to forecast the excess return of *c* over the next 1, 5, or 21 trading
days, over and above a set of naive baselines?

This is a research question, not a trading recommendation.

---

## 2. Universe

### 2.1 S&P 500 (primary cohort)

- Current S&P 500 constituents as of the study start date (2026-10-03).
- Point-in-time membership: built from Wikipedia's "S&P 500 selected changes"
  table. This introduces some survivorship bias and membership-date imprecision;
  both are documented in `docs/DATA_SOURCES.md`.
- Sector: GICS sector at time of observation, from the security master.

### 2.2 Retail-attention basket (separate cohort)

Fifteen tickers with elevated retail trading activity at study inception:
GME, AMC, TSLA, PLTR, SOFI, HOOD, COIN, MSTR, RIVN, LCID, IONQ, RGTI, QBTS,
RKLB, SNAP.

**Inclusion rule (pre-specified):** tickers showing elevated short interest
(>10% float) or elevated social media mention volume relative to market cap at
study inception, as documented in `config/universe_retail.yaml`.

**Why separate:** alt-data signals often behave differently for retail-driven
stocks. Analyzing them together with the S&P 500 would mask this heterogeneity.
All results report the two cohorts separately; aggregate statistics are never
blended.

### 2.3 Ticker changes and delistings

Each security has a stable internal `security_id`. Ticker changes (e.g.,
Block: SQ → XYZ) are tracked in the security master with `valid_from`/`valid_to`
dates. Returns use the post-change ticker for continuity; the mapping is logged.

---

## 3. Target Variable

**Excess return** over SPY (S&P 500 ETF) at horizons of 1, 5, and 21 trading
days, computed from adjusted closing prices:

```
excess_return(c, t, h) = adj_return(c, t, t+h) - adj_return(SPY, t, t+h)
```

where `adj_return` is the log return of adjusted closing prices.

Raw close prices are also stored (for reference and split/dividend auditing).
Only adjusted returns are used as prediction targets.

---

## 4. Signal Construction

### 4.1 Raw data → features, step by step

For each source and each trading date *t*:

1. **Observe:** collect the raw signal value for security *c* with
   `observed_at` timestamp (when we fetched it, ≥ 16:00 ET on date *t*).
2. **Trailing z-score:** compute the 60-day rolling mean and standard deviation
   of the raw signal for security *c*. The z-score is the deviation from the
   trailing mean divided by the trailing standard deviation. This controls for
   each security's own baseline level and volatility.
3. **Cross-sectional rank:** within each date and sector, rank the z-scores from
   0 to 1. This creates comparability across sectors and prevents any single
   sector from dominating the signal.
4. **Shrinkage:** for message-based signals with low sample counts (e.g., a
   company with few articles on a given day), apply empirical-Bayes shrinkage
   toward the cross-sectional mean to reduce noise inflation.

### 4.2 No-look-ahead guarantee

Every record stores `observed_at` (when we collected it). Features for decision
date *t* may only use data with `observed_at` before 16:00 ET on date *t*. The
leakage test suite (run in CI) fails the build if any feature uses data from
after the decision time.

### 4.3 Signal families

| Feature name | Source | Description |
|---|---|---|
| `wiki_pageviews` | Wikipedia Pageviews API | Trailing z-score of daily pageviews for the company's Wikipedia article |
| `gdelt_tone` | GDELT GKG | 60-day z-score of average article tone (−100 to +100) for articles mentioning the company |
| `gdelt_count` | GDELT GKG | z-score of daily article count |
| `gdelt_finbert` | GDELT + FinBERT | z-score of FinBERT positive-sentiment score across articles |
| `gdelt_vader` | GDELT + VADER | z-score of VADER compound score across articles (baseline scorer) |
| `trends_interest` | Google Trends | z-score of relative search interest (normalized to anchor term) |
| `edgar_form4_count` | SEC EDGAR | z-score of Form 4 insider-trade filing count |
| `edgar_8k_count` | SEC EDGAR | z-score of 8-K filing count |

Price and momentum are never part of a signal. They appear only as controls in
the Fama-MacBeth regressions and as baselines.

---

## 5. Evaluation

### 5.1 Information coefficient (IC)

Spearman rank correlation between the cross-sectional feature ranks on date *t*
and the cross-sectional excess return ranks at horizon *h*. A higher IC means
the signal ranked stocks in the same order as subsequent returns.

IC is computed for each date with at least 20 valid observations. The mean IC
and its Newey-West standard error (lag = 2 × horizon, to account for overlapping
returns) are reported. A t-statistic > 1.96 is the uncorrected significance
threshold; BH-FDR correction is applied across all tests.

### 5.2 Quintile analysis

Securities are sorted by signal into quintile bins within each date. The mean
excess return of each quintile is computed for each horizon. The Q5–Q1 spread
(top quintile minus bottom quintile) is the primary quintile summary statistic.

### 5.3 Fama-MacBeth regressions

For each date, regress security excess returns on the signal feature plus
controls (size, past 21-day return, 21-day realized volatility, sector dummy).
Average the cross-sectional regression coefficients across all dates. Standard
errors use Newey-West with 2 × horizon lags. This isolates the signal's
marginal predictive contribution.

### 5.4 Walk-forward (out-of-sample) validation

All reported IC statistics use a walk-forward expanding window:
- Training period: dates 1 through *t*.
- Prediction date: *t+1*.
- No random splits; no look-ahead.
- An embargo of 2 × horizon trading days is applied between training and test to
  prevent return overlap leakage.

### 5.5 Multiple-testing correction

All signal × horizon × cohort combinations are listed before evaluation (see
`study/PREREGISTRATION.md`, section 5). The Benjamini-Hochberg (BH) FDR
procedure is applied at q = 0.05. Only combinations that survive BH correction
are reported as "significant." The full unadjusted table is also shown so
readers can assess the raw results.

### 5.6 Baselines

Every signal is evaluated against three baselines:

| Baseline | Description |
|---|---|
| Zero | Constant zero excess return (predicts no edge) |
| Momentum | Past 21-day return rank |
| Sector mean | Mean return of the security's GICS sector |

A signal that does not beat all three baselines is noted as such.

---

## 6. Prediction Ledger

Each trading day, before market close, the pipeline generates a frozen
prediction for each security × horizon combination using the current model. The
prediction includes:

- `made_at`: UTC timestamp of the prediction
- `model_version`: git SHA of the model code
- `data_hash`: SHA-256 hash of the feature vector used
- `predicted_excess_return`: the model's point estimate
- `predicted_direction`: up (>0) or down (≤0)

Predictions are written to the append-only data archive and **never
overwritten**. When the outcome date arrives, the realized excess return is
recorded alongside the prediction. The running hit rate and calibration plot
are updated nightly.

---

## 7. Backfill Policy

| Source | Backfill allowed? | Reason |
|---|---|---|
| Prices | Yes | True historical data |
| Wikipedia pageviews | Yes | Official API returns historical data |
| GDELT | Yes | Full historical archive available |
| SEC EDGAR | Yes | Filing timestamps are immutable |
| Google Trends | Yes, with caveats | Historical Trends data is sampled and noisy; normalized relative to an anchor term |
| StockTwits / Reddit | No | Not collected (terms not cleared) |

Backfilled records are labeled `data_type = "backfilled"` in the archive.
"Live" records (collected on the day) are labeled `data_type = "live"`.
Evaluation results are reported separately for backfilled vs live data when
the distinction matters.

---

## 8. Known Limitations

1. **Short live history.** With less than 12 months of live data, most tests are
   underpowered, especially for the 21-day horizon. Results should be treated
   as exploratory until the power thresholds in section 8 of the
   pre-registration are met.

2. **GDELT entity matching.** GDELT articles are matched to companies via name
   and alias dictionaries. Ambiguous company names (e.g., "Target", "Block",
   "Apple") may include irrelevant articles, reducing signal quality. A
   precision audit (200-article sample per source) will be published when
   sufficient data exists.

3. **Google Trends sampling noise.** Trends data is a sampled, relative index,
   not an absolute count. Different API calls for the same period can return
   slightly different values. Aggressive caching and an anchor-term
   normalization scheme are used to reduce noise.

4. **Survivorship bias (partial).** The S&P 500 universe is built from the
   current composition plus Wikipedia's "selected changes" table, which is
   incomplete for changes before ~2000. Backfilled results may understate
   the difficulty of predicting returns, since failed companies are
   underrepresented in the universe.

5. **No transaction costs.** Quintile return spreads and hit rates do not
   account for bid-ask spreads, market impact, or short-borrowing costs. Real
   implementation costs would reduce any observed edge.

6. **No leverage, no execution model.** The prediction ledger records directional
   signals, not a portfolio. Translating directional IC into a realized return
   stream requires execution assumptions not modeled here.

---

## 9. Changes and Amendments

Any deviation from the pre-registered plan is documented in
`study/AMENDMENTS.md` with a date and rationale. Silent edits to the
methodology are not permitted after `study/PREREGISTRATION.md` was
committed on 2026-10-03.

---

## 10. Reproducibility

Any published number can be regenerated from the raw archive with:

```bash
make reproduce
```

This rebuilds all features, evaluations, and site data from the Parquet archive
using the pinned dependency lockfile. Runtime: approximately 30 minutes on a
standard laptop (FinBERT NLP is the bottleneck).

---

_For questions or corrections: leadlaglab@gmail.com_

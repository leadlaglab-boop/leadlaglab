# Lead/Lag Lab — Pre-Registration

**Status: PENDING OWNER REVIEW — owner has not yet reviewed or approved this pre-registration. See study/AMENDMENTS.md A001.**

Pre-registration date: 2026-10-03
Site: [leadlaglab.com](https://leadlaglab.com)
Contact: leadlaglab@gmail.com
Code repository: github.com/leadlaglab-boop/leadlaglab

---

## 0. Purpose of This Document

This document records the study design, hypotheses, and planned analyses
**before any evaluation code runs on real return data**. Its purpose is to
prevent unconscious p-hacking: once this file is committed, the analysis plan
is locked. Deviations must be disclosed as exploratory in any published results.

---

## 1. Research Questions

1. Do Wikipedia pageview spikes for S&P 500 companies predict short-term excess
   returns over the following 1, 5, and 21 trading days?
2. Do GDELT news-tone signals (average article tone; FinBERT sentiment;
   VADER sentiment) predict short-term excess returns?
3. Does a spike in SEC Form 4 (insider trades) filings predict returns?
4. Does Google Trends search interest in a company predict returns?
5. Which signals, if any, survive multiple-testing correction (BH FDR at q=0.05)?
6. Do signals from retail-attention stocks (GME, AMC, etc.) differ from the
   broad S&P 500 in terms of predictability and signal persistence?

### 1.1 Directional Priors (Pre-Specified)

| Signal | Prior direction | Rationale |
|--------|----------------|-----------|
| Wikipedia views (spike) | + short-term, reversal 5–21d | Attention → buying pressure → mean-reversion |
| GDELT tone (positive) | + 1d | Positive news → sentiment lift |
| GDELT FinBERT positive | + 1d | Same as tone |
| GDELT article count (spike) | ambiguous | High attention can be positive or negative |
| EDGAR Form 4 (buys) | + | Insider knowledge signal |
| EDGAR Form 4 (count only) | ambiguous | Volume without direction |
| EDGAR 8-K count (spike) | ambiguous | Material events: sign-dependent |
| Google Trends (spike) | + 1d, reversal 5–21d | Same attention hypothesis as Wikipedia |

Signals marked *ambiguous* will be tested two-sided.

---

## 2. Universe

### 2.1 Primary Universe
- S&P 500 current constituents as of the study start date (survivorship bias
  documented; see `docs/DATA_SOURCES.md`)
- Cohort: `sp500` in SecurityMaster

### 2.2 Retail-Attention Basket
- 15 tickers with elevated retail activity (short interest, social volume,
  options flow) — see `config/universe_retail.yaml`
- Cohort: `retail_attention`
- Analysed separately and jointly with S&P 500

### 2.3 Exclusions
- Stocks with fewer than 40 non-missing signal observations in any trailing
  window are excluded from that window's cross-section
- Stocks in the bottom 5% of market-cap (by month) are excluded from
  Fama-MacBeth regressions but retained for descriptive analysis

---

## 3. Data and Signals

### 3.1 Sources

| Source | Data | Terms |
|--------|------|-------|
| Wikimedia Pageviews API | Daily article views | Free, CC BY-SA |
| SEC EDGAR | Form 4, 8-K filing counts | Public domain |
| GDELT DOC API | Article count, average tone | Open/free |
| Google Trends (pytrends) | Relative search interest | Research only; not redistributed |

### 3.2 Look-Ahead Invariant

Features constructed for decision date t use only raw signal records with
`observed_at < market_close(t)` (16:00 ET, DST-corrected).

For backfilled data: `observed_at = market_close(effective_date)`, so signals
from effective_date t-1 are valid; signals from effective_date t are **not used**
(observed_at = market_close(t), which does not satisfy the strict inequality).

This invariant is enforced by Hypothesis property tests in `tests/test_leakage.py`
and `tests/test_features.py`.

### 3.3 Signal Lag

A 1-trading-day lag is applied between signal date and feature date:
- Feature for date t uses signals from effective_date ≤ t - 1
- This provides a buffer for API publication delays and ensures the "decision"
  is made after market close on t-1 with knowledge of t-1's signals

---

## 4. Feature Construction

### 4.1 Raw → Feature Pipeline

For each (security_id, date, signal_type) triple:

1. **Load trailing window**: 60 trading days of raw signal values ending on date
   t-1 (inclusive). Minimum 10 non-NaN observations required.

2. **Log-transform** (where applicable):
   - Wikipedia views: log1p(views)
   - GDELT article count: log1p(count)
   - Google Trends interest: raw (already 0–100 normalized)
   - Tone/sentiment scores: raw (already bounded)

3. **Rolling z-score**:
   ```
   z = (x_t - mean(x_{t-60:t-1})) / std(x_{t-60:t-1})
   ```
   Capped at ±4 to reduce outlier impact.

4. **Cross-sectional rank** (`rank_cs`): within each (date, GICS sector),
   rank all securities by z-score, map to [0, 1] using fractional rank.
   Securities with missing values are excluded from the cross-section.

5. **Cross-sectional z-score** (`z_cs`): within each (date, GICS sector),
   standardise z-scores to mean=0, std=1.

6. **Empirical-Bayes shrinkage** (`shrunk_value`): shrink `z_cs` toward the
   cross-sectional mean using the James-Stein-style estimator:
   ```
   B  = 1 / (1 + n_obs / 10)   (shrinkage factor; larger n → less shrinkage)
   shrunk_value = (1 - B) * z_cs + B * 0
   ```
   (Cross-sectional mean of z_cs is 0 by construction, so we shrink toward 0.)

### 4.2 Feature Names

| Feature name | Source | Transform |
|-------------|--------|-----------|
| `wiki_views_z60d` | Wikipedia | log1p(views), 60d rolling z-score |
| `edgar_form4_z60d` | EDGAR | form_4_count, 60d rolling z-score |
| `edgar_8k_z60d` | EDGAR | form_8k_count, 60d rolling z-score |
| `gdelt_n_z60d` | GDELT | log1p(article_count), 60d rolling z-score |
| `gdelt_tone_z60d` | GDELT | avg_tone, 60d rolling z-score |
| `gdelt_finbert_z60d` | GDELT/NLP | finbert_avg_positive - finbert_avg_negative, 60d rolling z-score |
| `gdelt_vader_z60d` | GDELT/NLP | vader_avg_compound, 60d rolling z-score |
| `trends_z60d` | Google Trends | anchor_normalized_interest, 60d rolling z-score |

### 4.3 Feature Version

All features in this study use `feature_version = "v1"`.

---

## 5. Return Targets

### 5.1 Definition

Excess return for security i over horizon h starting on day t+1:
```
R(i, t, h) = cumret(i, t+1, t+h) - cumret(SPY, t+1, t+h)
```
where cumret is the log-cumulative return over trading days t+1 through t+h
using adjusted closes.

### 5.2 Horizons

| Horizon | Days | Label |
|---------|------|-------|
| Short | 1 | `1d` |
| Medium | 5 | `5d` (approximately 1 week) |
| Long | 21 | `21d` (approximately 1 month) |

---

## 6. Evaluation Plan (M5)

### 6.1 Metrics

**Primary metric**: Information Coefficient (IC) = Spearman rank correlation
between feature value and forward excess return, computed cross-sectionally
each date and then averaged (mean IC and IC t-statistic).

**Secondary metrics**:
- Quintile portfolio returns: sort all stocks into 5 buckets by feature value
  on each date; measure excess return spread between Q5 (top) and Q1 (bottom)
- Fama-MacBeth regression: regress forward returns on feature values
  cross-sectionally each period, report time-series average of coefficients
  with Newey-West standard errors (lags = max(horizon, 5))

### 6.2 Multiple Testing Correction

With 8 features × 3 horizons × 2 cohorts = 48 primary tests, we apply
Benjamini-Hochberg FDR correction at q = 0.05.

Only features surviving BH FDR correction are classified as "significant
predictors". All p-values (corrected and uncorrected) are reported.

### 6.3 Walk-Forward Validation

- **No random train/test splits** (would violate temporal ordering)
- **Walk-forward with embargo**: training window expands from 2020-01-01;
  test starts 2021-01-01; embargo = max(horizon, 5) trading days between
  train end and test start to prevent leakage from overlapping return windows
- Minimum training set: 252 trading days (~1 year)

### 6.4 Benchmark

Baseline IC = 0 (uninformative signal). All IC t-statistics are tested against
this null.

Practical significance threshold: |mean IC| ≥ 0.03 (industry rough benchmark
for a "meaningful" signal in quantitative equity research).

---

## 7. Model Specification (M6)

The prediction ledger will record daily predictions using a ridge-regularised
linear model trained on the feature set above. Model selection is not a focus
of this study; the linear model is chosen for interpretability. Model details
will be pre-specified before any ledger entries are written.

---

## 8. Limitations and Disclosures

1. **Survivorship bias**: The stock universe uses current S&P 500 constituents
   only. Stocks that were removed from the index (typically due to poor
   performance) are excluded, biasing signal evaluation upward.

2. **Look-ahead bias in index membership**: We do not have point-in-time index
   membership data. Stocks are treated as always having been in the index
   from our study start date.

3. **GDELT ambiguity**: For companies with common-word names (e.g., "Target",
   "Visa"), GDELT queries may include irrelevant articles, adding noise.
   Ambiguous tickers are flagged in `config/gdelt_entity_aliases.yaml`.

4. **Wikipedia article coverage**: Manually mapped for ~60 tickers; remaining
   tickers use an automated opensearch lookup that may select incorrect articles.

5. **Google Trends unofficial API**: pytrends is an unofficial scraper subject
   to rate-limiting and ToS changes. Data gaps are expected and documented.

6. **EDGAR "recent" window**: EDGAR's submission API returns only the most
   recent ~1,000 filings per company. For high-frequency filers, data older
   than ~2 years may be missing from the backfill.

7. **NLP model**: FinBERT (ProsusAI/finbert) was trained on financial news;
   GDELT article titles are short and may not match the training distribution.

---

## 9. Timeline

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Foundations, CI, leakage tests | Complete |
| M2 | Universe + prices | Complete |
| M3 | Signal collectors | Complete |
| **M4** | **Feature pipeline** | **In progress** |
| M5 | Evaluation engine | Not started |
| M6 | Prediction ledger | Not started |
| M7 | Site build | Not started |
| M8 | Domain + launch | Not started |
| M9 | Monitoring | Not started |

---

*This pre-registration was committed before any M5 evaluation code was written.*

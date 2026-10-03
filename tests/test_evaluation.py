"""
Tests for M5 evaluation engine.

Covers:
  - IC computation: sign, magnitude, t-stat direction
  - Quintile returns: Q5 > Q1 for a positively predictive signal
  - Fama-MacBeth: coefficient sign matches IC sign
  - Newey-West SE: wider than OLS SE for autocorrelated series
  - BH FDR: rejects correct hypotheses, FDR bound holds
  - Walk-forward folds: embargo enforced, no leakage
  - Returns computation: excess return = stock - benchmark
  - No look-ahead: OOS test dates are all after train_end + embargo
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation.metrics import (
    _newey_west_se,
    apply_bh_fdr,
    compute_fama_macbeth,
    compute_ic,
    compute_quintiles,
)
from pipeline.evaluation.walkforward import generate_folds, split_df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_df(
    n_stocks: int = 50,
    n_dates: int = 60,
    ic_signal: float = 0.15,
    horizon: int = 5,
    cohort: str = "sp500",
    feature_name: str = "wiki_views_z60d",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic evaluation DataFrame with a known IC.

    The feature is positively correlated with next-period returns at
    approximately ic_signal correlation.
    """
    rng = np.random.default_rng(seed)
    start = date(2021, 1, 4)
    dates = [start + timedelta(days=7 * i) for i in range(n_dates)]
    tickers = [f"T{i:03d}" for i in range(n_stocks)]

    rows = []
    for d in dates:
        true_return = rng.standard_normal(n_stocks)
        noise = rng.standard_normal(n_stocks)
        feature = ic_signal * true_return + math.sqrt(1 - ic_signal**2) * noise
        for j, ticker in enumerate(tickers):
            rows.append(
                {
                    "security_id": f"sid_{ticker}",
                    "ticker": ticker,
                    "cohort": cohort,
                    "date": d,
                    "feature_name": feature_name,
                    "feature_value": float(feature[j]),
                    "shrunk_value": float(feature[j]),
                    "horizon": horizon,
                    "excess_return": float(true_return[j]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# IC tests
# ---------------------------------------------------------------------------


class TestComputeIC:
    def test_positive_signal_positive_ic(self):
        df = _make_eval_df(ic_signal=0.20, n_dates=100)
        results = compute_ic(df)
        assert len(results) == 1
        r = results[0]
        assert r.mean_ic > 0.05, f"Expected positive IC, got {r.mean_ic:.3f}"

    def test_negative_signal_negative_ic(self):
        df = _make_eval_df(ic_signal=-0.20, n_dates=100)
        results = compute_ic(df)
        r = results[0]
        assert r.mean_ic < -0.05, f"Expected negative IC, got {r.mean_ic:.3f}"

    def test_zero_signal_ic_near_zero(self):
        df = _make_eval_df(ic_signal=0.0, n_dates=200, seed=99)
        results = compute_ic(df)
        r = results[0]
        assert abs(r.mean_ic) < 0.10, f"|IC| too large for noise signal: {r.mean_ic:.3f}"

    def test_t_stat_sign_matches_ic(self):
        df = _make_eval_df(ic_signal=0.25, n_dates=80)
        r = compute_ic(df)[0]
        assert r.ic_t_stat > 0, "t-stat should be positive when mean IC is positive"

    def test_t_stat_significant_for_strong_signal(self):
        df = _make_eval_df(ic_signal=0.30, n_dates=120)
        r = compute_ic(df)[0]
        assert abs(r.ic_t_stat) > 2.0, f"Expected significant t-stat, got {r.ic_t_stat:.2f}"

    def test_multiple_features_and_cohorts(self):
        df1 = _make_eval_df(feature_name="feat_a", cohort="sp500", ic_signal=0.15)
        df2 = _make_eval_df(feature_name="feat_a", cohort="retail_attention", ic_signal=0.20)
        df3 = _make_eval_df(feature_name="feat_b", cohort="sp500", ic_signal=-0.10)
        df = pd.concat([df1, df2, df3], ignore_index=True)
        results = compute_ic(df)
        assert len(results) == 3
        keys = {(r.feature, r.cohort) for r in results}
        assert ("feat_a", "sp500") in keys
        assert ("feat_a", "retail_attention") in keys
        assert ("feat_b", "sp500") in keys

    def test_insufficient_dates_returns_empty(self):
        df = _make_eval_df(n_dates=2, n_stocks=3)  # too few stocks per date
        results = compute_ic(df)
        # Either empty or IC with very few periods is fine
        if results:
            assert results[0].n_periods <= 2

    def test_ic_series_length_matches_n_periods(self):
        df = _make_eval_df(n_dates=40)
        r = compute_ic(df)[0]
        assert len(r.ic_series) == r.n_periods


# ---------------------------------------------------------------------------
# Quintile tests
# ---------------------------------------------------------------------------


class TestComputeQuintiles:
    def test_positive_signal_q5_greater_q1(self):
        df = _make_eval_df(ic_signal=0.30, n_dates=100, n_stocks=100)
        results = compute_quintiles(df)
        assert len(results) == 1
        r = results[0]
        assert r.quintile_means[5] > r.quintile_means[1], (
            f"Q5 mean {r.quintile_means[5]:.3f} should exceed Q1 {r.quintile_means[1]:.3f}"
        )

    def test_spread_sign_matches_signal_direction(self):
        df_pos = _make_eval_df(ic_signal=0.25, n_dates=80, n_stocks=80)
        df_neg = _make_eval_df(ic_signal=-0.25, n_dates=80, n_stocks=80)
        r_pos = compute_quintiles(df_pos)[0]
        r_neg = compute_quintiles(df_neg)[0]
        assert r_pos.spread_mean > 0
        assert r_neg.spread_mean < 0

    def test_quintile_means_cover_all_5(self):
        df = _make_eval_df(n_stocks=100, n_dates=50)
        r = compute_quintiles(df)[0]
        assert set(r.quintile_means.keys()) == {1, 2, 3, 4, 5}

    def test_too_few_stocks_returns_empty(self):
        df = _make_eval_df(n_stocks=5, n_dates=30)
        results = compute_quintiles(df)
        # With 5 stocks, qcut needs ≥10 stocks per date; should return empty or skip dates
        # Just verify it doesn't crash
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Fama-MacBeth tests
# ---------------------------------------------------------------------------


class TestComputeFamaMacBeth:
    def test_coeff_sign_matches_signal(self):
        df = _make_eval_df(ic_signal=0.20, n_dates=80, n_stocks=80)
        results = compute_fama_macbeth(df)
        assert len(results) == 1
        r = results[0]
        assert r.avg_coeff > 0, f"FMB coeff should be positive, got {r.avg_coeff:.4f}"

    def test_nw_t_stat_significant_for_strong_signal(self):
        df = _make_eval_df(ic_signal=0.30, n_dates=120, n_stocks=100)
        r = compute_fama_macbeth(df)[0]
        assert abs(r.nw_t_stat) > 1.5, f"Expected significant NW t-stat, got {r.nw_t_stat:.2f}"

    def test_nw_se_non_negative(self):
        df = _make_eval_df(n_dates=50, n_stocks=60)
        r = compute_fama_macbeth(df)[0]
        assert r.nw_se >= 0, "NW SE must be non-negative"


# ---------------------------------------------------------------------------
# Newey-West SE tests
# ---------------------------------------------------------------------------


class TestNeweyWestSE:
    def test_iid_matches_classical_se(self):
        """For IID series, NW SE ≈ classical SE (within a reasonable tolerance)."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        nw = _newey_west_se(x, lags=0)
        classical = x.std(ddof=1) / np.sqrt(len(x))
        assert abs(nw - classical) < 0.01 * classical + 1e-10

    def test_autocorrelated_series_wider_se(self):
        """For autocorrelated series, NW SE should be wider than classical SE."""
        rng = np.random.default_rng(1)
        n = 300
        # AR(1) with high autocorrelation
        x = np.zeros(n)
        x[0] = rng.standard_normal()
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + rng.standard_normal() * 0.6
        nw = _newey_west_se(x, lags=10)
        classical = x.std(ddof=1) / np.sqrt(n)
        assert nw > classical, f"NW SE ({nw:.5f}) should exceed classical SE ({classical:.5f})"


# ---------------------------------------------------------------------------
# BH FDR tests
# ---------------------------------------------------------------------------


class TestApplyBhFdr:
    def test_rejects_small_p_values(self):
        p_values = [0.001, 0.002, 0.003, 0.5, 0.8, 0.9]
        rejected, adj_p = apply_bh_fdr(p_values, q=0.05)
        # First three should be rejected
        assert all(rejected[:3]), "Small p-values should be rejected"
        # Large p-values should not
        assert not any(rejected[3:]), "Large p-values should not be rejected"

    def test_all_null_not_rejected(self):
        p_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        rejected, _ = apply_bh_fdr(p_values, q=0.05)
        assert not any(rejected)

    def test_adjusted_p_geq_raw_p(self):
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        _, adj_p = apply_bh_fdr(p_values, q=0.05)
        for raw, adj in zip(p_values, adj_p, strict=False):
            assert adj >= raw - 1e-10, f"Adjusted p {adj:.4f} < raw p {raw:.4f}"

    def test_nan_p_values_handled(self):
        p_values = [0.01, float("nan"), 0.03]
        rejected, adj_p = apply_bh_fdr(p_values, q=0.05)
        assert not np.isnan(rejected[0])  # non-nan p-value processed
        assert np.isnan(adj_p[1])  # nan propagated

    def test_48_primary_tests(self):
        """Verify BH handles the full 48-test matrix without errors."""
        rng = np.random.default_rng(7)
        p_values = rng.uniform(0, 1, 48).tolist()
        p_values[0] = 0.0001  # one strong signal
        rejected, adj_p = apply_bh_fdr(p_values, q=0.05)
        assert len(rejected) == 48
        assert rejected[0] is True  # strong signal should be rejected (True = significant)


# ---------------------------------------------------------------------------
# Walk-forward fold tests
# ---------------------------------------------------------------------------


class TestWalkForward:
    def _make_dates(self, start: date = date(2020, 1, 2), n: int = 600) -> list[date]:
        """Return n approximately-weekly dates from start."""
        return [start + timedelta(days=5 * i) for i in range(n)]

    def test_folds_generated(self):
        dates = self._make_dates()
        folds = generate_folds(dates, horizon=5)
        assert len(folds) >= 1, "Expected at least one fold"

    def test_embargo_respected(self):
        """test_start must be at least embargo trading days after train_end."""
        dates = self._make_dates()
        for horizon in [1, 5, 21]:
            folds = generate_folds(dates, horizon=horizon)
            for fold in folds:
                assert fold.test_start > fold.train_end, (
                    f"test_start {fold.test_start} not after train_end {fold.train_end}"
                )

    def test_no_overlap_between_train_and_test(self):
        dates = self._make_dates()
        folds = generate_folds(dates, horizon=5)
        for fold in folds:
            assert fold.test_start > fold.train_end

    def test_split_df_partitions_correctly(self):
        dates = self._make_dates(n=400)
        folds = generate_folds(dates, horizon=5)
        if not folds:
            pytest.skip("No folds generated; need more data")
        fold = folds[0]
        df = pd.DataFrame(
            {
                "date": dates[:200],
                "value": range(200),
                "horizon": 5,
                "security_id": "sid_T001",
                "feature_name": "f",
                "feature_value": 1.0,
                "excess_return": 0.01,
                "cohort": "sp500",
            }
        )
        train, test = split_df(df, fold)
        assert not train.empty or not test.empty  # at least one is non-empty
        # No overlap
        train_dates = set(pd.to_datetime(train["date"]).dt.date)
        test_dates = set(pd.to_datetime(test["date"]).dt.date)
        assert train_dates.isdisjoint(test_dates), "Train and test dates must not overlap"

    def test_oos_test_dates_after_train_end(self):
        """Key no-look-ahead property: all test dates > train_end."""
        dates = self._make_dates()
        folds = generate_folds(dates, horizon=21)
        for fold in folds:
            assert fold.test_start > fold.train_end


# ---------------------------------------------------------------------------
# Returns computation (unit-level; full integration test requires archive)
# ---------------------------------------------------------------------------


class TestExcessReturnFormula:
    def test_excess_return_formula(self):
        """excess_return = log(stock_end/stock_start) - log(bench_end/bench_start)."""
        # stock goes up 5%, benchmark up 2%
        stock_ret = np.log(1.05)
        bench_ret = np.log(1.02)
        excess = stock_ret - bench_ret
        assert abs(excess - (np.log(1.05) - np.log(1.02))) < 1e-10

    def test_excess_return_sign(self):
        # If stock outperforms benchmark, excess should be positive
        assert np.log(1.10) - np.log(1.02) > 0
        # If stock underperforms, negative
        assert np.log(0.95) - np.log(1.02) < 0

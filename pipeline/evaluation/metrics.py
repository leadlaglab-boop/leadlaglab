"""
Signal evaluation metrics: IC, quintile returns, Fama-MacBeth.

All functions operate on a merged DataFrame with columns:
    security_id, ticker, date, feature_name, feature_value, horizon, excess_return

Public surface:
    compute_ic(df)           → per-(feature, horizon, cohort): mean IC, t-stat, p-value
    compute_quintiles(df)    → per-(feature, horizon, cohort, date, quintile): mean excess_return
    compute_fama_macbeth(df) → per-(feature, horizon, cohort): avg coeff, NW t-stat, p-value
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, SupportsInt, cast

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ---------------------------------------------------------------------------
# Information Coefficient (Spearman rank correlation)
# ---------------------------------------------------------------------------


class ICResult(NamedTuple):
    feature: str
    horizon: int
    cohort: str
    mean_ic: float
    ic_std: float
    ic_t_stat: float
    ic_p_value: float  # two-sided t-test vs 0
    n_periods: int
    ic_series: list[float]  # per-date ICs


def compute_ic(df: pd.DataFrame) -> list[ICResult]:
    """
    Compute cross-sectional IC (Spearman) per date, then summarise.

    df must have: date, feature_name, feature_value, horizon, excess_return, cohort
    Feature value used is `shrunk_value` if present, else `feature_value`.
    """
    val_col = "shrunk_value" if "shrunk_value" in df.columns else "feature_value"
    results: list[ICResult] = []

    for (feature, horizon, cohort), grp in df.groupby(
        ["feature_name", "horizon", "cohort"], sort=False
    ):
        per_date: list[float] = []
        for _date, day_grp in grp.groupby("date"):
            sub = day_grp[[val_col, "excess_return"]].dropna()
            if len(sub) < 5:
                continue
            rho, _ = stats.spearmanr(sub[val_col], sub["excess_return"])
            if not np.isnan(rho):
                per_date.append(float(rho))

        if not per_date:
            continue

        ic_arr = np.array(per_date)
        n = len(ic_arr)
        mean_ic = float(ic_arr.mean())
        ic_std = float(ic_arr.std(ddof=1)) if n > 1 else np.nan
        se = ic_std / np.sqrt(n) if n > 1 else np.nan
        t_stat = mean_ic / se if se and se > 0 else np.nan
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if not np.isnan(t_stat) else np.nan

        results.append(
            ICResult(
                feature=str(feature),
                horizon=int(cast("SupportsInt", horizon)),
                cohort=str(cohort),
                mean_ic=mean_ic,
                ic_std=ic_std,
                ic_t_stat=t_stat,
                ic_p_value=p_value,
                n_periods=n,
                ic_series=ic_arr.tolist(),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Quintile portfolio returns
# ---------------------------------------------------------------------------


class QuintileResult(NamedTuple):
    feature: str
    horizon: int
    cohort: str
    # Annualised mean excess return per quintile
    quintile_means: dict[int, float]  # 1..5 → mean excess return
    spread_mean: float  # Q5 – Q1
    spread_t_stat: float
    spread_p_value: float
    n_periods: int


def compute_quintiles(df: pd.DataFrame) -> list[QuintileResult]:
    """Compute equal-weighted quintile excess returns sorted by feature value per date."""
    val_col = "shrunk_value" if "shrunk_value" in df.columns else "feature_value"
    results: list[QuintileResult] = []

    for (feature, horizon, cohort), grp in df.groupby(
        ["feature_name", "horizon", "cohort"], sort=False
    ):
        # Per-date: assign quintile and record returns
        daily_spread: list[float] = []
        quintile_acc: dict[int, list[float]] = {q: [] for q in range(1, 6)}

        for _date, day_grp in grp.groupby("date"):
            sub = day_grp[[val_col, "excess_return"]].dropna()
            if len(sub) < 10:  # need enough stocks to form quintiles
                continue
            sub = sub.copy()
            sub["quintile"] = pd.qcut(sub[val_col], q=5, labels=False, duplicates="drop")
            sub["quintile"] = sub["quintile"] + 1  # 1-indexed
            sub = sub.dropna(subset=["quintile"])
            sub["quintile"] = sub["quintile"].astype(int)

            q_means = sub.groupby("quintile")["excess_return"].mean()
            for q in range(1, 6):
                if q in q_means:
                    quintile_acc[q].append(float(q_means[q]))

            if 5 in q_means and 1 in q_means:
                daily_spread.append(float(q_means[5] - q_means[1]))

        if not daily_spread:
            continue

        spread_arr = np.array(daily_spread)
        n = len(spread_arr)
        spread_mean = float(spread_arr.mean())
        spread_std = float(spread_arr.std(ddof=1)) if n > 1 else np.nan
        se = spread_std / np.sqrt(n) if n > 1 and spread_std > 0 else np.nan
        t_stat = spread_mean / se if se and not np.isnan(se) else np.nan
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if not np.isnan(t_stat) else np.nan

        q_means_out = {
            q: float(np.mean(vals)) if vals else np.nan for q, vals in quintile_acc.items()
        }

        results.append(
            QuintileResult(
                feature=str(feature),
                horizon=int(cast("SupportsInt", horizon)),
                cohort=str(cohort),
                quintile_means=q_means_out,
                spread_mean=spread_mean,
                spread_t_stat=t_stat,
                spread_p_value=p_value,
                n_periods=n,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Fama-MacBeth regression with Newey-West standard errors
# ---------------------------------------------------------------------------


class FMBResult(NamedTuple):
    feature: str
    horizon: int
    cohort: str
    avg_coeff: float
    nw_se: float
    nw_t_stat: float
    nw_p_value: float
    n_periods: int


def _newey_west_se(series: np.ndarray, lags: int) -> float:
    """
    Newey-West (HAC) standard error for the mean of a time series.
    Implements the Bartlett kernel estimator.
    """
    n = len(series)
    x = series - series.mean()
    gamma0 = float(np.dot(x, x) / n)
    variance = gamma0
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1)
        gamma_j = float(np.dot(x[j:], x[:-j]) / n)
        variance += 2 * w * gamma_j
    variance = max(variance, 1e-15)  # numerical floor
    return float(np.sqrt(variance / n))


def compute_fama_macbeth(df: pd.DataFrame, controls: list[str] | None = None) -> list[FMBResult]:
    """
    Fama-MacBeth cross-sectional regressions.

    Each period t: regress excess_return on [feature, controls].
    Time-series of coefficients → mean with Newey-West SEs (lags = max(horizon, 5)).

    Controls columns must be present in df if provided (e.g., size_z, mom_z).
    The feature coefficient is always reported; control coefficients are stored
    internally but not returned here (the engine can extract them if needed).
    """
    val_col = "shrunk_value" if "shrunk_value" in df.columns else "feature_value"
    results: list[FMBResult] = []

    for (feature, horizon, cohort), grp in df.groupby(
        ["feature_name", "horizon", "cohort"], sort=False
    ):
        nw_lags = max(int(cast("SupportsInt", horizon)), 5)
        period_coefs: list[float] = []

        for _date, day_grp in grp.groupby("date"):
            cols = [val_col, "excess_return"]
            if controls:
                cols += [c for c in controls if c in day_grp.columns]
            sub = day_grp[cols].dropna()
            if len(sub) < max(3, 2 + len(controls or [])):
                continue

            X = sub[[val_col] + [c for c in (controls or []) if c in sub.columns]]
            y = sub["excess_return"]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Xc = add_constant(X, has_constant="add")
                    res = OLS(y, Xc).fit()
                # Coefficient for the feature (index 1; 0 is the intercept)
                period_coefs.append(float(res.params.iloc[1]))
            except Exception:
                continue

        if len(period_coefs) < 2:
            continue

        coef_arr = np.array(period_coefs)
        avg_coeff = float(coef_arr.mean())
        nw_se = _newey_west_se(coef_arr, lags=nw_lags)
        t_stat = avg_coeff / nw_se if nw_se > 0 else np.nan
        n = len(coef_arr)
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if not np.isnan(t_stat) else np.nan

        results.append(
            FMBResult(
                feature=str(feature),
                horizon=int(cast("SupportsInt", horizon)),
                cohort=str(cohort),
                avg_coeff=avg_coeff,
                nw_se=nw_se,
                nw_t_stat=t_stat,
                nw_p_value=p_value,
                n_periods=n,
            )
        )

    return results


# ---------------------------------------------------------------------------
# BH FDR multiple-testing correction
# ---------------------------------------------------------------------------


def apply_bh_fdr(
    p_values: list[float],
    q: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """
    Benjamini-Hochberg FDR correction.

    Returns (rejected, adjusted_p_values) matching input order.
    rejected[i] is True if the null is rejected at FDR level q.
    """
    from statsmodels.stats.multitest import multipletests

    arr = np.array(p_values, dtype=float)
    mask = ~np.isnan(arr)
    rejected = np.zeros(len(arr), dtype=bool)
    adj_p = np.full(len(arr), np.nan)

    if mask.sum() > 0:
        rej_sub, adj_sub, _, _ = multipletests(arr[mask], alpha=q, method="fdr_bh")
        rejected[mask] = rej_sub
        adj_p[mask] = adj_sub

    return rejected.tolist(), adj_p.tolist()

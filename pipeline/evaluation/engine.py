"""
Evaluation engine: orchestrates IC, quintile returns, Fama-MacBeth,
walk-forward OOS evaluation, and BH FDR correction.

Writes results to:
  {data_repo}/processed/evaluation/
    results.parquet          — all metric rows (in-sample and OOS)
    ic_series.parquet        — per-(feature, horizon, cohort, date) IC values
    quintile_returns.parquet — per-(feature, horizon, cohort, quintile) returns
    manifest.json            — run metadata

Called from: pipeline/cli.py evaluate
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, SupportsInt, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from pipeline.evaluation.metrics import (
    FMBResult,
    ICResult,
    QuintileResult,
    apply_bh_fdr,
    compute_fama_macbeth,
    compute_ic,
    compute_quintiles,
)
from pipeline.evaluation.returns import compute_excess_returns
from pipeline.evaluation.walkforward import TRAIN_START, generate_folds, split_df
from pipeline.features.builder import load_features
from pipeline.universe.security_master import load_security_master
from pipeline.validate.schemas import SCHEMA_VERSION

log = structlog.get_logger()

HORIZONS = (1, 5, 21)
FEATURE_VERSION = "v1"

# Output schemas
EVAL_RESULTS_SCHEMA = pa.schema(
    [
        pa.field("feature", pa.string(), nullable=False),
        pa.field("horizon", pa.int32(), nullable=False),
        pa.field("cohort", pa.string(), nullable=False),
        pa.field("eval_type", pa.string(), nullable=False),  # "is" | "oos"
        pa.field("metric", pa.string(), nullable=False),  # "ic" | "fmb" | "quintile_spread"
        pa.field("value", pa.float64(), nullable=True),
        pa.field("t_stat", pa.float64(), nullable=True),
        pa.field("p_value_raw", pa.float64(), nullable=True),
        pa.field("p_value_bh", pa.float64(), nullable=True),
        pa.field("significant_bh", pa.bool_(), nullable=True),
        pa.field("n_periods", pa.int32(), nullable=True),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)

IC_SERIES_SCHEMA = pa.schema(
    [
        pa.field("feature", pa.string(), nullable=False),
        pa.field("horizon", pa.int32(), nullable=False),
        pa.field("cohort", pa.string(), nullable=False),
        pa.field("eval_type", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("ic", pa.float64(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)

QUINTILE_SCHEMA = pa.schema(
    [
        pa.field("feature", pa.string(), nullable=False),
        pa.field("horizon", pa.int32(), nullable=False),
        pa.field("cohort", pa.string(), nullable=False),
        pa.field("eval_type", pa.string(), nullable=False),
        pa.field("quintile", pa.int32(), nullable=False),
        pa.field("mean_excess_return", pa.float64(), nullable=True),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def _load_eval_data(
    data_repo: Path,
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Load features and returns for [start, end] and join them.

    Returns DataFrame with:
        security_id, ticker, cohort, date, feature_name, feature_value,
        shrunk_value, horizon, excess_return
    """
    sm = load_security_master(data_repo)
    if sm.empty:
        raise RuntimeError("SecurityMaster is empty; run `lll universe` first.")

    features = load_features(data_repo, start_date=start, end_date=end)
    if features.empty:
        log.warning("engine.no_features", start=str(start), end=str(end))
        return pd.DataFrame()

    returns = compute_excess_returns(data_repo, start=start, end=end)
    if returns.empty:
        log.warning("engine.no_returns", start=str(start), end=str(end))
        return pd.DataFrame()

    # Attach ticker and cohort from SecurityMaster
    sm_active = sm[sm["valid_to"].isna()][["security_id", "ticker", "cohort"]].copy()
    features = features.merge(sm_active, on="security_id", how="inner")

    # Join features with returns on (ticker, date, horizon)
    returns["date"] = pd.to_datetime(returns["date"]).dt.date
    features["date"] = pd.to_datetime(features["date"]).dt.date

    merged = features.merge(
        returns[["ticker", "date", "horizon", "excess_return"]],
        on=["ticker", "date"],
        how="inner",
    )

    # Rename for metrics functions
    merged = merged.rename(columns={"value": "feature_value"})

    log.info(
        "engine.data_assembled",
        rows=len(merged),
        features=merged["feature_name"].nunique(),
        tickers=merged["ticker"].nunique(),
        date_range=f"{merged['date'].min()} – {merged['date'].max()}",
    )
    return merged


# ---------------------------------------------------------------------------
# Result serialisation helpers
# ---------------------------------------------------------------------------


def _ic_to_rows(results: list[ICResult], eval_type: str) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        rows.append(
            {
                "feature": r.feature,
                "horizon": r.horizon,
                "cohort": r.cohort,
                "eval_type": eval_type,
                "metric": "ic",
                "value": r.mean_ic,
                "t_stat": r.ic_t_stat,
                "p_value_raw": r.ic_p_value,
                "p_value_bh": None,
                "significant_bh": None,
                "n_periods": r.n_periods,
            }
        )
    return rows


def _fmb_to_rows(results: list[FMBResult], eval_type: str) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        rows.append(
            {
                "feature": r.feature,
                "horizon": r.horizon,
                "cohort": r.cohort,
                "eval_type": eval_type,
                "metric": "fmb",
                "value": r.avg_coeff,
                "t_stat": r.nw_t_stat,
                "p_value_raw": r.nw_p_value,
                "p_value_bh": None,
                "significant_bh": None,
                "n_periods": r.n_periods,
            }
        )
    return rows


def _quintile_to_rows(
    results: list[QuintileResult], eval_type: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (summary_rows, quintile_rows)."""
    summary, detail = [], []
    for r in results:
        summary.append(
            {
                "feature": r.feature,
                "horizon": r.horizon,
                "cohort": r.cohort,
                "eval_type": eval_type,
                "metric": "quintile_spread",
                "value": r.spread_mean,
                "t_stat": r.spread_t_stat,
                "p_value_raw": r.spread_p_value,
                "p_value_bh": None,
                "significant_bh": None,
                "n_periods": r.n_periods,
            }
        )
        for q, mean_ret in r.quintile_means.items():
            detail.append(
                {
                    "feature": r.feature,
                    "horizon": r.horizon,
                    "cohort": r.cohort,
                    "eval_type": eval_type,
                    "quintile": q,
                    "mean_excess_return": mean_ret,
                }
            )
    return summary, detail


def _apply_bh_to_rows(rows: list[dict[str, Any]], q: float = 0.05) -> list[dict[str, Any]]:
    """In-place apply BH FDR to IC rows (primary metric for the 48-test matrix)."""
    ic_rows = [r for r in rows if r["metric"] == "ic" and r["eval_type"] == "is"]
    p_vals = [r["p_value_raw"] for r in ic_rows]
    if not p_vals:
        return rows

    rejected, adj_p = apply_bh_fdr(p_vals, q=q)
    for i, r in enumerate(ic_rows):
        r["p_value_bh"] = adj_p[i]
        r["significant_bh"] = bool(rejected[i])
    return rows


# ---------------------------------------------------------------------------
# IC time-series (for rolling charts on the site)
# ---------------------------------------------------------------------------


def _extract_ic_series(
    data: pd.DataFrame,
    eval_type: str,
) -> list[dict[str, Any]]:
    """Extract per-date IC values for time-series charts."""
    from scipy import stats as _stats

    from pipeline.evaluation.metrics import _newey_west_se  # noqa: F401 (unused here)

    val_col = "shrunk_value" if "shrunk_value" in data.columns else "feature_value"
    rows = []
    for (feature, horizon, cohort), grp in data.groupby(
        ["feature_name", "horizon", "cohort"], sort=False
    ):
        for d, day_grp in grp.groupby("date"):
            sub = day_grp[[val_col, "excess_return"]].dropna()
            if len(sub) < 5:
                continue
            rho, _ = _stats.spearmanr(sub[val_col], sub["excess_return"])
            if not np.isnan(rho):
                rows.append(
                    {
                        "feature": str(feature),
                        "horizon": int(cast("SupportsInt", horizon)),
                        "cohort": str(cohort),
                        "eval_type": eval_type,
                        "date": d,
                        "ic": float(rho),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Main engine entry point
# ---------------------------------------------------------------------------


def run_evaluation(
    data_repo: Path,
    start: date | None = None,
    end: date | None = None,
    fdr_q: float = 0.05,
) -> Path:
    """
    Full evaluation run: in-sample metrics + walk-forward OOS.

    Returns path to the output directory.
    """
    eval_start = start or TRAIN_START
    eval_end = end or date.today()

    log.info("engine.start", start=str(eval_start), end=str(eval_end))

    data = _load_eval_data(data_repo, eval_start, eval_end)
    if data.empty:
        raise RuntimeError("No feature+return data assembled. Run features and prices first.")

    all_rows: list[dict[str, Any]] = []
    ic_series_rows: list[dict[str, Any]] = []
    quintile_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # In-sample metrics (all data)
    # ------------------------------------------------------------------
    log.info("engine.in_sample")
    ic_is = compute_ic(data)
    fmb_is = compute_fama_macbeth(data)
    quint_is = compute_quintiles(data)

    all_rows += _ic_to_rows(ic_is, "is")
    all_rows += _fmb_to_rows(fmb_is, "is")
    quint_summary, quint_detail = _quintile_to_rows(quint_is, "is")
    all_rows += quint_summary
    quintile_rows += quint_detail
    ic_series_rows += _extract_ic_series(data, "is")

    # ------------------------------------------------------------------
    # Walk-forward OOS metrics
    # ------------------------------------------------------------------
    log.info("engine.walkforward_start")
    all_dates = sorted(data["date"].unique())

    oos_data_parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        folds = generate_folds(all_dates, horizon=horizon)
        for fold in folds:
            _, test_df = split_df(data[data["horizon"] == horizon], fold)
            if not test_df.empty:
                oos_data_parts.append(test_df)

    if oos_data_parts:
        oos_data = pd.concat(oos_data_parts, ignore_index=True)
        oos_data = oos_data.drop_duplicates(["security_id", "date", "feature_name", "horizon"])

        ic_oos = compute_ic(oos_data)
        fmb_oos = compute_fama_macbeth(oos_data)
        quint_oos = compute_quintiles(oos_data)

        all_rows += _ic_to_rows(ic_oos, "oos")
        all_rows += _fmb_to_rows(fmb_oos, "oos")
        qs, qd = _quintile_to_rows(quint_oos, "oos")
        all_rows += qs
        quintile_rows += qd
        ic_series_rows += _extract_ic_series(oos_data, "oos")
        log.info(
            "engine.walkforward_done",
            oos_rows=len(oos_data),
            oos_tickers=oos_data["ticker"].nunique(),
        )
    else:
        log.warning("engine.no_oos_data", note="Not enough data for walk-forward folds yet.")

    # ------------------------------------------------------------------
    # Apply BH FDR correction to in-sample IC primary tests
    # ------------------------------------------------------------------
    all_rows = _apply_bh_to_rows(all_rows, q=fdr_q)

    # ------------------------------------------------------------------
    # Write Parquet outputs
    # ------------------------------------------------------------------
    out_dir = data_repo / "processed" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_results(all_rows, out_dir)
    _write_ic_series(ic_series_rows, out_dir)
    _write_quintile_returns(quintile_rows, out_dir)
    _write_manifest(out_dir, eval_start, eval_end, len(data))

    log.info("engine.done", output=str(out_dir))
    return out_dir


def _write_results(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["schema_version"] = SCHEMA_VERSION
    for col in ["horizon", "n_periods"]:
        if col in df.columns:
            df[col] = df[col].astype("Int32")  # nullable int
    table = pa.Table.from_pandas(df, schema=EVAL_RESULTS_SCHEMA, safe=False)
    pq.write_table(table, out_dir / "results.parquet", compression="snappy")
    log.info("engine.wrote_results", path=str(out_dir / "results.parquet"), rows=len(df))


def _write_ic_series(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["schema_version"] = SCHEMA_VERSION
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["horizon"] = df["horizon"].astype("Int32")
    table = pa.Table.from_pandas(df, schema=IC_SERIES_SCHEMA, safe=False)
    pq.write_table(table, out_dir / "ic_series.parquet", compression="snappy")
    log.info("engine.wrote_ic_series", rows=len(df))


def _write_quintile_returns(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["schema_version"] = SCHEMA_VERSION
    df["horizon"] = df["horizon"].astype("Int32")
    df["quintile"] = df["quintile"].astype("Int32")
    table = pa.Table.from_pandas(df, schema=QUINTILE_SCHEMA, safe=False)
    pq.write_table(table, out_dir / "quintile_returns.parquet", compression="snappy")
    log.info("engine.wrote_quintile_returns", rows=len(df))


def _write_manifest(out_dir: Path, start: date, end: date, n_rows: int) -> None:
    manifest = {
        "build_time": datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "eval_start": str(start),
        "eval_end": str(end),
        "input_rows": n_rows,
        "results_hash": _hash_file(out_dir / "results.parquet"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Load helpers (used by site builder and model pipeline)
# ---------------------------------------------------------------------------


def load_eval_results(data_repo: Path) -> pd.DataFrame:
    path = data_repo / "processed" / "evaluation" / "results.parquet"
    if not path.exists():
        return pd.DataFrame()
    return cast("pd.DataFrame", pq.read_table(path).to_pandas())


def load_ic_series(data_repo: Path) -> pd.DataFrame:
    path = data_repo / "processed" / "evaluation" / "ic_series.parquet"
    if not path.exists():
        return pd.DataFrame()
    return cast("pd.DataFrame", pq.read_table(path).to_pandas())


def load_quintile_returns(data_repo: Path) -> pd.DataFrame:
    path = data_repo / "processed" / "evaluation" / "quintile_returns.parquet"
    if not path.exists():
        return pd.DataFrame()
    return cast("pd.DataFrame", pq.read_table(path).to_pandas())

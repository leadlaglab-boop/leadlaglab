"""
Generate site/public/data/ JSON files from the processed archive.
These are the files the Astro site reads at runtime.

Published:
  - universe.json: all securities with ticker, name, sector, cohort
  - prices_sample.json: last 90 days of prices for a sample of tickers (for demo charts)
  - signals_summary.json: per-(feature, horizon, cohort) IC and FMB summary statistics
  - ic_series.json: per-(feature, horizon, cohort, date) rolling IC values
  - quintile_returns.json: per-(feature, horizon, cohort, quintile) mean excess returns
  - manifest.json: build metadata, file hashes, "as of" dates

NOT published: raw price files (redistribution not permitted by Stooq/Tiingo terms).
Published charts and statistics are derived, not raw vendor data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, SupportsInt, cast

import numpy as np
import pandas as pd
import structlog

from pipeline.evaluation.engine import load_eval_results, load_ic_series, load_quintile_returns
from pipeline.ledger.scorer import load_scored_ledger
from pipeline.sources.prices import load_prices
from pipeline.universe.security_master import load_security_master

log = structlog.get_logger()

# Sample tickers for demo charts on home page (not all 500)
SAMPLE_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "GME", "AMC"]


def build_universe_json(
    data_repo_path: Path,
    site_data_path: Path,
) -> dict[str, Any]:
    """Generate universe.json — list of all securities."""
    sm = load_security_master(data_repo_path)

    # Only active securities (valid_to is null)
    active = sm[sm["valid_to"].isna()].copy()

    records = []
    for _, row in active.iterrows():
        records.append(
            {
                "security_id": row["security_id"],
                "ticker": row["ticker"],
                "name": row["name"],
                "gics_sector": row["gics_sector"],
                "cohort": row["cohort"],
            }
        )

    # Sort: sp500 by sector then ticker, retail_attention after
    records.sort(key=lambda r: (0 if r["cohort"] == "sp500" else 1, r["gics_sector"], r["ticker"]))

    out = {
        "as_of": date.today().isoformat(),
        "sp500_count": sum(1 for r in records if r["cohort"] == "sp500"),
        "retail_attention_count": sum(1 for r in records if r["cohort"] == "retail_attention"),
        "securities": records,
    }

    site_data_path.mkdir(parents=True, exist_ok=True)
    path = site_data_path / "universe.json"
    path.write_text(json.dumps(out, indent=2))
    log.info("wrote universe.json", path=str(path), securities=len(records))
    return out


def build_prices_sample_json(
    data_repo_path: Path,
    site_data_path: Path,
    lookback_days: int = 90,
) -> dict[str, Any]:
    """
    Generate prices_sample.json — last N days of close prices for sample tickers.
    Only close and adj_close (where available); no raw OHLCV in published JSON.
    """
    end = date.today()
    start = date.fromordinal(end.toordinal() - lookback_days)

    prices = load_prices(data_repo_path, tickers=SAMPLE_TICKERS, start=start, end=end)

    if prices.empty:
        log.warning("no price data found for sample tickers")
        out: dict[str, Any] = {"as_of": end.isoformat(), "note": "no data yet", "series": {}}
        (site_data_path / "prices_sample.json").write_text(json.dumps(out, indent=2))
        return out

    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    series: dict[str, list[dict[str, Any]]] = {}
    for ticker_upper in SAMPLE_TICKERS:
        ticker_rows = prices[
            prices["security_id"].str.upper().isin([ticker_upper])
            | (prices.get("ticker", "") == ticker_upper)
        ]
        if ticker_rows.empty:
            continue

        series[ticker_upper] = [
            {
                "date": row["date"].isoformat(),
                "close": round(float(row["close"]), 4),
                "adj_close": round(float(row["adj_close"]), 4)
                if pd.notna(row.get("adj_close"))
                else None,
                "data_type": row.get("data_type", "unknown"),
            }
            for _, row in ticker_rows.sort_values("date").iterrows()
        ]

    out = {
        "as_of": end.isoformat(),
        "lookback_days": lookback_days,
        "tickers": SAMPLE_TICKERS,
        "note": "close and adj_close only; raw OHLCV not published per vendor terms",
        "series": series,
    }

    path = site_data_path / "prices_sample.json"
    path.write_text(json.dumps(out, indent=2))
    log.info("wrote prices_sample.json", path=str(path), tickers=len(series))
    return out


def build_manifest(
    site_data_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate manifest.json with build metadata and file hashes."""
    files: dict[str, Any] = {}
    for json_file in sorted(site_data_path.glob("*.json")):
        if json_file.name == "manifest.json":
            continue
        content = json_file.read_bytes()
        files[json_file.name] = {
            "sha256": hashlib.sha256(content).hexdigest()[:16],
            "size_bytes": len(content),
        }

    run_at = datetime.now(tz=UTC).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "built_at": run_at,
        "last_pipeline_run_at": run_at,
        "files": files,
    }
    if extra:
        manifest.update(extra)

    path = site_data_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    log.info("wrote manifest.json", path=str(path), files=len(files))
    return manifest


def build_signals_summary_json(
    data_repo_path: Path,
    site_data_path: Path,
) -> dict[str, Any]:
    """
    Generate signals_summary.json — IC and FMB results for the Signals pages.

    Contains only summary statistics (mean IC, t-stat, BH-adjusted p-value,
    significant_bh) for each (feature, horizon, cohort, eval_type, metric).
    No raw return data; all statistics are derived.
    """
    df = load_eval_results(data_repo_path)
    if df.empty:
        out: dict[str, Any] = {
            "as_of": date.today().isoformat(),
            "note": "evaluation not yet run",
            "results": [],
        }
        path = site_data_path / "signals_summary.json"
        path.write_text(json.dumps(out, indent=2))
        return out

    records = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {
            "feature": row["feature"],
            "horizon": int(row["horizon"]),
            "cohort": row["cohort"],
            "eval_type": row["eval_type"],
            "metric": row["metric"],
            "value": _safe_float(row.get("value")),
            "t_stat": _safe_float(row.get("t_stat")),
            "p_value_raw": _safe_float(row.get("p_value_raw")),
            "p_value_bh": _safe_float(row.get("p_value_bh")),
            "significant_bh": bool(row["significant_bh"])
            if pd.notna(row.get("significant_bh"))
            else None,
            "n_periods": int(row["n_periods"]) if pd.notna(row.get("n_periods")) else None,
        }
        records.append(rec)

    out = {
        "as_of": date.today().isoformat(),
        "n_primary_tests": len(df[(df["metric"] == "ic") & (df["eval_type"] == "is")]),
        "fdr_q": 0.05,
        "results": records,
    }
    path = site_data_path / "signals_summary.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    log.info("wrote signals_summary.json", path=str(path), records=len(records))
    return out


def build_ic_series_json(
    data_repo_path: Path,
    site_data_path: Path,
) -> dict[str, Any]:
    """Generate ic_series.json — rolling IC time series for Signal detail pages."""
    df = load_ic_series(data_repo_path)
    if df.empty:
        out: dict[str, Any] = {"as_of": date.today().isoformat(), "series": {}}
        (site_data_path / "ic_series.json").write_text(json.dumps(out, indent=2))
        return out

    # Structure as nested dict: eval_type → feature → cohort → horizon → [{date, ic}]
    series: dict[str, Any] = {}
    for (eval_type, feature, cohort, horizon), grp in df.groupby(
        ["eval_type", "feature", "cohort", "horizon"]
    ):
        key = f"{eval_type}/{feature}/{cohort}/{horizon}"
        grp_sorted = grp.sort_values("date")
        series[key] = [
            {"date": str(row["date"]), "ic": _safe_float(row["ic"])}
            for _, row in grp_sorted.iterrows()
        ]

    out = {"as_of": date.today().isoformat(), "series": series}
    path = site_data_path / "ic_series.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    log.info("wrote ic_series.json", path=str(path), keys=len(series))
    return out


def build_quintile_returns_json(
    data_repo_path: Path,
    site_data_path: Path,
) -> dict[str, Any]:
    """Generate quintile_returns.json — quintile excess return means for bar charts."""
    df = load_quintile_returns(data_repo_path)
    if df.empty:
        out: dict[str, Any] = {"as_of": date.today().isoformat(), "quintiles": {}}
        (site_data_path / "quintile_returns.json").write_text(json.dumps(out, indent=2))
        return out

    quintiles: dict[str, Any] = {}
    for (eval_type, feature, cohort, horizon), grp in df.groupby(
        ["eval_type", "feature", "cohort", "horizon"]
    ):
        key = f"{eval_type}/{feature}/{cohort}/{horizon}"
        q_dict = {
            int(row["quintile"]): _safe_float(row["mean_excess_return"])
            for _, row in grp.iterrows()
        }
        quintiles[key] = q_dict

    out = {"as_of": date.today().isoformat(), "quintiles": quintiles}
    path = site_data_path / "quintile_returns.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    log.info("wrote quintile_returns.json", path=str(path), keys=len(quintiles))
    return out


def _safe_float(v: Any) -> float | None:
    """Convert a value to float, returning None for NaN/None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else round(f, 8)  # NaN check
    except (TypeError, ValueError):
        return None


def build_ledger_json(
    data_repo_path: Path,
    site_data_path: Path,
) -> dict[str, Any]:
    """
    Generate ledger.json — predictions + outcomes for the Predictions Ledger page.

    Only includes: prediction_id, security_id, made_at, target_date_start/end,
    horizon_days, model_id, predicted_direction, predicted_excess_return,
    ci_lower, ci_upper, realized_excess_return, scored_at.
    Does NOT include data_hash, features_used, or model internals.
    """
    df = load_scored_ledger(data_repo_path)
    if df.empty:
        out: dict[str, Any] = {
            "as_of": date.today().isoformat(),
            "note": "no predictions yet",
            "predictions": [],
        }
        path = site_data_path / "ledger.json"
        path.write_text(json.dumps(out, indent=2))
        return out

    keep_cols = [
        "prediction_id",
        "security_id",
        "made_at",
        "target_date_start",
        "target_date_end",
        "horizon_days",
        "model_id",
        "predicted_direction",
        "predicted_excess_return",
        "ci_lower",
        "ci_upper",
        "realized_excess_return",
        "scored_at",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Scoring summary
    scored = df[df["realized_excess_return"].notna()]
    n_scored = len(scored)
    hit_rate = None
    if n_scored > 0:
        correct = (scored["predicted_direction"] == 1) & (scored["realized_excess_return"] > 0) | (
            scored["predicted_direction"] == -1
        ) & (scored["realized_excess_return"] < 0)
        hit_rate = round(float(correct.mean()), 4)

    records = []
    for _, row in df.sort_values("made_at", ascending=False).head(5000).iterrows():
        rec: dict[str, Any] = {}
        for col in df.columns:
            v = row[col]
            if pd.isna(v) if not isinstance(v, list | dict) else False:
                rec[col] = None
            elif hasattr(v, "isoformat"):
                rec[col] = v.isoformat()
            elif isinstance(v, np.integer):
                rec[col] = int(v)
            elif isinstance(v, np.floating):
                rec[col] = _safe_float(v)
            else:
                rec[col] = v
        records.append(rec)

    out = {
        "as_of": date.today().isoformat(),
        "n_total": len(df),
        "n_scored": n_scored,
        "hit_rate": hit_rate,
        "predictions": records,
    }
    path = site_data_path / "ledger.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    log.info("wrote ledger.json", path=str(path), records=len(records))
    return out


def build_eval_results_json(
    data_repo_path: Path,
    site_data_path: Path,
) -> dict[str, Any]:
    """
    Generate eval_results.json — per-(feature, horizon_days, cohort) IC summary
    with embedded ic_series, in the schema data.ts expects.
    """
    results_df = load_eval_results(data_repo_path)
    ic_df = load_ic_series(data_repo_path)
    quintile_df = load_quintile_returns(data_repo_path)

    empty_out: dict[str, Any] = {
        "as_of": date.today().isoformat(),
        "run_id": "",
        "results": [],
    }

    if results_df.empty:
        path = site_data_path / "eval_results.json"
        path.write_text(json.dumps(empty_out, indent=2))
        return empty_out

    # Build ic_series lookup: (feature, cohort, horizon) -> [{date, ic}]
    ic_lookup: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    if not ic_df.empty:
        for (feature, cohort, horizon), grp in ic_df.groupby(["feature", "cohort", "horizon"]):
            ic_lookup[(str(feature), str(cohort), int(cast("SupportsInt", horizon)))] = [
                {"date": str(row["date"]), "ic": _safe_float(row["ic"])}
                for _, row in grp.sort_values("date").iterrows()
            ]

    # Build quintile_spread lookup: (feature, cohort, horizon) -> spread
    q_lookup: dict[tuple[str, str, int], float | None] = {}
    if not quintile_df.empty:
        for (feature, cohort, horizon), grp in quintile_df.groupby(
            ["feature", "cohort", "horizon"]
        ):
            q5 = (
                grp[grp["quintile"] == 5]["mean_excess_return"].iloc[0]
                if len(grp[grp["quintile"] == 5])
                else None
            )
            q1 = (
                grp[grp["quintile"] == 1]["mean_excess_return"].iloc[0]
                if len(grp[grp["quintile"] == 1])
                else None
            )
            if q5 is not None and q1 is not None:
                q_lookup[(str(feature), str(cohort), int(cast("SupportsInt", horizon)))] = (
                    _safe_float(float(q5) - float(q1))
                )

    # Pivot: one row per (feature, horizon, cohort) with IC + FMB stats
    ic_rows = results_df[(results_df["metric"] == "ic") & (results_df["eval_type"] == "oos")].copy()

    records = []
    for _, row in ic_rows.iterrows():
        feature = str(row["feature"])
        cohort = str(row["cohort"])
        horizon = int(row["horizon"])
        key = (feature, cohort, horizon)

        # Compute n_obs = n_periods * approx securities per period (use n_periods as proxy)
        n_dates = int(row["n_periods"]) if pd.notna(row.get("n_periods")) else 0

        rec: dict[str, Any] = {
            "feature": feature,
            "horizon_days": horizon,
            "cohort": cohort,
            "ic_mean": _safe_float(row.get("value")),
            "ic_std": None,  # not stored separately; could be derived from t_stat + n
            "ic_tstat": _safe_float(row.get("t_stat")),
            "ic_pvalue": _safe_float(row.get("p_value_bh")),
            "ic_bh_reject": bool(row["significant_bh"])
            if pd.notna(row.get("significant_bh"))
            else False,
            "quintile_spread": q_lookup.get(key),
            "hit_rate": None,  # computed separately if available
            "n_obs": n_dates,  # best proxy without security-level count
            "n_dates": n_dates,
            "ic_series": ic_lookup.get(key, []),
        }
        records.append(rec)

    # Pull run_id from evaluation manifest if it exists
    eval_manifest_path = data_repo_path / "processed" / "evaluation" / "manifest.json"
    run_id = ""
    if eval_manifest_path.exists():
        import contextlib

        with contextlib.suppress(Exception):
            run_id = json.loads(eval_manifest_path.read_text()).get("run_id", "")

    out: dict[str, Any] = {
        "as_of": date.today().isoformat(),
        "run_id": run_id,
        "results": records,
    }
    path = site_data_path / "eval_results.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    log.info("wrote eval_results.json", path=str(path), records=len(records))
    return out


def build_predictions_json(
    data_repo_path: Path,
    site_data_path: Path,
    max_rows: int = 5000,
) -> dict[str, Any]:
    """
    Generate predictions.json — ledger + outcomes for the Predictions Ledger page.
    Schema matches data.ts Prediction type.
    """
    df = load_scored_ledger(data_repo_path)

    empty_out: dict[str, Any] = {
        "as_of": date.today().isoformat(),
        "predictions": [],
    }
    if df.empty:
        path = site_data_path / "predictions.json"
        path.write_text(json.dumps(empty_out, indent=2))
        return empty_out

    # Load security master to join ticker
    try:
        from pipeline.universe.security_master import load_security_master

        sm = load_security_master(data_repo_path)
        sid_to_ticker = sm.set_index("security_id")["ticker"].to_dict()
    except Exception:
        sid_to_ticker = {}

    records = []
    df_sorted = df.sort_values("made_at", ascending=False).head(max_rows)

    for _, row in df_sorted.iterrows():
        security_id = str(row.get("security_id", ""))
        realized = _safe_float(row.get("realized_excess_return"))
        predicted = _safe_float(row.get("predicted_excess_return"))

        # Derive direction + correctness
        pred_dir = row.get("predicted_direction")
        if isinstance(pred_dir, int | np.integer):
            direction = "up" if int(pred_dir) == 1 else "down"
        else:
            direction = str(pred_dir) if pred_dir else "up"

        correct: bool | None = None
        if realized is not None and predicted is not None:
            correct = (realized > 0) == (predicted > 0)

        ci_lower = _safe_float(row.get("ci_lower"))
        ci_upper = _safe_float(row.get("ci_upper"))

        rec: dict[str, Any] = {
            "prediction_id": str(row.get("prediction_id", "")),
            "made_at": row["made_at"].isoformat()
            if hasattr(row.get("made_at"), "isoformat")
            else str(row.get("made_at", "")),
            "security_id": security_id,
            "ticker": sid_to_ticker.get(security_id, security_id),
            "target_date_start": row["target_date_start"].isoformat()
            if hasattr(row.get("target_date_start"), "isoformat")
            else str(row.get("target_date_start", "")),
            "target_date_end": row["target_date_end"].isoformat()
            if hasattr(row.get("target_date_end"), "isoformat")
            else str(row.get("target_date_end", "")),
            "horizon_days": int(row.get("horizon_days", 0)),
            "model_id": str(row.get("model_id", "")),
            "predicted_direction": direction,
            "predicted_excess_return": predicted,
            "confidence_interval": [ci_lower, ci_upper],
            "realized_excess_return": realized,
            "scored_at": row["scored_at"].isoformat()
            if hasattr(row.get("scored_at"), "isoformat")
            else None,
            "correct": correct,
        }
        records.append(rec)

    out: dict[str, Any] = {
        "as_of": date.today().isoformat(),
        "predictions": records,
    }
    path = site_data_path / "predictions.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    log.info("wrote predictions.json", path=str(path), records=len(records))
    return out


def build_pipeline_status_json(
    data_repo_path: Path,
    site_data_path: Path,
) -> dict[str, Any]:
    """
    Generate pipeline_status.json — per-source health and recent run log.
    Reads from {data_repo}/processed/pipeline_runs/ if it exists.
    Falls back to an "unknown" status if no runs have been recorded.
    """
    runs_dir = data_repo_path / "processed" / "pipeline_runs"

    sources = ["wikipedia", "edgar", "gdelt", "trends", "prices"]
    source_statuses = []
    overall = "ok"

    for source in sources:
        source_log = runs_dir / f"{source}_status.json" if runs_dir.exists() else None
        if source_log and source_log.exists():
            try:
                s = json.loads(source_log.read_text())
                source_statuses.append(s)
                if s.get("status") == "error":
                    overall = "error"
                elif s.get("status") == "stale" and overall != "error":
                    overall = "degraded"
            except Exception:
                source_statuses.append(
                    {
                        "source": source,
                        "last_success": None,
                        "last_attempt": None,
                        "status": "error",
                        "records_today": 0,
                        "error_message": "could not read status file",
                    }
                )
        else:
            source_statuses.append(
                {
                    "source": source,
                    "last_success": None,
                    "last_attempt": None,
                    "status": "never",
                    "records_today": 0,
                    "error_message": None,
                }
            )
            if overall == "ok":
                overall = "degraded"

    # Recent runs log
    recent_runs = []
    run_log_path = runs_dir / "runs.jsonl" if runs_dir.exists() else None
    if run_log_path and run_log_path.exists():
        try:
            lines = run_log_path.read_text().strip().splitlines()
            for line in reversed(lines[-20:]):
                recent_runs.append(json.loads(line))
        except Exception:
            pass

    out: dict[str, Any] = {
        "as_of": datetime.now(tz=UTC).isoformat(),
        "overall_status": overall,
        "sources": source_statuses,
        "recent_runs": recent_runs,
    }
    path = site_data_path / "pipeline_status.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    log.info("wrote pipeline_status.json", path=str(path), overall=overall)
    return out


def build_all(
    data_repo_path: Path,
    site_data_path: Path,
) -> None:
    """Build all published site data files."""
    log.info("building site data", data_repo=str(data_repo_path), site_data=str(site_data_path))
    build_universe_json(data_repo_path, site_data_path)
    build_prices_sample_json(data_repo_path, site_data_path)
    build_signals_summary_json(data_repo_path, site_data_path)
    build_ic_series_json(data_repo_path, site_data_path)
    build_quintile_returns_json(data_repo_path, site_data_path)
    build_eval_results_json(data_repo_path, site_data_path)
    build_predictions_json(data_repo_path, site_data_path)
    build_pipeline_status_json(data_repo_path, site_data_path)
    build_ledger_json(data_repo_path, site_data_path)  # keep for backwards compat
    build_manifest(site_data_path)
    log.info("site data build complete")

"""
Prediction ledger writer — frozen, immutable daily predictions.

Invariants (from PREREGISTRATION.md and principle §5):
  - Each prediction row has a UUID, a timestamp, and a SHA-256 hash of the
    input feature data. Once committed to the archive these rows are NEVER
    modified or deleted.
  - Predictions are written BEFORE outcomes are known (the daily run fires at
    18:30 ET, after market close; outcomes for that day are not yet available).
  - model_version and data_hash let anyone reproduce the exact prediction
    from the archived feature data.

Output: appended to {data_repo}/processed/ledger/predictions.parquet
        (written as a separate date-partitioned file to preserve immutability).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from pipeline.evaluation.engine import _load_eval_data
from pipeline.evaluation.walkforward import TRAIN_START
from pipeline.models.base import BaseModel
from pipeline.validate.schemas import PREDICTION_LEDGER_SCHEMA, SCHEMA_VERSION

log = structlog.get_logger()

HORIZONS = (1, 5, 21)


# ---------------------------------------------------------------------------
# Feature hash
# ---------------------------------------------------------------------------


def _hash_features(df: pd.DataFrame) -> str:
    """SHA-256 of the sorted feature DataFrame contents. Deterministic."""
    canonical = df.sort_values(["security_id", "feature_name"]).to_csv(index=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def _train_model(
    model: BaseModel,
    data_repo: Path,
    horizon: int,
    as_of: date,
) -> BaseModel:
    """Train model on all data up to as_of (exclusive of the as_of day)."""
    from datetime import timedelta

    train_data = _load_eval_data(data_repo, start=TRAIN_START, end=as_of - timedelta(days=1))
    if train_data.empty:
        raise RuntimeError(f"No training data found up to {as_of}.")

    horizon_data = train_data[train_data["horizon"] == horizon].copy()
    if len(horizon_data) < 50:
        raise RuntimeError(f"Insufficient training rows for horizon={horizon}: {len(horizon_data)}")

    val_col = "shrunk_value" if "shrunk_value" in horizon_data.columns else "feature_value"

    # Pivot to wide format: one row per (security_id, date), one column per feature
    X = horizon_data.pivot_table(
        index=["security_id", "date"], columns="feature_name", values=val_col
    ).reset_index()
    y_df = horizon_data.drop_duplicates(["security_id", "date"])[
        ["security_id", "date", "excess_return"]
    ]
    Xy = X.merge(y_df, on=["security_id", "date"])

    y = Xy["excess_return"]
    X_feat = Xy.drop(columns=["security_id", "date", "excess_return"], errors="ignore")

    model.fit(X_feat, y)
    log.info(
        "ledger.model_trained",
        model_id=model.model_id,
        horizon=horizon,
        n_rows=len(Xy),
        as_of=str(as_of),
    )
    return model


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------


def _load_today_features(
    data_repo: Path,
    as_of: date,
) -> pd.DataFrame:
    """Load today's feature data — the inputs to tomorrow's prediction."""
    from pipeline.features.builder import load_features

    features = load_features(data_repo, start_date=as_of, end_date=as_of)
    return features


def _make_prediction_rows(
    model: BaseModel,
    features_wide: pd.DataFrame,
    security_ids: list[str],
    as_of: date,
    horizon: int,
    data_hash: str,
    cohort_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Generate one ledger row per security for a given model + horizon."""
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")

    # Compute target_date_start = next trading day after as_of
    from datetime import timedelta

    candidate = as_of + timedelta(days=1)
    schedule = nyse.schedule(
        start_date=candidate.isoformat(),
        end_date=(candidate + timedelta(days=14)).isoformat(),
    )
    if schedule.empty:
        raise RuntimeError(f"Cannot find next trading day after {as_of}")
    trading_days = sorted(d.date() if hasattr(d, "date") else d for d in schedule.index)
    target_start = trading_days[0]
    target_end = trading_days[min(horizon - 1, len(trading_days) - 1)]

    preds, ci_lower, ci_upper = model.predict(features_wide)
    made_at = datetime.now(UTC)

    rows = []
    for i, sid in enumerate(security_ids):
        pred = float(preds[i])
        direction = 1 if pred > 0 else (-1 if pred < 0 else 0)
        rows.append(
            {
                "prediction_id": str(uuid.uuid4()),
                "made_at": made_at,
                "security_id": sid,
                "target_date_start": target_start,
                "target_date_end": target_end,
                "horizon_days": horizon,
                "model_id": model.model_id,
                "model_version": model.model_version,
                "data_hash": data_hash,
                "predicted_excess_return": pred,
                "predicted_direction": direction,
                "ci_lower": float(ci_lower[i]),
                "ci_upper": float(ci_upper[i]),
                "features_used_json": json.dumps(list(features_wide.columns)),
                "schema_version": SCHEMA_VERSION,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Ledger append (immutable)
# ---------------------------------------------------------------------------


def _append_to_ledger(rows: list[dict], data_repo: Path, as_of: date) -> Path:
    """
    Write a date-partitioned Parquet file.  Never overwrites existing files.
    Returns the path written.
    """
    out_dir = data_repo / "processed" / "ledger" / f"date={as_of.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "predictions.parquet"
    if out_path.exists():
        raise FileExistsError(
            f"Ledger file already exists for {as_of}: {out_path}. "
            "Predictions are immutable; delete the file manually if you intend to re-run."
        )

    df = pd.DataFrame(rows)
    df["made_at"] = pd.to_datetime(df["made_at"], utc=True)
    df["target_date_start"] = pd.to_datetime(df["target_date_start"])
    df["target_date_end"] = pd.to_datetime(df["target_date_end"])
    df["horizon_days"] = df["horizon_days"].astype("int32")
    df["predicted_direction"] = df["predicted_direction"].astype("int8")

    table = pa.Table.from_pandas(df, schema=PREDICTION_LEDGER_SCHEMA, safe=False)
    pq.write_table(table, out_path, compression="snappy")
    log.info("ledger.written", path=str(out_path), rows=len(rows))
    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def write_daily_predictions(
    data_repo: Path,
    as_of: date | None = None,
    models: list[BaseModel] | None = None,
) -> list[Path]:
    """
    Generate and commit today's frozen predictions.

    Steps:
      1. Load today's features.
      2. For each model × horizon: train on history, predict, write ledger.
      3. Return list of written paths.

    Idempotent check: raises FileExistsError if ledger for as_of already exists.
    """
    from pipeline.models.baselines import MomentumModel, ZeroModel
    from pipeline.models.ridge import RidgeModel
    from pipeline.universe.security_master import load_security_master

    as_of = as_of or date.today()

    features = _load_today_features(data_repo, as_of)
    if features.empty:
        log.warning("ledger.no_features", as_of=str(as_of))
        return []

    data_hash = _hash_features(features)
    log.info("ledger.data_hash", as_of=str(as_of), hash=data_hash[:12])

    sm = load_security_master(data_repo)
    active_sm = sm[sm["valid_to"].isna()][["security_id", "cohort"]].copy()
    cohort_map = dict(zip(active_sm["security_id"], active_sm["cohort"], strict=False))

    if models is None:
        models = [ZeroModel(), MomentumModel(), RidgeModel(horizon=1)]

    written: list[Path] = []
    for horizon in HORIZONS:
        val_col = "shrunk_value" if "shrunk_value" in features.columns else "value"
        feat_wide = features.pivot_table(
            index="security_id", columns="feature_name", values=val_col
        ).fillna(0.0)
        security_ids = list(feat_wide.index)
        if not security_ids:
            continue

        for model in models:
            # Clone the model so each horizon gets a fresh fit
            try:
                m = pickle.loads(pickle.dumps(model))
                m = _train_model(m, data_repo, horizon=horizon, as_of=as_of)
            except Exception as exc:
                log.warning(
                    "ledger.model_train_failed",
                    model=model.model_id,
                    horizon=horizon,
                    error=str(exc),
                )
                continue

            rows = _make_prediction_rows(
                model=m,
                features_wide=feat_wide,
                security_ids=security_ids,
                as_of=as_of,
                horizon=horizon,
                data_hash=data_hash,
                cohort_map=cohort_map,
            )
            try:
                path = _append_to_ledger(rows, data_repo, as_of)
                written.append(path)
            except FileExistsError as exc:
                log.warning("ledger.already_exists", error=str(exc))

    return written


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------


def load_ledger(data_repo: Path) -> pd.DataFrame:
    """Load all prediction rows from the ledger (all dates)."""
    ledger_dir = data_repo / "processed" / "ledger"
    if not ledger_dir.exists():
        return pd.DataFrame()

    parts = []
    for f in sorted(ledger_dir.rglob("predictions.parquet")):
        parts.append(pq.read_table(f).to_pandas())

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)

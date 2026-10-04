"""
Prediction ledger writer — frozen, immutable daily predictions.

Invariants (from PREREGISTRATION.md and principle §5):
  - Each prediction row has a UUID, a timestamp, and a SHA-256 hash of the
    input feature data. Once committed to the archive these rows are NEVER
    modified or deleted.
  - Predictions are written BEFORE outcomes are known (the daily run fires at
    18:30 ET, after market close; outcomes for that day are not yet available).
  - Models are trained only on returns fully realized by the as-of date, so a
    prediction made for a past as_of (manual backfill) cannot see the future.
  - model_version and data_hash let anyone reproduce the exact prediction
    from the archived feature data.

Output: one immutable file per (as_of, model, horizon):
  {data_repo}/processed/ledger/date={YYYY-MM-DD}/model={model_id}/horizon={h}/predictions.parquet
"""

from __future__ import annotations

import hashlib
import json
import pickle
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pandas_market_calendars as mcal
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from pipeline.evaluation.engine import _load_eval_data
from pipeline.evaluation.returns import compute_trailing_excess_returns
from pipeline.evaluation.walkforward import TRAIN_START
from pipeline.models.base import BaseModel
from pipeline.models.baselines import MomentumModel
from pipeline.validate.schemas import PREDICTION_LEDGER_SCHEMA, SCHEMA_VERSION

log = structlog.get_logger()

HORIZONS = (1, 5, 21)
MOM_COL = MomentumModel._MOM_COL

_NYSE = mcal.get_calendar("NYSE")


# ---------------------------------------------------------------------------
# Feature hash
# ---------------------------------------------------------------------------


def _hash_features(df: pd.DataFrame) -> str:
    """SHA-256 of the sorted feature DataFrame contents. Deterministic."""
    canonical = df.sort_values(["security_id", "feature_name"]).to_csv(index=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Target window
# ---------------------------------------------------------------------------


def _target_window(as_of: date, horizon: int) -> tuple[date, date]:
    """
    First and last trading day of the return window a prediction made after
    the close on as_of refers to: trading days t+1 .. t+horizon.
    """
    # Two calendar days per trading day plus a holiday buffer always covers the horizon
    schedule = _NYSE.schedule(
        start_date=(as_of + timedelta(days=1)).isoformat(),
        end_date=(as_of + timedelta(days=horizon * 2 + 14)).isoformat(),
    )
    trading_days = sorted(d.date() for d in schedule.index)
    if len(trading_days) < horizon:
        raise RuntimeError(f"Cannot find {horizon} trading days after {as_of}")
    return trading_days[0], trading_days[horizon - 1]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def _training_frame(
    train_data: pd.DataFrame,
    horizon: int,
    momentum: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Wide (security, date) × feature matrix plus target for one horizon."""
    horizon_data = train_data[train_data["horizon"] == horizon]
    if len(horizon_data) < 50:
        raise RuntimeError(f"Insufficient training rows for horizon={horizon}: {len(horizon_data)}")

    val_col = "shrunk_value" if "shrunk_value" in horizon_data.columns else "feature_value"

    # Pivot to wide format: one row per (security_id, date), one column per feature
    X = horizon_data.pivot_table(
        index=["security_id", "date"], columns="feature_name", values=val_col
    ).reset_index()
    y_df = horizon_data.drop_duplicates(["security_id", "date"])[
        ["security_id", "ticker", "date", "excess_return"]
    ]
    Xy = X.merge(y_df, on=["security_id", "date"])
    Xy = Xy.merge(
        momentum.rename(columns={"trailing_excess_return": MOM_COL}),
        on=["ticker", "date"],
        how="left",
    )

    y = Xy["excess_return"]
    X_feat = Xy.drop(columns=["security_id", "ticker", "date", "excess_return"], errors="ignore")
    return X_feat, y


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------


def _load_today_features(
    data_repo: Path,
    as_of: date,
) -> pd.DataFrame:
    """Load today's feature data — the inputs to tomorrow's prediction."""
    from pipeline.features.builder import load_features

    return load_features(data_repo, start_date=as_of, end_date=as_of)


def _latest_momentum(momentum: pd.DataFrame, as_of: date) -> dict[str, float]:
    """Most recent trailing excess return per ticker on or before as_of."""
    if momentum.empty:
        return {}
    known = momentum[momentum["date"] <= as_of].sort_values("date")
    latest = known.drop_duplicates("ticker", keep="last")
    return dict(zip(latest["ticker"], latest["trailing_excess_return"], strict=True))


def _make_prediction_rows(
    model: BaseModel,
    features_wide: pd.DataFrame,
    security_ids: list[str],
    as_of: date,
    horizon: int,
    data_hash: str,
) -> list[dict[str, Any]]:
    """Generate one ledger row per security for a given model + horizon."""
    target_start, target_end = _target_window(as_of, horizon)

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


def _ledger_path(data_repo: Path, as_of: date, model_id: str, horizon: int) -> Path:
    return (
        data_repo
        / "processed"
        / "ledger"
        / f"date={as_of.isoformat()}"
        / f"model={model_id}"
        / f"horizon={horizon}"
        / "predictions.parquet"
    )


def _append_to_ledger(rows: list[dict[str, Any]], data_repo: Path, as_of: date) -> Path:
    """
    Write one immutable Parquet file for a single (as_of, model, horizon).
    Never overwrites existing files. Returns the path written.
    """
    keys = {(r["model_id"], int(r["horizon_days"])) for r in rows}
    if len(keys) != 1:
        raise ValueError(f"Ledger rows must share one model and horizon; got {sorted(keys)}")
    model_id, horizon = keys.pop()

    out_path = _ledger_path(data_repo, as_of, model_id, horizon)
    if out_path.exists():
        raise FileExistsError(
            f"Ledger file already exists for {as_of} {model_id} h={horizon}: {out_path}. "
            "Predictions are immutable; delete the file manually if you intend to re-run."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
      2. Load training data once, capped so no return realized after as_of is used.
      3. For each horizon × model: train on history, predict, write its own ledger file.
      4. Return list of written paths.

    Already-written (as_of, model, horizon) files are skipped, never overwritten.
    """
    from pipeline.models.baselines import ZeroModel
    from pipeline.models.ridge import RidgeModel
    from pipeline.universe.security_master import load_security_master

    as_of = as_of or date.today()

    features = _load_today_features(data_repo, as_of)
    if features.empty:
        log.warning("ledger.no_features", as_of=str(as_of))
        return []

    data_hash = _hash_features(features)
    log.info("ledger.data_hash", as_of=str(as_of), hash=data_hash[:12])

    train_data = _load_eval_data(
        data_repo, start=TRAIN_START, end=as_of - timedelta(days=1), prices_until=as_of
    )
    if train_data.empty:
        log.warning("ledger.no_training_data", as_of=str(as_of))
        return []

    momentum = compute_trailing_excess_returns(data_repo, start=min(train_data["date"]), end=as_of)

    sm = load_security_master(data_repo)
    active_sm = sm[sm["valid_to"].isna()]
    sid_to_ticker = dict(zip(active_sm["security_id"], active_sm["ticker"], strict=False))

    val_col = "shrunk_value" if "shrunk_value" in features.columns else "value"
    feat_wide = features.pivot_table(
        index="security_id", columns="feature_name", values=val_col
    ).fillna(0.0)
    latest_mom = _latest_momentum(momentum, as_of)
    feat_wide[MOM_COL] = [
        latest_mom.get(sid_to_ticker.get(sid, ""), 0.0) for sid in feat_wide.index
    ]
    security_ids = [str(s) for s in feat_wide.index]
    if not security_ids:
        return []

    if models is None:
        models = [ZeroModel(), MomentumModel(), RidgeModel(horizon=1)]

    written: list[Path] = []
    for horizon in HORIZONS:
        try:
            X_train, y_train = _training_frame(train_data, horizon, momentum)
        except RuntimeError as exc:
            log.warning("ledger.no_training_frame", horizon=horizon, error=str(exc))
            continue

        for model in models:
            if _ledger_path(data_repo, as_of, model.model_id, horizon).exists():
                log.info("ledger.already_exists", model=model.model_id, horizon=horizon)
                continue
            # Clone the model so each horizon gets a fresh fit
            m = cast("BaseModel", pickle.loads(pickle.dumps(model)))
            try:
                m.fit(X_train, y_train)
            except Exception as exc:
                log.warning(
                    "ledger.model_train_failed",
                    model=model.model_id,
                    horizon=horizon,
                    error=str(exc),
                )
                continue
            log.info(
                "ledger.model_trained",
                model_id=m.model_id,
                horizon=horizon,
                n_rows=len(X_train),
                as_of=str(as_of),
            )

            rows = _make_prediction_rows(
                model=m,
                features_wide=feat_wide,
                security_ids=security_ids,
                as_of=as_of,
                horizon=horizon,
                data_hash=data_hash,
            )
            written.append(_append_to_ledger(rows, data_repo, as_of))

    return written


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------


def load_ledger(data_repo: Path) -> pd.DataFrame:
    """Load all prediction rows from the ledger (all dates, models, horizons)."""
    ledger_dir = data_repo / "processed" / "ledger"
    if not ledger_dir.exists():
        return pd.DataFrame()

    parts = []
    for f in sorted(ledger_dir.rglob("predictions.parquet")):
        parts.append(pq.read_table(f).to_pandas())

    if not parts:
        return pd.DataFrame()

    return cast("pd.DataFrame", pd.concat(parts, ignore_index=True))

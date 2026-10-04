"""
Ledger scorer — attach realized excess returns to frozen predictions.

Runs after prices are updated. For each unscored prediction whose
target_date_end has passed, computes the realized excess return and
writes an outcome row.

Output appended to: {data_repo}/processed/ledger/outcomes/
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from pipeline.evaluation.returns import compute_excess_returns
from pipeline.ledger.writer import load_ledger
from pipeline.utils.calendar import prev_trading_day
from pipeline.validate.schemas import OUTCOME_SCHEMA, SCHEMA_VERSION

log = structlog.get_logger()


def _load_outcomes(data_repo: Path) -> pd.DataFrame:
    out_dir = data_repo / "processed" / "ledger" / "outcomes"
    if not out_dir.exists():
        return pd.DataFrame()
    parts = []
    for f in sorted(out_dir.rglob("*.parquet")):
        parts.append(pq.read_table(f).to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def score_predictions(data_repo: Path, as_of: date | None = None) -> pd.DataFrame:
    """
    Score all predictions whose target_date_end ≤ as_of and that have no
    outcome record yet.

    Returns a DataFrame of newly scored outcomes.
    """
    as_of = as_of or date.today()

    ledger = load_ledger(data_repo)
    if ledger.empty:
        log.info("scorer.no_predictions")
        return pd.DataFrame()

    existing_outcomes = _load_outcomes(data_repo)
    scored_ids: set[str] = (
        set(existing_outcomes["prediction_id"].tolist()) if not existing_outcomes.empty else set()
    )

    ledger["target_date_end"] = pd.to_datetime(ledger["target_date_end"]).dt.date
    pending = ledger[
        (ledger["target_date_end"] <= as_of) & (~ledger["prediction_id"].isin(scored_ids))
    ].copy()

    if pending.empty:
        log.info("scorer.nothing_to_score", as_of=str(as_of))
        return pd.DataFrame()

    log.info("scorer.scoring", n=len(pending), as_of=str(as_of))

    # Returns are keyed by decision date d and cover trading days d+1 .. d+h, so the
    # key for a prediction is the last trading day before its target window starts.
    pending["target_date_start"] = pd.to_datetime(pending["target_date_start"]).dt.date
    pending["decision_date"] = [
        date.fromisoformat(prev_trading_day(d.isoformat())) for d in pending["target_date_start"]
    ]
    min_date = pending["decision_date"].min()
    max_date = pending["target_date_end"].max()

    # Fetch realized returns (ticker-level)
    returns_df = compute_excess_returns(
        data_repo,
        start=min_date,
        end=max_date,
        horizons=tuple(int(h) for h in sorted(pending["horizon_days"].unique())),
    )

    if returns_df.empty:
        log.warning("scorer.no_returns_available")
        return pd.DataFrame()

    # Build lookup: (ticker, start_date, horizon) → excess_return
    returns_df["date"] = pd.to_datetime(returns_df["date"]).dt.date
    ret_lookup = {
        (row["ticker"], row["date"], int(row["horizon"])): row["excess_return"]
        for _, row in returns_df.iterrows()
    }

    # Join SecurityMaster to get ticker from security_id
    from pipeline.universe.security_master import load_security_master

    sm = load_security_master(data_repo)
    id_to_ticker = dict(zip(sm["security_id"], sm["ticker"], strict=False))

    new_outcome_rows = []
    scored_at = datetime.now(UTC)

    for _, pred in pending.iterrows():
        ticker = id_to_ticker.get(pred["security_id"])
        if ticker is None:
            continue
        key = (ticker, pred["decision_date"], int(pred["horizon_days"]))
        realized = ret_lookup.get(key)
        if realized is None or np.isnan(realized):
            continue
        new_outcome_rows.append(
            {
                "prediction_id": pred["prediction_id"],
                "realized_excess_return": float(realized),
                "scored_at": scored_at,
                "schema_version": SCHEMA_VERSION,
            }
        )

    if not new_outcome_rows:
        log.info("scorer.no_new_scores")
        return pd.DataFrame()

    new_df = pd.DataFrame(new_outcome_rows)
    new_df["scored_at"] = pd.to_datetime(new_df["scored_at"], utc=True)

    # Write outcomes
    out_dir = data_repo / "processed" / "ledger" / "outcomes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"outcomes_{as_of.isoformat()}.parquet"
    table = pa.Table.from_pandas(new_df, schema=OUTCOME_SCHEMA, safe=False)
    pq.write_table(table, out_path, compression="snappy")

    log.info("scorer.wrote_outcomes", path=str(out_path), n=len(new_df))
    return new_df


def load_scored_ledger(data_repo: Path) -> pd.DataFrame:
    """
    Return the full ledger joined with outcomes.

    Unscored predictions have NaN in realized_excess_return and scored_at.
    """
    ledger = load_ledger(data_repo)
    outcomes = _load_outcomes(data_repo)

    if ledger.empty:
        return pd.DataFrame()
    if outcomes.empty:
        ledger["realized_excess_return"] = np.nan
        ledger["scored_at"] = pd.NaT
        return ledger

    return ledger.merge(
        outcomes[["prediction_id", "realized_excess_return", "scored_at"]],
        on="prediction_id",
        how="left",
    )

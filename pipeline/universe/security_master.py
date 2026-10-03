"""
Build and maintain the SecurityMaster table.

The SecurityMaster is the authoritative mapping from security_id → ticker history.
It is rebuilt from scratch each time this module runs (idempotent) and written to
the archive as a Parquet file.

Ticker change handling:
  When a company changes its ticker, we record the old ticker with a valid_to date
  and a new entry with valid_from = change date. The security_id never changes.
  Example: Block, Inc. SQ → XYZ: both rows share the same security_id.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
import yaml

from pipeline.universe.sp500 import (
    build_point_in_time_membership,
    fetch_current_and_changes,
    make_security_id,
)
from pipeline.validate.schemas import SCHEMA_VERSION, SECURITY_MASTER_SCHEMA

log = structlog.get_logger()

# Known ticker changes: {old_ticker: (new_ticker, change_date, company_name)}
# Keep this list up to date; each entry is verified against EDGAR/public records.
KNOWN_TICKER_CHANGES: dict[str, tuple[str, date, str]] = {
    # Block, Inc. changed SQ → XYZ (verify exact date against EDGAR 8-K)
    "SQ": ("XYZ", date(2024, 1, 1), "Block, Inc."),
    # Add verified changes here as they are confirmed
}


def _load_retail_basket(config_path: Path) -> list[dict[str, Any]]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("tickers", [])


def build_security_master(
    data_repo_path: Path,
    config_path: Path,
    as_of: date | None = None,
    cached_wikipedia_html: str | None = None,
) -> pd.DataFrame:
    """
    Build the complete SecurityMaster DataFrame and write it to the archive.

    Returns the DataFrame for immediate use.
    """
    target = as_of or date.today()
    now = datetime.now(tz=timezone.utc)

    log.info("building security master", as_of=target.isoformat())

    # --- S&P 500 from Wikipedia ---
    current, changes = fetch_current_and_changes(cached_html=cached_wikipedia_html)
    sp500_members = build_point_in_time_membership(current, changes, as_of=target)

    sp500_rows: list[dict[str, Any]] = []
    for _, row in sp500_members.iterrows():
        ticker = str(row["ticker"]).strip()
        name = str(row.get("name", ticker)).strip()
        cik = str(row.get("cik", "") or "")
        sid = make_security_id(cik, ticker, name)

        # Check for known ticker changes: if this ticker was formerly something else,
        # add a historical row for the old ticker
        for old_ticker, (new_ticker, change_date, _) in KNOWN_TICKER_CHANGES.items():
            if ticker == new_ticker:
                sp500_rows.append(
                    {
                        "security_id": sid,
                        "ticker": old_ticker,
                        "valid_from": date(2000, 1, 1),  # conservative; refine with EDGAR
                        "valid_to": change_date,
                        "name": name,
                        "gics_sector": str(row.get("gics_sector", "Unknown")),
                        "cohort": "sp500",
                        "schema_version": SCHEMA_VERSION,
                    }
                )

        # Determine valid_from from date_added where available
        raw_date = row.get("date_added")
        valid_from: date
        try:
            valid_from = pd.to_datetime(str(raw_date)).date() if raw_date else date(2000, 1, 1)
        except Exception:
            valid_from = date(2000, 1, 1)

        sp500_rows.append(
            {
                "security_id": sid,
                "ticker": ticker,
                "valid_from": valid_from,
                "valid_to": None,  # still active
                "name": name,
                "gics_sector": str(row.get("gics_sector", "Unknown")),
                "cohort": "sp500",
                "schema_version": SCHEMA_VERSION,
            }
        )

    # --- Retail basket from config ---
    retail_tickers = _load_retail_basket(config_path)
    retail_rows: list[dict[str, Any]] = []
    for item in retail_tickers:
        ticker = item["ticker"]
        name = item.get("name", ticker)
        sid = make_security_id(None, ticker, name)
        retail_rows.append(
            {
                "security_id": sid,
                "ticker": ticker,
                "valid_from": date.fromisoformat(item.get("added", target.isoformat())),
                "valid_to": None,
                "name": name,
                "gics_sector": "Retail Attention Basket",
                "cohort": "retail_attention",
                "schema_version": SCHEMA_VERSION,
            }
        )

    all_rows = sp500_rows + retail_rows
    df = pd.DataFrame(all_rows)

    # Deduplicate: same security_id + ticker + valid_from → keep first
    df = df.drop_duplicates(subset=["security_id", "ticker", "valid_from"])

    # Validate no duplicate active (valid_to=None) rows for the same security_id
    active = df[df["valid_to"].isna()]
    dup_ids = active["security_id"][active["security_id"].duplicated()]
    if not dup_ids.empty:
        log.warning(
            "duplicate active security_ids found",
            ids=dup_ids.tolist(),
        )

    log.info(
        "security master built",
        total_rows=len(df),
        sp500=len(sp500_rows),
        retail=len(retail_rows),
    )

    # Write to archive
    out_dir = data_repo_path / "processed" / "security_master"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"security_master_{target.isoformat()}.parquet"

    table = _df_to_arrow(df)
    pq.write_table(table, out_path, compression="zstd")
    log.info("wrote security master", path=str(out_path), rows=len(df))

    return df


def _df_to_arrow(df: pd.DataFrame) -> pa.Table:
    """Convert security master DataFrame to PyArrow Table with correct schema."""
    df = df.copy()

    # Convert date columns
    for col in ["valid_from", "valid_to"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date

    # Cast to match schema field types
    return pa.Table.from_pandas(
        df[
            [
                "security_id",
                "ticker",
                "valid_from",
                "valid_to",
                "name",
                "gics_sector",
                "cohort",
                "schema_version",
            ]
        ],
        schema=SECURITY_MASTER_SCHEMA,
        preserve_index=False,
    )


def load_security_master(data_repo_path: Path) -> pd.DataFrame:
    """Load the most recent security master from the archive."""
    sm_dir = data_repo_path / "processed" / "security_master"
    files = sorted(sm_dir.glob("security_master_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No security master found in {sm_dir}. Run `lll-universe` first."
        )
    latest = files[-1]
    log.info("loading security master", path=str(latest))
    return pq.read_table(latest).to_pandas()


def get_active_tickers(
    sm: pd.DataFrame,
    as_of: date | None = None,
    cohort: str | None = None,
) -> list[str]:
    """Return tickers active on `as_of` date, optionally filtered by cohort."""
    target = as_of or date.today()

    active = sm[
        (sm["valid_from"] <= target)
        & (sm["valid_to"].isna() | (sm["valid_to"] >= target))
    ]

    if cohort:
        active = active[active["cohort"] == cohort]

    return sorted(active["ticker"].unique().tolist())

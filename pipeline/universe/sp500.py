"""
S&P 500 point-in-time membership from Wikipedia.

PRIMARY source: Wikipedia current constituents table.
  https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

HISTORICAL CHANGES: Wikipedia previously maintained a "Selected changes" table
on this page; as of late 2026 it has been removed. We therefore cannot reconstruct
point-in-time membership from Wikipedia alone.

LIMITATION (documented per build spec):
  - This module provides the CURRENT S&P 500 constituents with high confidence.
  - Point-in-time historical membership is best-effort only. Without a reliable
    free source of historical changes, features are computed on the current
    constituents and backfilled. Survivorship bias is documented on the site.
  - Any study conclusion is limited to "returns of current and recent S&P 500
    members", not the full historical index population.
  - We revisit this in Phase 2 (potential: CRSP, Bloomberg data donation, or a
    maintained open-source constituent history dataset).

ENTITY MAPPING NOTES:
  - Tickers with dots in the Wikipedia table are stored with hyphens (e.g.,
    BRK.B → BRK-B) to match Stooq's URL convention.
  - "Block changed SQ → XYZ" is handled via KNOWN_TICKER_CHANGES in
    security_master.py. Verify each entry against public filings before shipping.
"""

from __future__ import annotations

import hashlib
import io
from datetime import date
from typing import Any

import pandas as pd
import requests
import structlog

log = structlog.get_logger()

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "LeadLagLab/1.0 (leadlaglab@gmail.com)"

_SYMBOL_VARIANTS = {"symbol", "ticker", "ticker symbol"}
_SECTOR_VARIANTS = {"gics sector", "sector"}
_NAME_VARIANTS = {"security", "company", "name", "company name"}
_DATE_ADDED_VARIANTS = {"date added", "date first added"}
_CIK_VARIANTS = {"cik", "sec cik"}


def _normalize_col(col: Any) -> str:
    if isinstance(col, tuple):
        return " ".join(str(c) for c in col if str(c) != "nan").lower().strip()
    return str(col).lower().strip()


def _find_col(columns: list[str], variants: set[str]) -> str | None:
    for col in columns:
        if col in variants:
            return col
    return None


def _fetch_html(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def _find_current_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Find the current constituents table by looking for Symbol + Security + GICS Sector."""
    best: pd.DataFrame | None = None
    for t in tables:
        norm_cols = [_normalize_col(c) for c in t.columns]
        has_symbol = _find_col(norm_cols, _SYMBOL_VARIANTS) is not None
        has_sector = _find_col(norm_cols, _SECTOR_VARIANTS) is not None
        has_name = _find_col(norm_cols, _NAME_VARIANTS) is not None
        if has_symbol and has_sector and has_name and (best is None or len(t) > len(best)):
            best = t
    return best


def _find_changes_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """
    Find the historical changes table. Returns None if not present
    (Wikipedia removed it as of late 2026).
    Must have BOTH an "added" column AND a "removed" column to avoid
    matching the current table's "date added" column.
    """
    for t in tables:
        norm_cols = [_normalize_col(c) for c in t.columns]
        has_added = any("added" in c and ("ticker" in c or "security" in c) for c in norm_cols)
        has_removed = any("removed" in c for c in norm_cols)
        has_date = any(c == "date" or c.startswith("date ") for c in norm_cols)
        if has_added and has_removed and has_date:
            return t
    return None


def fetch_current_and_changes(
    cached_html: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (current_members, changes_history).

    current_members columns: ticker, name, gics_sector, gics_sub_industry, date_added, cik
    changes_history columns: date, added_ticker, added_name, removed_ticker, removed_name, reason
    (changes_history may be empty if Wikipedia no longer publishes the table)
    """
    html = cached_html or _fetch_html(WIKIPEDIA_URL)
    tables = pd.read_html(io.StringIO(html), flavor="lxml")

    # --- Current constituents ---
    current_raw = _find_current_table(tables)
    if current_raw is None:
        raise ValueError(
            f"Could not find current S&P 500 constituents table. "
            f"Found {len(tables)} tables with column sets: "
            + str([[_normalize_col(c) for c in t.columns[:3]] for t in tables[:5]])
        )

    current_raw = current_raw.copy()
    current_raw.columns = [_normalize_col(c) for c in current_raw.columns]
    norm_cols = list(current_raw.columns)

    sym_col = _find_col(norm_cols, _SYMBOL_VARIANTS)
    name_col = _find_col(norm_cols, _NAME_VARIANTS)
    sector_col = _find_col(norm_cols, _SECTOR_VARIANTS)
    date_col = _find_col(norm_cols, _DATE_ADDED_VARIANTS)
    cik_col = _find_col(norm_cols, _CIK_VARIANTS)
    sub_col = _find_col(norm_cols, {"gics sub-industry", "sub-industry", "gics sub industry"})

    current = pd.DataFrame(
        {
            "ticker": (
                current_raw[sym_col].astype(str).str.strip().str.replace(r"\.", "-", regex=True)
            ),
            "name": current_raw[name_col].astype(str).str.strip(),
            "gics_sector": current_raw[sector_col].astype(str).str.strip(),
            "gics_sub_industry": (current_raw[sub_col].astype(str).str.strip() if sub_col else ""),
            "date_added": (current_raw[date_col].astype(str) if date_col else None),
            "cik": (current_raw[cik_col].astype(str).str.zfill(10) if cik_col else None),
        }
    )
    current = current[current["ticker"].str.len() > 0]

    # --- Changes history (best-effort; may be empty) ---
    changes_raw = _find_changes_table(tables)
    if changes_raw is None:
        log.warning(
            "S&P 500 historical changes table not found on Wikipedia. "
            "Point-in-time membership reconstruction is limited to current members. "
            "This is a known limitation; see docs/DATA_SOURCES.md."
        )
        empty_changes = pd.DataFrame(
            columns=[
                "date",
                "added_ticker",
                "added_name",
                "removed_ticker",
                "removed_name",
                "reason",
            ]
        )
        log.info(
            "wikipedia fetch complete (current only)",
            current_members=len(current),
            change_events=0,
        )
        return current, empty_changes

    changes_raw = changes_raw.copy()
    changes_raw.columns = [_normalize_col(c) for c in changes_raw.columns]
    cols = list(changes_raw.columns)

    def _pick(*candidates: str) -> str | None:
        for c in candidates:
            if c in cols:
                return c
        return None

    date_c = _pick("date", "date date")
    add_ticker_c = _pick("added ticker", "ticker", "added symbol")
    add_name_c = _pick("added security", "added name", "security")
    rem_ticker_c = _pick("removed ticker", "removed symbol")
    rem_name_c = _pick("removed security", "removed name")
    reason_c = _pick("reason", "reason reason", "notes")

    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(
                changes_raw[date_c] if date_c else pd.Series([], dtype=str),
                errors="coerce",
            ).dt.date,
            "added_ticker": (
                changes_raw[add_ticker_c].astype(str).str.strip() if add_ticker_c else None
            ),
            "added_name": (changes_raw[add_name_c].astype(str).str.strip() if add_name_c else None),
            "removed_ticker": (
                changes_raw[rem_ticker_c].astype(str).str.strip() if rem_ticker_c else None
            ),
            "removed_name": (
                changes_raw[rem_name_c].astype(str).str.strip() if rem_name_c else None
            ),
            "reason": (changes_raw[reason_c].astype(str) if reason_c else None),
        }
    ).dropna(subset=["date"])

    log.info(
        "wikipedia fetch complete",
        current_members=len(current),
        change_events=len(changes),
    )
    return current, changes


def build_point_in_time_membership(
    current: pd.DataFrame,
    changes: pd.DataFrame,
    as_of: date | None = None,
) -> pd.DataFrame:
    """
    Reconstruct S&P 500 membership as of `as_of` (defaults to today).

    If changes is empty (no historical data), returns current members directly.
    """
    target = as_of or date.today()

    members: dict[str, dict[str, Any]] = {
        row["ticker"]: {
            "ticker": row["ticker"],
            "name": row["name"],
            "gics_sector": row["gics_sector"],
            "date_added": row.get("date_added"),
            "cik": row.get("cik"),
            "membership_source": "wikipedia_current",
        }
        for _, row in current.iterrows()
    }

    if changes.empty:
        return pd.DataFrame(list(members.values()))

    future_changes = changes[changes["date"] > target].sort_values("date", ascending=False)

    for _, chg in future_changes.iterrows():
        added = str(chg.get("added_ticker", "") or "").strip()
        if added and added in members:
            del members[added]

        removed = str(chg.get("removed_ticker", "") or "").strip()
        removed_name = str(chg.get("removed_name", "") or "").strip()
        if removed and removed not in members:
            members[removed] = {
                "ticker": removed,
                "name": removed_name or removed,
                "gics_sector": "Unknown",
                "date_added": None,
                "cik": None,
                "membership_source": "wikipedia_historical",
            }

    return pd.DataFrame(list(members.values()))


def make_security_id(cik: str | None, ticker: str, name: str) -> str:
    """
    Stable internal security_id.
    Uses zero-padded CIK if available (e.g., "0000320193"),
    otherwise a deterministic hash of (name.lower + ticker).
    """
    if cik and cik not in {"None", "nan", ""} and cik.replace("0", "").isdigit():
        return cik.zfill(10)
    seed = f"{name.lower().strip()}|{ticker.upper().strip()}"
    return "h" + hashlib.sha1(seed.encode()).hexdigest()[:9]

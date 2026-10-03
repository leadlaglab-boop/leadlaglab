"""Tests for SecurityMaster building logic."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pipeline.universe.sp500 import (
    build_point_in_time_membership,
    fetch_current_and_changes,
    make_security_id,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic Wikipedia HTML
# ---------------------------------------------------------------------------

FAKE_CURRENT_HTML = textwrap.dedent("""
<html><body>
<table>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th><th>CIK</th></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>1982-11-30</td><td>0000320193</td></tr>
<tr><td>MSFT</td><td>Microsoft Corp.</td><td>Information Technology</td><td>1994-06-01</td><td>0000789019</td></tr>
<tr><td>XYZ</td><td>Block Inc.</td><td>Financials</td><td>2021-09-15</td><td>0001512673</td></tr>
</table>
<table>
<tr><th>Date</th><th>Added Ticker</th><th>Added Security</th><th>Removed Ticker</th><th>Removed Security</th><th>Reason</th></tr>
<tr><td>2025-01-10</td><td>XYZ</td><td>Block Inc.</td><td>SQ</td><td>Block Inc. (old)</td><td>Ticker change</td></tr>
<tr><td>2023-03-01</td><td>NVDA</td><td>Nvidia Corp.</td><td>DISCA</td><td>Discovery Inc.</td><td>Acquisition</td></tr>
</table>
</body></html>
""")


@pytest.fixture()
def wiki_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    with patch("pipeline.universe.sp500._fetch_html", return_value=FAKE_CURRENT_HTML):
        return fetch_current_and_changes()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fetch_current_columns(wiki_tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    current, _ = wiki_tables
    assert "ticker" in current.columns
    assert "name" in current.columns
    assert "gics_sector" in current.columns


def test_fetch_current_row_count(wiki_tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    current, _ = wiki_tables
    assert len(current) == 3


def test_fetch_changes_has_date(wiki_tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, changes = wiki_tables
    assert "date" in changes.columns
    assert len(changes) == 2


def test_point_in_time_today(wiki_tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    current, changes = wiki_tables
    members = build_point_in_time_membership(current, changes, as_of=date(2026, 1, 1))
    tickers = set(members["ticker"])
    # XYZ was added 2025-01-10, so it should be present as of 2026
    assert "XYZ" in tickers
    # SQ was removed 2025-01-10; should NOT be present as of 2026
    assert "SQ" not in tickers


def test_point_in_time_before_change(wiki_tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    current, changes = wiki_tables
    # As of 2024-01-01, XYZ hadn't been added yet and SQ should be present
    members = build_point_in_time_membership(current, changes, as_of=date(2024, 1, 1))
    tickers = set(members["ticker"])
    assert "SQ" in tickers
    assert "XYZ" not in tickers


def test_no_duplicate_active_security_ids(wiki_tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    current, changes = wiki_tables
    members = build_point_in_time_membership(current, changes)
    dups = members["ticker"][members["ticker"].duplicated()]
    assert dups.empty, f"Duplicate tickers: {dups.tolist()}"


# ---------------------------------------------------------------------------
# make_security_id tests
# ---------------------------------------------------------------------------


def test_security_id_uses_cik_when_available() -> None:
    sid = make_security_id("320193", "AAPL", "Apple Inc.")
    assert sid == "0000320193"


def test_security_id_hash_fallback() -> None:
    sid = make_security_id(None, "GME", "GameStop Corp")
    assert sid.startswith("h")
    assert len(sid) == 10


def test_security_id_stable() -> None:
    """Same inputs always produce same ID."""
    a = make_security_id(None, "GME", "GameStop Corp")
    b = make_security_id(None, "GME", "GameStop Corp")
    assert a == b


def test_security_id_differs_for_different_companies() -> None:
    gme = make_security_id(None, "GME", "GameStop Corp")
    amc = make_security_id(None, "AMC", "AMC Entertainment")
    assert gme != amc


def test_cik_none_variants_treated_as_missing() -> None:
    for bad in ("None", "nan", ""):
        sid = make_security_id(bad, "TEST", "Test Corp")
        assert sid.startswith("h"), f"Expected hash fallback for CIK={bad!r}"

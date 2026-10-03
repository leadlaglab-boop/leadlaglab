"""Tests for price ingestion logic."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pytest

from pipeline.sources.base import DateRange
from pipeline.sources.stooq import StooqPriceSource

FAKE_STOOQ_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,185.00,188.50,184.50,187.15,55000000
2024-01-03,187.00,189.00,186.00,185.92,48000000
2024-01-04,186.00,187.00,183.00,182.50,52000000
"""

FAKE_STOOQ_EMPTY = "No data"


@pytest.fixture()
def stooq() -> StooqPriceSource:
    return StooqPriceSource()


# ---------------------------------------------------------------------------
# Stooq parsing
# ---------------------------------------------------------------------------


def test_stooq_parses_csv(stooq: StooqPriceSource) -> None:
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = FAKE_STOOQ_CSV
        mock_resp.content = FAKE_STOOQ_CSV.encode()
        mock_get.return_value = mock_resp

        records = stooq.fetch_ticker(
            "AAPL",
            "sec_001",
            DateRange(date(2024, 1, 2), date(2024, 1, 4)),
            data_type="backfilled",
        )

    assert len(records) == 3
    assert records[0]["close"] == pytest.approx(187.15)
    assert records[0]["security_id"] == "sec_001"
    assert records[0]["source"] == "stooq"


def test_stooq_adj_close_is_none(stooq: StooqPriceSource) -> None:
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = FAKE_STOOQ_CSV
        mock_resp.content = FAKE_STOOQ_CSV.encode()
        mock_get.return_value = mock_resp

        records = stooq.fetch_ticker(
            "AAPL", "sec_001", DateRange(date(2024, 1, 2), date(2024, 1, 4))
        )

    for r in records:
        assert r["adj_close"] is None, "Stooq should not produce adj_close"


def test_stooq_empty_response_returns_empty(stooq: StooqPriceSource) -> None:
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = FAKE_STOOQ_EMPTY
        mock_resp.content = FAKE_STOOQ_EMPTY.encode()
        mock_get.return_value = mock_resp

        records = stooq.fetch_ticker(
            "FAKE", "sec_999", DateRange(date(2024, 1, 2), date(2024, 1, 4))
        )

    assert records == []


def test_stooq_404_returns_empty(stooq: StooqPriceSource) -> None:
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = ""
        mock_resp.content = b""
        mock_get.return_value = mock_resp

        records = stooq.fetch_ticker(
            "FAKE", "sec_999", DateRange(date(2024, 1, 2), date(2024, 1, 4))
        )

    assert records == []


def test_stooq_filters_zero_close(stooq: StooqPriceSource) -> None:
    csv_with_zero = (
        "Date,Open,High,Low,Close,Volume\n2024-01-02,0,0,0,0,0\n2024-01-03,185,188,184,187,5000\n"
    )
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_with_zero
        mock_resp.content = csv_with_zero.encode()
        mock_get.return_value = mock_resp

        records = stooq.fetch_ticker(
            "AAPL", "sec_001", DateRange(date(2024, 1, 2), date(2024, 1, 3))
        )

    assert len(records) == 1
    assert records[0]["close"] == pytest.approx(187.0)


# ---------------------------------------------------------------------------
# Look-ahead invariant for price records
# ---------------------------------------------------------------------------


def test_backfilled_observed_at_is_market_close(stooq: StooqPriceSource) -> None:
    """For backfilled records, observed_at must equal market_close(date), not fetch time."""
    from pipeline.utils.calendar import market_close_utc

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = FAKE_STOOQ_CSV
        mock_resp.content = FAKE_STOOQ_CSV.encode()
        mock_get.return_value = mock_resp

        records = stooq.fetch_ticker(
            "AAPL",
            "sec_001",
            DateRange(date(2024, 1, 2), date(2024, 1, 4)),
            data_type="backfilled",
        )

    for r in records:
        expected = market_close_utc(r["date"].isoformat())
        assert (
            r["observed_at"] == expected
        ), f"observed_at={r['observed_at']} != market_close({r['date']})={expected}"


def test_live_observed_at_is_recent(stooq: StooqPriceSource) -> None:
    """For live records, observed_at should be close to now (not a historical date)."""
    before = datetime.now(tz=UTC)

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = FAKE_STOOQ_CSV
        mock_resp.content = FAKE_STOOQ_CSV.encode()
        mock_get.return_value = mock_resp

        records = stooq.fetch_ticker(
            "AAPL",
            "sec_001",
            DateRange(date(2024, 1, 2), date(2024, 1, 4)),
            data_type="live",
        )

    after = datetime.now(tz=UTC)
    for r in records:
        assert (
            before <= r["observed_at"] <= after
        ), f"Live observed_at={r['observed_at']} is not within expected range"


def test_date_range_validation() -> None:
    with pytest.raises(ValueError, match="start"):
        DateRange(start=date(2024, 1, 10), end=date(2024, 1, 5))

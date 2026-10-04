"""
Price ingestion orchestrator.

Strategy:
  1. Tiingo (primary; needs TIINGO_API_KEY): OHLCV + adj_close
  2. Stooq (fallback when Tiingo has no key or returns nothing): raw OHLCV,
     adj_close=None. Stooq currently serves a Cloudflare challenge to automated
     clients, so in practice this fallback usually returns nothing too.
  3. Store records in the archive

Records are written as Parquet, partitioned by source/year/month/day.
Idempotent: re-running for the same date range is safe (will overwrite the partition).
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from pipeline.sources.base import DateRange
from pipeline.sources.stooq import StooqPriceSource
from pipeline.sources.tiingo import TiingoPriceSource
from pipeline.validate.schemas import PRICES_DAILY_SCHEMA

log = structlog.get_logger()


class PriceIngestor:
    def __init__(
        self,
        data_repo_path: Path,
        tiingo_api_key: str | None = None,
    ) -> None:
        self.data_repo_path = data_repo_path
        self.stooq = StooqPriceSource()
        self.tiingo = TiingoPriceSource(api_key=tiingo_api_key)

    def fetch_and_store(
        self,
        ticker: str,
        security_id: str,
        date_range: DateRange,
        data_type: str = "backfilled",
    ) -> int:
        """
        Fetch prices for one ticker, merge sources, and write to archive.
        Returns number of records written.
        """
        records: list[dict[str, Any]] = []
        if self.tiingo.available:
            records = self.tiingo.fetch_ticker(ticker, security_id, date_range, data_type)
        if not records:
            records = self.stooq.fetch_ticker(ticker, security_id, date_range, data_type)

        if not records:
            log.warning("no price data from any source", ticker=ticker)
            return 0

        df = pd.DataFrame(records)

        # Write partitioned by year/month/day of the price date
        # (not observed_at; price date is the natural partition key)
        rows_written = 0
        for price_date, group in df.groupby("date"):
            assert isinstance(price_date, date)
            out_dir = (
                self.data_repo_path
                / "raw"
                / "prices"
                / f"year={price_date.year}"
                / f"month={price_date.month:02d}"
                / f"day={price_date.day:02d}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{ticker.lower()}.parquet"

            table = _df_to_arrow(group)
            pq.write_table(table, out_path, compression="zstd")
            rows_written += len(group)

        return rows_written

    def batch_fetch(
        self,
        tickers: list[tuple[str, str]],  # [(ticker, security_id), ...]
        date_range: DateRange,
        data_type: str = "backfilled",
    ) -> dict[str, int]:
        """
        Fetch prices for multiple tickers. Returns {ticker: rows_written}.
        Logs progress; continues on per-ticker errors.
        """
        results: dict[str, int] = {}
        total = len(tickers)

        for i, (ticker, security_id) in enumerate(tickers, 1):
            log.info(
                "price ingest progress",
                ticker=ticker,
                progress=f"{i}/{total}",
            )
            try:
                n = self.fetch_and_store(ticker, security_id, date_range, data_type)
                results[ticker] = n
            except Exception as e:
                log.error("price fetch failed", ticker=ticker, error=str(e))
                results[ticker] = -1

        succeeded = sum(1 for v in results.values() if v >= 0)
        total_rows = sum(v for v in results.values() if v > 0)
        log.info(
            "batch fetch complete",
            succeeded=succeeded,
            failed=total - succeeded,
            total_rows=total_rows,
        )
        return results


def _df_to_arrow(df: pd.DataFrame) -> pa.Table:
    df = df.copy()

    # Ensure observed_at is UTC-aware datetime
    if "observed_at" in df.columns:
        df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)

    return pa.Table.from_pandas(
        df[
            [
                "security_id",
                "date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "observed_at",
                "source",
                "data_type",
                "schema_version",
            ]
        ],
        schema=PRICES_DAILY_SCHEMA,
        preserve_index=False,
    )


def load_prices(
    data_repo_path: Path,
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """
    Load price data from the archive into a DataFrame.
    Optionally filter by ticker list and date range.
    """
    prices_dir = data_repo_path / "raw" / "prices"
    if not prices_dir.exists():
        return pd.DataFrame()

    parts: list[pa.Table] = []
    for parquet_file in sorted(prices_dir.rglob("*.parquet")):
        # Extract date from path: .../year=2024/month=01/day=15/aapl.parquet
        match = re.search(r"year=(\d+)/month=(\d+)/day=(\d+)", str(parquet_file))
        if match:
            file_date = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            if start and file_date < start:
                continue
            if end and file_date > end:
                continue

        if tickers:
            file_ticker = parquet_file.stem.upper()
            if file_ticker not in [t.upper() for t in tickers]:
                continue

        # The ticker lives only in the filename; attach it so callers can pivot by ticker
        table = pq.read_table(parquet_file)
        parts.append(
            table.append_column("ticker", pa.array([parquet_file.stem.upper()] * table.num_rows))
        )

    if not parts:
        return pd.DataFrame()

    return cast("pd.DataFrame", pa.concat_tables(parts).to_pandas())

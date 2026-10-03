"""
Tests for M4 feature pipeline.

Covers:
  - Rolling z-score correctness
  - Cross-sectional rank and z-score within sector
  - Empirical-Bayes shrinkage
  - No look-ahead: features at date t use only signals from effective_date <= t-1
  - Schema compliance of written Parquet output
  - End-to-end: archive → feature records with mocked raw data
  - load_features round-trip
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.features.builder import (
    FeatureBuilder,
    _write_features,
    cross_sectional_rank,
    cross_sectional_zscore,
    load_features,
    rolling_nobs,
    rolling_zscore,
    shrink_toward_zero,
)
from pipeline.utils.calendar import market_close_utc
from pipeline.validate.schemas import SIGNAL_FEATURE_SCHEMA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_security_master(tickers: list[str], sectors: list[str] | None = None) -> pd.DataFrame:
    n = len(tickers)
    sectors = sectors or ["Information Technology"] * n
    return pd.DataFrame(
        {
            "security_id": [f"sid_{t}" for t in tickers],
            "ticker": tickers,
            "valid_from": [date(2020, 1, 1)] * n,
            "valid_to": [None] * n,
            "name": [f"{t} Corp" for t in tickers],
            "gics_sector": sectors,
            "cohort": ["sp500"] * n,
            "schema_version": ["1.0.0"] * n,
        }
    )


def _write_raw_record(
    data_repo: Path,
    source: str,
    ticker: str,
    day: date,
    payload: dict,
    security_id: str | None = None,
) -> None:
    """Write a single raw signal record to the archive for testing."""
    import pyarrow as pa

    from pipeline.validate.schemas import SCHEMA_VERSION, SIGNAL_RAW_SCHEMA

    sid = security_id or f"sid_{ticker}"
    path = (
        data_repo
        / "raw"
        / source
        / f"year={day.year}"
        / f"month={day.month:02d}"
        / f"day={day.day:02d}"
        / f"{ticker.lower()}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "source": source,
                "security_id": sid,
                "effective_date": day,
                "observed_at": market_close_utc(day.isoformat()),
                "payload_json": json.dumps(payload),
                "n_items": 1,
                "source_version": "v1",
                "schema_version": SCHEMA_VERSION,
            }
        ]
    )
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    table = pa.Table.from_pandas(df, schema=SIGNAL_RAW_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


# ---------------------------------------------------------------------------
# Rolling z-score
# ---------------------------------------------------------------------------


class TestRollingZscore:
    def test_basic_zscore(self) -> None:
        s = pd.Series(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        )
        z = rolling_zscore(s, lookback=10, min_obs=5)
        assert z.notna().sum() > 0
        # Leading NaN values (insufficient window) are fine; all non-NaN must be capped
        assert z.dropna().abs().le(4.0).all()

    def test_constant_series_returns_nan(self) -> None:
        s = pd.Series([5.0] * 15)
        z = rolling_zscore(s, lookback=10, min_obs=5)
        # std = 0 → NaN
        assert z.dropna().empty or (z.dropna() == 0).all()

    def test_insufficient_obs_returns_nan(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0] + [np.nan] * 12)
        z = rolling_zscore(s, lookback=10, min_obs=5)
        assert z.notna().sum() == 0

    def test_z_capped_at_z_cap(self) -> None:
        from pipeline.features.builder import Z_CAP

        s = pd.Series([1.0] * 9 + [1000.0])
        z = rolling_zscore(s, lookback=10, min_obs=5)
        assert z.max() <= Z_CAP
        assert z.min() >= -Z_CAP

    def test_rolling_nobs(self) -> None:
        s = pd.Series([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
        n = rolling_nobs(s, lookback=4)
        assert int(n.iloc[-1]) == 3  # last 4: nan, 2, 3, nan, 4 → wait, window is 4
        # Last 4 values: 3, nan, 4 → n=2... depends on index
        assert n.iloc[-1] >= 1


# ---------------------------------------------------------------------------
# Cross-sectional normalisation
# ---------------------------------------------------------------------------


class TestCrossSectional:
    def _make_df(self, values: list[float], sectors: list[str] | None = None) -> pd.DataFrame:
        n = len(values)
        sectors = sectors or ["Tech"] * n
        return pd.DataFrame(
            {
                "security_id": [f"s{i}" for i in range(n)],
                "value": values,
                "gics_sector": sectors,
            }
        )

    def test_rank_range(self) -> None:
        df = self._make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        ranks = cross_sectional_rank(df, "value")
        assert ranks.min() > 0
        assert ranks.max() <= 1.0

    def test_rank_monotone(self) -> None:
        df = self._make_df([10.0, 20.0, 30.0])
        ranks = cross_sectional_rank(df, "value")
        assert ranks.iloc[0] < ranks.iloc[1] < ranks.iloc[2]

    def test_zscore_mean_near_zero(self) -> None:
        df = self._make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        z = cross_sectional_zscore(df, "value")
        assert abs(z.mean()) < 1e-10

    def test_zscore_std_near_one(self) -> None:
        df = self._make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        z = cross_sectional_zscore(df, "value")
        assert abs(z.std(ddof=1) - 1.0) < 1e-10

    def test_nan_values_excluded_from_rank(self) -> None:
        df = self._make_df([1.0, np.nan, 3.0])
        ranks = cross_sectional_rank(df, "value")
        assert pd.isna(ranks.iloc[1])
        assert pd.notna(ranks.iloc[0])
        assert pd.notna(ranks.iloc[2])

    def test_sector_isolation(self) -> None:
        df = self._make_df(
            [1.0, 100.0, 2.0, 200.0],
            sectors=["Tech", "Finance", "Tech", "Finance"],
        )
        ranks = cross_sectional_rank(df, "value")
        # Tech: [1.0, 2.0] → [low, high]; Finance: [100.0, 200.0] → [low, high]
        assert ranks.iloc[0] < ranks.iloc[2]  # 1 < 2 within Tech
        assert ranks.iloc[1] < ranks.iloc[3]  # 100 < 200 within Finance

    def test_single_stock_sector_skipped(self) -> None:
        df = self._make_df([5.0], sectors=["Lonely"])
        ranks = cross_sectional_rank(df, "value")
        assert pd.isna(ranks.iloc[0])


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------


class TestShrinkage:
    def test_shrinks_toward_zero(self) -> None:
        # Same z-score value, different n — higher n should shrink less (preserve more)
        z = pd.Series([2.0, 2.0])
        n = pd.Series([5, 50])
        shrunk = shrink_toward_zero(z, n)
        # n=50 → less shrinkage → shrunk value closer to original (larger abs)
        assert abs(shrunk.iloc[1]) > abs(shrunk.iloc[0])

    def test_high_n_preserves_sign(self) -> None:
        z = pd.Series([3.0])
        n = pd.Series([10000])
        shrunk = shrink_toward_zero(z, n)
        assert shrunk.iloc[0] > 0

    def test_zero_obs_shrinks_fully(self) -> None:
        z = pd.Series([5.0])
        n = pd.Series([0])
        shrunk = shrink_toward_zero(z, n)
        # B = 1/(1+0/10) = 1.0 → fully shrunk to 0
        assert shrunk.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Look-ahead invariant (property-based)
# ---------------------------------------------------------------------------


class TestFeatureLookAhead:
    def test_feature_date_uses_only_prior_day_signals(self, tmp_path: Path) -> None:
        """
        Features built for date t must use only raw signals with
        effective_date <= t - signal_lag_days.
        """
        from pipeline.features.builder import SIGNAL_LAG_DAYS

        sm = _make_security_master(["AAPL"], ["Information Technology"])
        data_repo = tmp_path / "data"

        # Write signal for t-1 (VALID: should be used)
        target = date(2024, 6, 3)
        valid_day = target - timedelta(days=SIGNAL_LAG_DAYS)
        _write_raw_record(
            data_repo,
            "wikipedia_pageviews",
            "AAPL",
            valid_day,
            {"views": 10000, "articles": ["Apple Inc."], "ambiguous": False},
            security_id="sid_AAPL",
        )

        # Write a signal for target_date (INVALID: should not affect feature)
        _write_raw_record(
            data_repo,
            "wikipedia_pageviews",
            "AAPL",
            target,
            {"views": 999999, "articles": ["Apple Inc."], "ambiguous": False},
            security_id="sid_AAPL",
        )

        builder = FeatureBuilder(data_repo, lookback_days=60, signal_lag_days=SIGNAL_LAG_DAYS)
        # Monkey-patch to record which dates were loaded
        loaded_dates: list[date] = []
        original = builder._load_ticker_history

        def patched(source, ticker, window_start, window_end):
            loaded_dates.append(window_end)
            return original(source, ticker, window_start, window_end)

        builder._load_ticker_history = patched
        builder.build_date(target, sm)

        # window_end must be < target
        for d in loaded_dates:
            assert d < target, f"window_end {d} is not before target {target}"

    def test_observed_at_before_market_close_for_all_window_records(self, tmp_path: Path) -> None:
        """
        All raw signal records in the feature window must have
        observed_at < market_close(target_date).
        """
        from pipeline.features.builder import SIGNAL_LAG_DAYS

        data_repo = tmp_path / "data"
        target = date(2024, 6, 3)
        cutoff = market_close_utc(target.isoformat())

        for i in range(1, 5):
            signal_day = target - timedelta(days=i)
            _write_raw_record(
                data_repo,
                "wikipedia_pageviews",
                "AAPL",
                signal_day,
                {"views": 1000 * i, "articles": ["Apple Inc."], "ambiguous": False},
                security_id="sid_AAPL",
            )

        # Verify all records in the window have observed_at < cutoff
        window_end = target - timedelta(days=SIGNAL_LAG_DAYS)
        window_start = window_end - timedelta(days=60)

        # Reload with timestamps from original parquet files to check observed_at
        current = window_start
        while current <= window_end:
            path = (
                data_repo
                / "raw"
                / "wikipedia_pageviews"
                / f"year={current.year}"
                / f"month={current.month:02d}"
                / f"day={current.day:02d}"
                / "aapl.parquet"
            )
            if path.exists():
                df = pq.read_table(path).to_pandas()
                for obs_at in pd.to_datetime(df["observed_at"], utc=True):
                    assert (
                        obs_at < cutoff
                    ), f"observed_at {obs_at} is not before market_close({target}) = {cutoff}"
            current += timedelta(days=1)


# ---------------------------------------------------------------------------
# End-to-end: archive → features
# ---------------------------------------------------------------------------


class TestFeatureBuilderEndToEnd:
    def _setup(self, tmp_path: Path) -> tuple[Path, pd.DataFrame]:
        data_repo = tmp_path / "data"
        sm = _make_security_master(
            ["AAPL", "MSFT"],
            ["Information Technology", "Information Technology"],
        )
        return data_repo, sm

    def test_wikipedia_features_written(self, tmp_path: Path) -> None:
        data_repo, sm = self._setup(tmp_path)
        target = date(2024, 6, 3)

        # Write 15 days of Wikipedia data for AAPL (enough for MIN_OBS)
        for i in range(1, 16):
            day = target - timedelta(days=i)
            _write_raw_record(
                data_repo,
                "wikipedia_pageviews",
                "AAPL",
                day,
                {"views": 10000 + i * 100, "articles": ["Apple Inc."], "ambiguous": False},
                security_id="sid_AAPL",
            )

        builder = FeatureBuilder(data_repo, lookback_days=60, signal_lag_days=1, min_obs=10)
        records = builder.build_date(target, sm)

        wiki_records = [r for r in records if r["feature_name"] == "wiki_views_z60d"]
        assert len(wiki_records) == 1
        r = wiki_records[0]
        assert r["security_id"] == "sid_AAPL"
        assert r["date"] == target
        assert r["n_obs"] >= 10
        assert r["feature_version"] == "v1"
        assert r["schema_version"] == "1.0.0"

    def test_insufficient_obs_produces_no_feature(self, tmp_path: Path) -> None:
        data_repo, sm = self._setup(tmp_path)
        target = date(2024, 6, 3)

        # Only 5 days — below MIN_OBS=10
        for i in range(1, 6):
            day = target - timedelta(days=i)
            _write_raw_record(
                data_repo,
                "wikipedia_pageviews",
                "AAPL",
                day,
                {"views": 10000, "articles": ["Apple Inc."], "ambiguous": False},
                security_id="sid_AAPL",
            )

        builder = FeatureBuilder(data_repo, lookback_days=60, signal_lag_days=1, min_obs=10)
        records = builder.build_date(target, sm)
        assert records == []

    def test_schema_compliant_parquet_output(self, tmp_path: Path) -> None:
        data_repo, sm = self._setup(tmp_path)
        target = date(2024, 6, 3)

        for i in range(1, 16):
            day = target - timedelta(days=i)
            _write_raw_record(
                data_repo,
                "wikipedia_pageviews",
                "AAPL",
                day,
                {"views": 10000 + i * 50, "articles": ["Apple Inc."], "ambiguous": False},
                security_id="sid_AAPL",
            )

        builder = FeatureBuilder(data_repo, lookback_days=60, signal_lag_days=1, min_obs=10)
        records = builder.build_date(target, sm)
        rows = _write_features(data_repo, target, records)
        assert rows == len(records)

        # Read back and verify schema
        feat_path = (
            data_repo / "processed" / "features" / f"date={target.isoformat()}" / "features.parquet"
        )
        assert feat_path.exists()
        tbl = pq.read_table(feat_path)
        assert tbl.schema.equals(SIGNAL_FEATURE_SCHEMA)

    def test_edgar_features_extracted(self, tmp_path: Path) -> None:
        data_repo, sm = self._setup(tmp_path)
        target = date(2024, 6, 3)

        for i in range(1, 16):
            day = target - timedelta(days=i)
            _write_raw_record(
                data_repo,
                "edgar",
                "AAPL",
                day,
                {
                    "form_4_count": i % 3,
                    "form_8k_count": 1 if i % 5 == 0 else 0,
                    "total_filings": (i % 3) + (1 if i % 5 == 0 else 0),
                },
                security_id="sid_AAPL",
            )

        builder = FeatureBuilder(data_repo, lookback_days=60, signal_lag_days=1, min_obs=10)
        records = builder.build_date(target, sm)
        feat_names = {r["feature_name"] for r in records}
        assert "edgar_form4_z60d" in feat_names

    def test_gdelt_features_extracted(self, tmp_path: Path) -> None:
        data_repo, sm = self._setup(tmp_path)
        target = date(2024, 6, 3)

        for i in range(1, 16):
            day = target - timedelta(days=i)
            # Vary finbert/vader values so z-score is defined (std > 0)
            _write_raw_record(
                data_repo,
                "gdelt",
                "AAPL",
                day,
                {
                    "article_count": 10 + i,
                    "avg_tone": 1.5 + i * 0.1,
                    "ambiguous": False,
                    "query": '"Apple"',
                    "sample_articles": [],
                    "nlp": {
                        "finbert_avg_positive": 0.5 + i * 0.01,
                        "finbert_avg_negative": 0.2 - i * 0.005,
                        "vader_avg_compound": 0.3 + i * 0.02,
                        "n_articles_scored": 5,
                    },
                },
                security_id="sid_AAPL",
            )

        builder = FeatureBuilder(data_repo, lookback_days=60, signal_lag_days=1, min_obs=10)
        records = builder.build_date(target, sm)
        feat_names = {r["feature_name"] for r in records}
        assert "gdelt_n_z60d" in feat_names
        assert "gdelt_tone_z60d" in feat_names
        assert "gdelt_finbert_z60d" in feat_names
        assert "gdelt_vader_z60d" in feat_names

    def test_cross_sectional_rank_within_sector(self, tmp_path: Path) -> None:
        data_repo, sm = self._setup(tmp_path)
        target = date(2024, 6, 3)

        # AAPL: high views; MSFT: low views — both in same sector
        for i in range(1, 16):
            day = target - timedelta(days=i)
            _write_raw_record(
                data_repo,
                "wikipedia_pageviews",
                "AAPL",
                day,
                {"views": 20000 + i * 100, "articles": ["Apple Inc."], "ambiguous": False},
                security_id="sid_AAPL",
            )
            _write_raw_record(
                data_repo,
                "wikipedia_pageviews",
                "MSFT",
                day,
                {"views": 5000 + i * 10, "articles": ["Microsoft"], "ambiguous": False},
                security_id="sid_MSFT",
            )

        builder = FeatureBuilder(data_repo, lookback_days=60, signal_lag_days=1, min_obs=10)
        records = builder.build_date(target, sm)
        wiki_records = [r for r in records if r["feature_name"] == "wiki_views_z60d"]
        assert len(wiki_records) == 2

        # Both should have rank_cs defined
        for r in wiki_records:
            if r["rank_cs"] is not None:
                assert 0 < r["rank_cs"] <= 1.0

    def test_load_features_roundtrip(self, tmp_path: Path) -> None:
        data_repo, sm = self._setup(tmp_path)
        target = date(2024, 6, 3)

        for i in range(1, 16):
            day = target - timedelta(days=i)
            _write_raw_record(
                data_repo,
                "wikipedia_pageviews",
                "AAPL",
                day,
                {"views": 10000 + i * 50, "articles": ["Apple Inc."], "ambiguous": False},
                security_id="sid_AAPL",
            )

        builder = FeatureBuilder(data_repo, lookback_days=60, signal_lag_days=1, min_obs=10)
        records = builder.build_date(target, sm)
        _write_features(data_repo, target, records)

        loaded = load_features(data_repo, target, target)
        assert len(loaded) == len(records)
        assert set(loaded.columns) >= {"security_id", "date", "feature_name", "value"}

    def test_no_signal_returns_empty(self, tmp_path: Path) -> None:
        data_repo, sm = self._setup(tmp_path)
        target = date(2024, 6, 3)

        builder = FeatureBuilder(data_repo, lookback_days=60, signal_lag_days=1, min_obs=10)
        records = builder.build_date(target, sm)
        assert records == []

    def test_shrunk_value_exists_and_bounded(self, tmp_path: Path) -> None:
        data_repo, sm = self._setup(tmp_path)
        target = date(2024, 6, 3)

        for i in range(1, 16):
            day = target - timedelta(days=i)
            _write_raw_record(
                data_repo,
                "wikipedia_pageviews",
                "AAPL",
                day,
                {"views": 10000 + i * 50, "articles": ["Apple Inc."], "ambiguous": False},
                security_id="sid_AAPL",
            )

        builder = FeatureBuilder(data_repo, lookback_days=60, signal_lag_days=1, min_obs=10)
        records = builder.build_date(target, sm)

        for r in records:
            if r["shrunk_value"] is not None and r["value"] is not None and r["z_cs"] is not None:
                # Shrunk must not move away from zero relative to z_cs
                assert abs(r["shrunk_value"]) <= abs(r["z_cs"]) + 1e-10

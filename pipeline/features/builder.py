"""
Feature pipeline: raw signals → normalised, ranked, shrunk features.

Pipeline per (security_id, date, feature_name):
  1. Load trailing LOOKBACK_DAYS of raw signal values
  2. Log-transform where appropriate (views, counts)
  3. Rolling z-score (mean/std from trailing window)
  4. Cross-sectional rank within (date × GICS sector)
  5. Cross-sectional z-score within (date × GICS sector)
  6. Empirical-Bayes shrinkage toward cross-sectional mean (= 0)

Look-ahead invariant:
  Features for date t use only signals with effective_date ≤ t - SIGNAL_LAG_DAYS.
  Since observed_at = market_close(effective_date) for backfilled signals, this
  ensures observed_at < market_close(t). Enforced and tested in test_features.py.

Output written to:
  {data_repo}/processed/features/date={YYYY-MM-DD}/features.parquet
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from pipeline.universe.security_master import load_security_master
from pipeline.validate.schemas import SCHEMA_VERSION, SIGNAL_FEATURE_SCHEMA

log = structlog.get_logger()

LOOKBACK_DAYS = 60  # rolling window length (calendar days; ~42 trading days)
SIGNAL_LAG_DAYS = 1  # trading days lag between signal and feature date
MIN_OBS = 10  # minimum non-NaN observations for a valid z-score
Z_CAP = 4.0  # symmetric z-score cap to reduce outlier impact
FEATURE_VERSION = "v1"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------


def _raw_value(source: str, payload: dict[str, Any]) -> dict[str, float | None]:
    """Extract raw numeric values from a signal payload_json dict."""
    if source == "wikipedia_pageviews":
        views = payload.get("views")
        return {"wiki_views_z60d": float(np.log1p(views)) if views is not None else None}

    if source == "edgar":
        f4 = payload.get("form_4_count")
        k8 = payload.get("form_8k_count")
        return {
            "edgar_form4_z60d": float(f4) if f4 is not None else None,
            "edgar_8k_z60d": float(k8) if k8 is not None else None,
        }

    if source == "gdelt":
        n = payload.get("article_count")
        tone = payload.get("avg_tone")
        nlp = payload.get("nlp", {})
        finbert = (
            nlp.get("finbert_avg_positive") - nlp.get("finbert_avg_negative")
            if nlp.get("finbert_avg_positive") is not None
            and nlp.get("finbert_avg_negative") is not None
            else None
        )
        vader = nlp.get("vader_avg_compound")
        return {
            "gdelt_n_z60d": float(np.log1p(n)) if n is not None else None,
            "gdelt_tone_z60d": float(tone) if tone is not None else None,
            "gdelt_finbert_z60d": float(finbert) if finbert is not None else None,
            "gdelt_vader_z60d": float(vader) if vader is not None else None,
        }

    if source == "google_trends":
        val = payload.get("anchor_normalized_interest")
        return {"trends_z60d": float(val) if val is not None else None}

    return {}


# ---------------------------------------------------------------------------
# Archive I/O
# ---------------------------------------------------------------------------


def _load_raw_source(
    data_repo: Path,
    source: str,
    ticker: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Load all raw signal records for one ticker from the archive,
    across [start_date, end_date]. Returns a DataFrame with columns:
      security_id, effective_date, payload_json (as dict)
    """
    rows = []
    current = start_date
    while current <= end_date:
        path = (
            data_repo
            / "raw"
            / source
            / f"year={current.year}"
            / f"month={current.month:02d}"
            / f"day={current.day:02d}"
            / f"{ticker.lower()}.parquet"
        )
        if path.exists():
            try:
                tbl = pq.read_table(path)
                df = tbl.to_pandas()
                for _, row in df.iterrows():
                    rows.append(
                        {
                            "security_id": row["security_id"],
                            "effective_date": current,
                            "payload": json.loads(row["payload_json"]),
                        }
                    )
            except Exception as e:
                log.warning("features: failed to read parquet", path=str(path), error=str(e))
        current += timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["security_id", "effective_date", "payload"])
    return pd.DataFrame(rows)


def _write_features(
    data_repo: Path,
    target_date: date,
    records: list[dict[str, Any]],
) -> int:
    if not records:
        return 0
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    out_path = (
        data_repo
        / "processed"
        / "features"
        / f"date={target_date.isoformat()}"
        / "features.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        df[
            [
                "security_id",
                "date",
                "feature_name",
                "value",
                "rank_cs",
                "z_cs",
                "n_obs",
                "shrunk_value",
                "feature_version",
                "schema_version",
            ]
        ],
        schema=SIGNAL_FEATURE_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, out_path, compression="zstd")
    return len(df)


# ---------------------------------------------------------------------------
# Rolling z-score
# ---------------------------------------------------------------------------


def rolling_zscore(series: pd.Series, lookback: int, min_obs: int) -> pd.Series:
    """
    Compute rolling z-score: (x - rolling_mean) / rolling_std.
    Uses a trailing window of `lookback` observations.
    Requires >= min_obs non-NaN values in the window.
    """
    mean = series.rolling(window=lookback, min_periods=min_obs).mean()
    std = series.rolling(window=lookback, min_periods=min_obs).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.clip(-Z_CAP, Z_CAP)


def rolling_nobs(series: pd.Series, lookback: int) -> pd.Series:
    """Count non-NaN observations in the trailing window."""
    return series.rolling(window=lookback, min_periods=1).count().astype(int)


# ---------------------------------------------------------------------------
# Cross-sectional normalisation
# ---------------------------------------------------------------------------


def cross_sectional_rank(
    df: pd.DataFrame,
    value_col: str,
    sector_col: str = "gics_sector",
) -> pd.Series:
    """
    Within each GICS sector, rank securities by value_col.
    Returns fractional rank in [0, 1]. NaN for missing values.
    """
    ranks = pd.Series(index=df.index, dtype=float)
    for _, grp in df.groupby(sector_col):
        valid = grp[value_col].notna()
        if valid.sum() < 2:
            continue
        r = grp.loc[valid, value_col].rank(pct=True)
        ranks.loc[r.index] = r
    return ranks


def cross_sectional_zscore(
    df: pd.DataFrame,
    value_col: str,
    sector_col: str = "gics_sector",
) -> pd.Series:
    """Within each GICS sector, standardise value_col to mean=0, std=1."""
    zscores = pd.Series(index=df.index, dtype=float)
    for _, grp in df.groupby(sector_col):
        valid = grp[value_col].notna()
        if valid.sum() < 3:
            continue
        vals = grp.loc[valid, value_col]
        mu = vals.mean()
        sigma = vals.std()
        z = (vals - mu) / sigma if sigma > 0 else pd.Series(0.0, index=vals.index)
        zscores.loc[z.index] = z
    return zscores


# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage
# ---------------------------------------------------------------------------


def shrink_toward_zero(z_cs: pd.Series, n_obs: pd.Series) -> pd.Series:
    """
    Shrink cross-sectional z-scores toward zero using a simple James-Stein
    shrinkage factor based on the number of observations in the rolling window.

    B = 1 / (1 + n_obs / 10)   (B close to 1 → heavy shrinkage for low n_obs)
    shrunk = (1 - B) * z_cs
    """
    shrinkage = 1.0 / (1.0 + n_obs / 10.0)
    return (1.0 - shrinkage) * z_cs


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

SOURCES = [
    "wikipedia_pageviews",
    "edgar",
    "gdelt",
    "google_trends",
]


class FeatureBuilder:
    def __init__(
        self,
        data_repo_path: Path,
        lookback_days: int = LOOKBACK_DAYS,
        signal_lag_days: int = SIGNAL_LAG_DAYS,
        min_obs: int = MIN_OBS,
    ) -> None:
        self.data_repo = data_repo_path
        self.lookback_days = lookback_days
        self.signal_lag_days = signal_lag_days
        self.min_obs = min_obs

    def _load_ticker_history(
        self,
        source: str,
        ticker: str,
        window_start: date,
        window_end: date,
    ) -> pd.DataFrame:
        """Load and expand raw signal records into a per-feature timeseries."""
        raw = _load_raw_source(self.data_repo, source, ticker, window_start, window_end)
        if raw.empty:
            return pd.DataFrame()

        rows = []
        for _, row in raw.iterrows():
            feature_vals = _raw_value(source, row["payload"])
            for feat_name, val in feature_vals.items():
                rows.append(
                    {
                        "effective_date": row["effective_date"],
                        "security_id": row["security_id"],
                        "feature_name": feat_name,
                        "raw_value": val,
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def build_date(
        self,
        target_date: date,
        security_master: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Build all features for target_date.
        Returns a list of SIGNAL_FEATURE_SCHEMA records.
        """
        # Window: [target_date - lookback - lag, target_date - lag]
        window_end = target_date - timedelta(days=self.signal_lag_days)
        window_start = window_end - timedelta(days=self.lookback_days)

        active = security_master[security_master["valid_to"].isna()].copy()

        # Accumulate rolling z-score values per (security_id, feature_name)
        zscore_rows: list[dict[str, Any]] = []

        for _, sec_row in active.iterrows():
            ticker = sec_row["ticker"]
            sid = sec_row["security_id"]
            sector = sec_row["gics_sector"]

            for source in SOURCES:
                hist = self._load_ticker_history(source, ticker, window_start, window_end)
                if hist.empty:
                    continue

                for feat_name, grp in hist.groupby("feature_name"):
                    grp = grp.sort_values("effective_date").drop_duplicates("effective_date")
                    grp = grp.set_index("effective_date")["raw_value"]

                    # Compute rolling z-score on the full history
                    z = rolling_zscore(grp, self.lookback_days, self.min_obs)
                    n = rolling_nobs(grp, self.lookback_days)

                    # Take the value as of window_end
                    if window_end not in z.index or pd.isna(z.loc[window_end]):
                        continue
                    n_obs_val = int(n.loc[window_end]) if window_end in n.index else 0
                    if n_obs_val < self.min_obs:
                        continue

                    zscore_rows.append(
                        {
                            "security_id": sid,
                            "date": target_date,
                            "feature_name": feat_name,
                            "value": float(z.loc[window_end]),
                            "n_obs": n_obs_val,
                            "gics_sector": sector,
                        }
                    )

        if not zscore_rows:
            return []

        df = pd.DataFrame(zscore_rows)

        # Cross-sectional normalisation per feature
        all_records: list[dict[str, Any]] = []
        for _feat_name, feat_grp in df.groupby("feature_name"):
            feat_grp = feat_grp.copy().reset_index(drop=True)
            feat_grp["rank_cs"] = cross_sectional_rank(feat_grp, "value")
            feat_grp["z_cs"] = cross_sectional_zscore(feat_grp, "value")
            feat_grp["shrunk_value"] = shrink_toward_zero(feat_grp["z_cs"], feat_grp["n_obs"])

            for _, r in feat_grp.iterrows():
                all_records.append(
                    {
                        "security_id": r["security_id"],
                        "date": r["date"],
                        "feature_name": r["feature_name"],
                        "value": r["value"] if pd.notna(r["value"]) else None,
                        "rank_cs": r["rank_cs"] if pd.notna(r["rank_cs"]) else None,
                        "z_cs": r["z_cs"] if pd.notna(r["z_cs"]) else None,
                        "n_obs": int(r["n_obs"]),
                        "shrunk_value": r["shrunk_value"] if pd.notna(r["shrunk_value"]) else None,
                        "feature_version": FEATURE_VERSION,
                        "schema_version": SCHEMA_VERSION,
                    }
                )

        return all_records

    def build_range(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, int]:
        """Build features for every date in [start_date, end_date]."""
        sm = load_security_master(self.data_repo)
        summary: dict[str, int] = {}
        current = start_date
        total_rows = 0

        while current <= end_date:
            records = self.build_date(current, sm)
            rows = _write_features(self.data_repo, current, records)
            total_rows += rows
            log.info("features: built", date=current.isoformat(), rows=rows)
            summary[current.isoformat()] = rows
            current += timedelta(days=1)

        log.info("features: range complete", total_rows=total_rows)
        return summary


def load_features(
    data_repo: Path,
    start_date: date,
    end_date: date,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Load feature records from the archive for a date range."""
    frames = []
    current = start_date
    while current <= end_date:
        path = (
            data_repo
            / "processed"
            / "features"
            / f"date={current.isoformat()}"
            / "features.parquet"
        )
        if path.exists():
            try:
                df = pq.read_table(path).to_pandas()
                if feature_names:
                    df = df[df["feature_name"].isin(feature_names)]
                frames.append(df)
            except Exception as e:
                log.warning("features: failed to read", path=str(path), error=str(e))
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(
            columns=[
                "security_id",
                "date",
                "feature_name",
                "value",
                "rank_cs",
                "z_cs",
                "n_obs",
                "shrunk_value",
                "feature_version",
                "schema_version",
            ]
        )
    return pd.concat(frames, ignore_index=True)

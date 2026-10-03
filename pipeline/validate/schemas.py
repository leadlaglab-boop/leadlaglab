"""
Canonical data contracts for all pipeline schemas.
Schema version is embedded in every Parquet file and in manifest.json.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Pydantic models (for row-level validation and documentation)
# ---------------------------------------------------------------------------


class SecurityMaster(BaseModel):
    security_id: str = Field(description="Stable internal identifier; never changes")
    ticker: str
    valid_from: date
    valid_to: date | None = Field(None, description="None means still active")
    name: str
    gics_sector: str
    cohort: Literal["sp500", "retail_attention"]


class PricesDaily(BaseModel):
    security_id: str
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int
    observed_at: datetime = Field(
        description="When this record was fetched; used for look-ahead enforcement"
    )
    source: str


class SignalRaw(BaseModel):
    source: str
    security_id: str
    effective_date: date = Field(
        description="The period this observation describes"
    )
    observed_at: datetime = Field(
        description="When we collected it; must be < market_close(decision_date)"
    )
    payload: dict = Field(description="Source-specific typed fields")
    n_items: int = Field(description="Number of underlying items (articles, filings, etc.)")
    source_version: str


class SignalFeature(BaseModel):
    security_id: str
    date: date
    feature_name: str
    value: float
    rank_cs: float = Field(description="Cross-sectional rank within date and sector, 0–1")
    z_cs: float = Field(description="Cross-sectional z-score within date and sector")
    n_obs: int = Field(description="Observations in the trailing window used for this value")
    shrunk_value: float = Field(
        description="Value after empirical-Bayes shrinkage toward cross-sectional mean"
    )
    feature_version: str


class PredictionLedger(BaseModel):
    prediction_id: str = Field(description="UUID; unique per prediction row")
    made_at: datetime = Field(description="Timestamp when prediction was committed; immutable")
    security_id: str
    target_date_start: date
    target_date_end: date
    horizon_days: Literal[1, 5, 21]
    model_id: str
    model_version: str
    data_hash: str = Field(description="SHA-256 of input features at prediction time")
    predicted_excess_return: float
    predicted_direction: Literal[-1, 0, 1]
    ci_lower: float = Field(description="95% confidence interval lower bound")
    ci_upper: float = Field(description="95% confidence interval upper bound")
    features_used: list[str]


class Outcome(BaseModel):
    prediction_id: str
    realized_excess_return: float
    scored_at: datetime


# ---------------------------------------------------------------------------
# PyArrow schemas (for Parquet storage)
# ---------------------------------------------------------------------------


SECURITY_MASTER_SCHEMA = pa.schema([
    pa.field("security_id", pa.string(), nullable=False),
    pa.field("ticker", pa.string(), nullable=False),
    pa.field("valid_from", pa.date32(), nullable=False),
    pa.field("valid_to", pa.date32(), nullable=True),
    pa.field("name", pa.string(), nullable=False),
    pa.field("gics_sector", pa.string(), nullable=False),
    pa.field("cohort", pa.string(), nullable=False),
    pa.field("schema_version", pa.string(), nullable=False),
])

PRICES_DAILY_SCHEMA = pa.schema([
    pa.field("security_id", pa.string(), nullable=False),
    pa.field("date", pa.date32(), nullable=False),
    pa.field("open", pa.float64(), nullable=True),
    pa.field("high", pa.float64(), nullable=True),
    pa.field("low", pa.float64(), nullable=True),
    pa.field("close", pa.float64(), nullable=False),
    pa.field("adj_close", pa.float64(), nullable=False),
    pa.field("volume", pa.int64(), nullable=True),
    pa.field("observed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("schema_version", pa.string(), nullable=False),
])

SIGNAL_RAW_SCHEMA = pa.schema([
    pa.field("source", pa.string(), nullable=False),
    pa.field("security_id", pa.string(), nullable=False),
    pa.field("effective_date", pa.date32(), nullable=False),
    pa.field("observed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("payload_json", pa.string(), nullable=False),  # serialized payload
    pa.field("n_items", pa.int32(), nullable=False),
    pa.field("source_version", pa.string(), nullable=False),
    pa.field("schema_version", pa.string(), nullable=False),
])

SIGNAL_FEATURE_SCHEMA = pa.schema([
    pa.field("security_id", pa.string(), nullable=False),
    pa.field("date", pa.date32(), nullable=False),
    pa.field("feature_name", pa.string(), nullable=False),
    pa.field("value", pa.float64(), nullable=True),
    pa.field("rank_cs", pa.float64(), nullable=True),
    pa.field("z_cs", pa.float64(), nullable=True),
    pa.field("n_obs", pa.int32(), nullable=False),
    pa.field("shrunk_value", pa.float64(), nullable=True),
    pa.field("feature_version", pa.string(), nullable=False),
    pa.field("schema_version", pa.string(), nullable=False),
])

PREDICTION_LEDGER_SCHEMA = pa.schema([
    pa.field("prediction_id", pa.string(), nullable=False),
    pa.field("made_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("security_id", pa.string(), nullable=False),
    pa.field("target_date_start", pa.date32(), nullable=False),
    pa.field("target_date_end", pa.date32(), nullable=False),
    pa.field("horizon_days", pa.int32(), nullable=False),
    pa.field("model_id", pa.string(), nullable=False),
    pa.field("model_version", pa.string(), nullable=False),
    pa.field("data_hash", pa.string(), nullable=False),
    pa.field("predicted_excess_return", pa.float64(), nullable=False),
    pa.field("predicted_direction", pa.int8(), nullable=False),
    pa.field("ci_lower", pa.float64(), nullable=False),
    pa.field("ci_upper", pa.float64(), nullable=False),
    pa.field("features_used_json", pa.string(), nullable=False),
    pa.field("schema_version", pa.string(), nullable=False),
])

OUTCOME_SCHEMA = pa.schema([
    pa.field("prediction_id", pa.string(), nullable=False),
    pa.field("realized_excess_return", pa.float64(), nullable=False),
    pa.field("scored_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("schema_version", pa.string(), nullable=False),
])

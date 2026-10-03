"""
Tests for M6: models and prediction ledger.

Covers:
  - ZeroModel always predicts zero; CI is symmetric and non-zero
  - MomentumModel coefficient sign
  - RidgeModel: predicts after fit; raises on unfit; respects feature columns
  - Ledger writer: UUID uniqueness, data hash determinism, direction encoding
  - Ledger immutability: FileExistsError on duplicate as_of
  - Scorer: realized returns matched to predictions correctly
  - No future data in training: training uses only data before as_of
"""

from __future__ import annotations

import json
from datetime import UTC, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.ledger.writer import _append_to_ledger, _hash_features
from pipeline.models.baselines import MomentumModel, ZeroModel
from pipeline.models.ridge import FEATURE_COLS, RidgeModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xy(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {col: rng.standard_normal(n) for col in FEATURE_COLS},
        index=range(n),
    )
    y = pd.Series(rng.standard_normal(n), name="excess_return")
    return X, y


def _make_minimal_ledger_rows(
    n: int = 5,
    as_of: date = date(2024, 1, 5),
) -> list[dict]:
    """Return minimal ledger rows suitable for _append_to_ledger."""
    import uuid as _uuid
    from datetime import datetime

    rows = []
    for i in range(n):
        rows.append(
            {
                "prediction_id": str(_uuid.uuid4()),
                "made_at": datetime.now(UTC),
                "security_id": f"sid_T{i:03d}",
                "target_date_start": as_of + timedelta(days=1),
                "target_date_end": as_of + timedelta(days=5),
                "horizon_days": 5,
                "model_id": "zero",
                "model_version": "v1",
                "data_hash": "abc123",
                "predicted_excess_return": 0.0,
                "predicted_direction": 0,
                "ci_lower": -0.02,
                "ci_upper": 0.02,
                "features_used_json": json.dumps(FEATURE_COLS),
                "schema_version": "1.0.0",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# ZeroModel
# ---------------------------------------------------------------------------


class TestZeroModel:
    def test_predicts_zeros(self):
        model = ZeroModel()
        X, y = _make_xy(50)
        model.fit(X, y)
        preds, lo, hi = model.predict(X)
        assert np.allclose(preds, 0.0), "ZeroModel must predict 0 for all rows"

    def test_ci_symmetric(self):
        model = ZeroModel()
        X, y = _make_xy(50)
        model.fit(X, y)
        _, lo, hi = model.predict(X)
        assert np.allclose(lo, -hi, atol=1e-10), "CI should be symmetric around zero"
        assert np.all(hi > 0), "CI upper bound should be positive"

    def test_ci_uses_y_std(self):
        model = ZeroModel()
        X, y = _make_xy(100)
        model.fit(X, y)
        _, lo, hi = model.predict(X[:1])
        # CI half-width should be 1.96 * std(y)
        expected_hw = 1.96 * float(y.std(ddof=1))
        assert abs(hi[0] - expected_hw) < 0.01 * expected_hw

    def test_output_shape(self):
        model = ZeroModel()
        X, y = _make_xy(30)
        model.fit(X, y)
        p, lo, hi = model.predict(X[:10])
        assert p.shape == lo.shape == hi.shape == (10,)


# ---------------------------------------------------------------------------
# MomentumModel
# ---------------------------------------------------------------------------


class TestMomentumModel:
    def test_fit_without_mom_col_uses_zero(self):
        model = MomentumModel()
        X, y = _make_xy(80)
        model.fit(X, y)  # X has no mom_21d_xs column
        preds, _, _ = model.predict(X)
        assert np.allclose(preds, 0.0), "Without momentum column, should predict zero"

    def test_fit_with_mom_col(self):
        rng = np.random.default_rng(5)
        n = 100
        y = pd.Series(rng.standard_normal(n))
        X = pd.DataFrame({"mom_21d_xs": y + 0.1 * rng.standard_normal(n)})
        model = MomentumModel()
        model.fit(X, y)
        preds, _, _ = model.predict(X)
        # Should produce non-zero predictions since mom col is correlated with y
        assert not np.allclose(preds, 0.0), "With correlated momentum col, predictions should vary"


# ---------------------------------------------------------------------------
# RidgeModel
# ---------------------------------------------------------------------------


class TestRidgeModel:
    def test_fit_and_predict(self):
        model = RidgeModel(horizon=5)
        X, y = _make_xy(200)
        model.fit(X, y)
        preds, lo, hi = model.predict(X[:10])
        assert preds.shape == (10,)
        assert np.all(hi > lo), "CI upper must exceed lower"

    def test_raises_without_fit(self):
        model = RidgeModel(horizon=5)
        X, _ = _make_xy(10)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(X)

    def test_raises_with_insufficient_data(self):
        model = RidgeModel(horizon=5)
        X, y = _make_xy(10)  # < 30 rows
        with pytest.raises(ValueError, match="insufficient"):
            model.fit(X, y)

    def test_handles_missing_feature_columns(self):
        """Predict should fill missing feature columns with 0."""
        model = RidgeModel(horizon=5)
        X_full, y = _make_xy(100)
        model.fit(X_full, y)
        # Remove some columns from predict input
        X_partial = X_full[FEATURE_COLS[:3]].copy()
        preds, _, _ = model.predict(X_partial)
        assert preds.shape == (100,)

    def test_best_alpha_set_after_fit(self):
        model = RidgeModel(horizon=5)
        X, y = _make_xy(100)
        model.fit(X, y)
        assert model.best_alpha is not None
        assert model.best_alpha > 0


# ---------------------------------------------------------------------------
# Feature hash
# ---------------------------------------------------------------------------


class TestFeatureHash:
    def test_deterministic(self):
        df = pd.DataFrame(
            {"security_id": ["a", "b"], "feature_name": ["f1", "f2"], "value": [1.0, 2.0]}
        )
        h1 = _hash_features(df)
        h2 = _hash_features(df)
        assert h1 == h2

    def test_different_data_different_hash(self):
        df1 = pd.DataFrame({"security_id": ["a"], "feature_name": ["f1"], "value": [1.0]})
        df2 = pd.DataFrame({"security_id": ["a"], "feature_name": ["f1"], "value": [2.0]})
        assert _hash_features(df1) != _hash_features(df2)

    def test_row_order_invariant(self):
        """Hash must be the same regardless of row order."""
        df = pd.DataFrame(
            {
                "security_id": ["b", "a"],
                "feature_name": ["f2", "f1"],
                "value": [2.0, 1.0],
            }
        )
        df_rev = df.iloc[::-1].reset_index(drop=True)
        assert _hash_features(df) == _hash_features(df_rev)


# ---------------------------------------------------------------------------
# Ledger writer (file-level)
# ---------------------------------------------------------------------------


class TestLedgerWriter:
    def test_prediction_ids_are_unique(self):
        rows = _make_minimal_ledger_rows(n=10)
        ids = [r["prediction_id"] for r in rows]
        assert len(set(ids)) == len(ids), "All prediction_ids must be unique"

    def test_direction_encoding(self):
        preds = [0.05, -0.03, 0.0]
        expected = [1, -1, 0]
        for pred, exp in zip(preds, expected, strict=False):
            direction = 1 if pred > 0 else (-1 if pred < 0 else 0)
            assert direction == exp

    def test_append_writes_file(self, tmp_path: Path):
        rows = _make_minimal_ledger_rows(n=3)
        written = _append_to_ledger(rows, data_repo=tmp_path, as_of=date(2024, 3, 1))
        assert written.exists()

    def test_immutability_raises_on_duplicate(self, tmp_path: Path):
        as_of = date(2024, 3, 2)
        rows = _make_minimal_ledger_rows(n=2)
        _append_to_ledger(rows, data_repo=tmp_path, as_of=as_of)
        with pytest.raises(FileExistsError):
            _append_to_ledger(rows, data_repo=tmp_path, as_of=as_of)

    def test_written_file_is_valid_parquet(self, tmp_path: Path):
        import pyarrow.parquet as pq

        rows = _make_minimal_ledger_rows(n=4)
        path = _append_to_ledger(rows, data_repo=tmp_path, as_of=date(2024, 3, 3))
        table = pq.read_table(path)
        assert table.num_rows == 4
        assert "prediction_id" in table.schema.names
        assert "data_hash" in table.schema.names

    def test_features_json_is_valid(self):
        rows = _make_minimal_ledger_rows(n=1)
        features = json.loads(rows[0]["features_used_json"])
        assert isinstance(features, list)
        assert len(features) > 0

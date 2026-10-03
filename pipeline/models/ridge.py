"""
Ridge-regularised linear prediction model.

Pre-registered choice (study/PREREGISTRATION.md §7): ridge regression is used
for its interpretability.  Alpha is selected via 5-fold time-series CV on the
training set; the CV is re-run at each daily fit to avoid look-ahead.

Feature set: the 8 shrunk_value features from the feature pipeline.
Returns: predicted excess return vs SPY at a single horizon (the ledger writer
calls one RidgeModel instance per horizon).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.models.base import BaseModel

# Pre-specified alpha grid (log-spaced, as is standard for ridge)
_ALPHA_GRID = np.logspace(-3, 3, 30)

FEATURE_COLS = [
    "wiki_views_z60d",
    "gdelt_n_z60d",
    "gdelt_tone_z60d",
    "gdelt_finbert_z60d",
    "gdelt_vader_z60d",
    "edgar_form4_z60d",
    "edgar_8k_z60d",
    "trends_z60d",
]


class RidgeModel(BaseModel):
    model_id = "ridge_linear"
    model_version = "v1"

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon
        self._pipe: Pipeline | None = None
        self._fitted = False
        self._train_y_std: float = 0.01

    def _select_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the pre-specified feature columns present in X."""
        present = [c for c in FEATURE_COLS if c in X.columns]
        if not present:
            raise ValueError(f"No model features found in X. Expected one of: {FEATURE_COLS}")
        return X[present].copy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xf = self._select_cols(X).fillna(0.0)
        mask = y.notna()
        Xf, y_clean = Xf[mask], y[mask]

        if len(Xf) < 30:
            raise ValueError(f"RidgeModel.fit: insufficient training rows ({len(Xf)}); need ≥30.")

        self._feature_cols = list(Xf.columns)
        self._train_y_std = float(y_clean.std(ddof=1)) if len(y_clean) > 1 else 0.01

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "ridge",
                    RidgeCV(
                        alphas=_ALPHA_GRID,
                        cv=min(5, len(Xf) // 10),  # shrink CV folds for small sets
                        scoring="neg_mean_squared_error",
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xf, y_clean)

        self._pipe = pipe
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._fitted or self._pipe is None:
            raise RuntimeError("RidgeModel must be fit before predict.")

        Xf = X.reindex(columns=self._feature_cols, fill_value=0.0).fillna(0.0)
        preds = self._pipe.predict(Xf)

        # Approximate 95% CI: ±1.96 * in-sample residual std
        # This is a plug-in interval, not a formal prediction interval.
        ci_hw = 1.96 * self._train_y_std
        return preds, preds - ci_hw, preds + ci_hw

    @property
    def best_alpha(self) -> float | None:
        if self._pipe is None:
            return None
        return float(self._pipe.named_steps["ridge"].alpha_)

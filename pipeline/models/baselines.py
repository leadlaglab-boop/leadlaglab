"""
Baseline models for the prediction ledger.

Two baselines, as pre-registered:
  ZeroModel    — always predicts 0 excess return (the null)
  MomentumModel — predicts based on trailing 1-month return vs SPY
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.models.base import BaseModel

# 95% CI half-width derived from rolling cross-sectional std of excess returns;
# set to a reasonable placeholder (0.02 = 2%) until calibrated from data.
_DEFAULT_CI_HALFWIDTH = 0.02


class ZeroModel(BaseModel):
    """Predicts zero excess return for all stocks. The hard null baseline."""

    model_id = "zero"
    model_version = "v1"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Store cross-sectional std of y for CI calibration
        self._y_std = float(y.std(ddof=1)) if len(y) > 1 else _DEFAULT_CI_HALFWIDTH
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        ci_hw = 1.96 * self._y_std if self.is_fitted else _DEFAULT_CI_HALFWIDTH
        zeros = np.zeros(n)
        return zeros, zeros - ci_hw, zeros + ci_hw


class MomentumModel(BaseModel):
    """
    Momentum baseline: predicts trailing 21-day excess return (vs SPY) as the
    signal for the next period.

    Requires a column `mom_21d_xs` in X (pre-computed by the feature pipeline
    or caller).  If absent, falls back to zero.
    """

    model_id = "momentum"
    model_version = "v1"

    _MOM_COL = "mom_21d_xs"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self._MOM_COL in X.columns:
            from sklearn.linear_model import Ridge

            mom = X[[self._MOM_COL]].fillna(0)
            self._coef = float(Ridge(alpha=1.0).fit(mom, y).coef_[0])
            self._intercept = float(Ridge(alpha=1.0).fit(mom, y).intercept_)
        else:
            self._coef = 0.0
            self._intercept = 0.0
        self._y_std = float(y.std(ddof=1)) if len(y) > 1 else _DEFAULT_CI_HALFWIDTH
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._MOM_COL in X.columns:
            preds = X[self._MOM_COL].fillna(0).to_numpy(dtype=float) * self._coef + self._intercept
        else:
            preds = np.zeros(len(X))
        ci_hw = 1.96 * getattr(self, "_y_std", _DEFAULT_CI_HALFWIDTH)
        return preds, preds - ci_hw, preds + ci_hw

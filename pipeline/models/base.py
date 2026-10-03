"""
Base interface for all prediction models.

Every model must implement:
  fit(X: pd.DataFrame, y: pd.Series) -> None
  predict(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    Returns (predicted_excess_return, ci_lower, ci_upper)

Models are stateless until fit(); serialised via pickle by the ledger writer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    model_id: str  # short identifier used in the ledger
    model_version: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point_estimates, ci_lower, ci_upper) arrays of shape (n,)."""
        ...

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "_fitted", False)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class QlibStyleBaseline:
    """
    Qlib-inspired institutional public baseline:
    gradient boosting classifier with conservative capacity,
    analogous to a LightGBM Alpha workflow reference.
    """

    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = HistGradientBoostingClassifier(
            max_depth=8,
            max_iter=350,
            learning_rate=0.04,
            min_samples_leaf=30,
            l2_regularization=0.20,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        target = pd.Series(y).astype(int)
        if target.nunique() < 2:
            raise ValueError("Baseline target has only one class")
        self.model.fit(X, target)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

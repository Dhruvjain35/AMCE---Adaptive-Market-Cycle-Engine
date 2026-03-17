"""Legacy compatibility wrapper for ensemble modeling."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from amce.config.schema import ModelConfig
from amce.models import EnsembleStack, learn_threshold



def train_ensemble(
    data: pd.DataFrame,
    features: List[str],
    embargo_months: int = 4,
    train_pct: float = 0.40,
    rf_params: dict | None = None,
    gb_params: dict | None = None,
) -> Tuple[pd.DataFrame, EnsembleStack, pd.DataFrame]:
    if not 0.0 < train_pct < 1.0:
        raise ValueError(f"train_pct must be between 0 and 1, got {train_pct}")

    split = int(len(data) * train_pct)
    embargo_days = int((embargo_months / 12) * 252)
    test_start = split + embargo_days
    if test_start >= len(data):
        test_start = split + 1

    train = data.iloc[:split].copy()
    test = data.iloc[test_start:].copy()

    if len(train) < 100 or len(test) < 50:
        raise ValueError("Insufficient rows for train/test after embargo.")

    cfg = ModelConfig()
    if rf_params:
        if "n_estimators" in rf_params:
            cfg.rf_estimators = int(rf_params["n_estimators"])
        if "max_depth" in rf_params:
            cfg.rf_max_depth = int(rf_params["max_depth"])
        if "min_samples_leaf" in rf_params:
            cfg.rf_min_samples_leaf = int(rf_params["min_samples_leaf"])
    if gb_params:
        if "n_estimators" in gb_params:
            cfg.gb_estimators = int(gb_params["n_estimators"])
        if "max_depth" in gb_params:
            cfg.gb_max_depth = int(gb_params["max_depth"])
        if "learning_rate" in gb_params:
            cfg.gb_learning_rate = float(gb_params["learning_rate"])

    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(train[features]), index=train.index, columns=features)
    x_test = pd.DataFrame(scaler.transform(test[features]), index=test.index, columns=features)

    model = EnsembleStack(cfg)
    model.fit(x_train, train["Target"])

    train_probs = model.predict_proba(x_train)
    test_probs = model.predict_proba(x_test)

    threshold = learn_threshold(
        train_probs,
        train["Risk"].pct_change().fillna(0.0),
        train["Safe"].pct_change().fillna(0.0),
        cfg,
    )

    test["Prob_Avg"] = test_probs
    test["Prob_Smooth"] = pd.Series(test_probs, index=test.index).ewm(span=10).mean()
    test["Signal"] = (test["Prob_Smooth"] >= threshold).astype(int)

    risk_off = (test.get("MA_200_Dist", 0) < 0) & (test.get("VIX_Change_21D", 0) > 0)
    panic = (test.get("Drawdown_63", 0) < -0.1) & (test.get("VIX_Change_5D", 0) > 0.10)
    test.loc[risk_off | panic, "Signal"] = 0

    test["Regime"] = "Normal"
    test.loc[risk_off, "Regime"] = "Risk-Off"
    test.loc[panic, "Regime"] = "Panic"

    return test, model, train

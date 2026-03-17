"""Legacy compatibility wrapper for feature engineering."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from amce.features import compute_feature_frame
from amce.labels import add_targets

FEATURE_COLUMNS: List[str] = [
    "Mom_21D",
    "Mom_63D",
    "MA_200_Dist",
    "Yield_Change_21D",
    "Risk_Safe_Vol_Ratio_21",
    "RSI_14",
]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    ff = compute_feature_frame(df)
    data = add_targets(ff.data, horizon_days=1)
    data = data.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return data, FEATURE_COLUMNS

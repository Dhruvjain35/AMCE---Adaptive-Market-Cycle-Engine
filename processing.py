from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller


@dataclass
class DataProcessingReport:
    rows_in: int
    rows_out: int
    missing_filled: int
    stationarity_pvalues: dict[str, float]


def clean_market_data(df: pd.DataFrame, winsor_limits: tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
    out = df.sort_index().copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().bfill()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            arr = out[col].to_numpy(dtype=float)
            out[col] = winsorize(arr, limits=winsor_limits)

    return out


def compute_stationarity_pvalues(df: pd.DataFrame, max_cols: int = 50) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in list(df.columns)[:max_cols]:
        s = df[col].dropna()
        if len(s) < 50 or not pd.api.types.is_numeric_dtype(s):
            continue
        try:
            out[col] = float(adfuller(s, autolag="AIC")[1])
        except Exception:
            continue
    return out


def add_lagged_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, columns: list[str], windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for w in windows:
            out[f"{col}_roll_mean_{w}"] = out[col].rolling(w).mean()
            out[f"{col}_roll_std_{w}"] = out[col].rolling(w).std()
            out[f"{col}_roll_z_{w}"] = (out[col] - out[col].rolling(w).mean()) / out[col].rolling(w).std()
    return out


def scale_train_test(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    x_tr = pd.DataFrame(scaler.fit_transform(train[feature_cols]), index=train.index, columns=feature_cols)
    x_te = pd.DataFrame(scaler.transform(test[feature_cols]), index=test.index, columns=feature_cols)
    return x_tr, x_te, scaler


def prepare_data_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, DataProcessingReport]:
    rows_in = len(df)
    cleaned = clean_market_data(df)
    missing_filled = int(df.isna().sum().sum())
    pvals = compute_stationarity_pvalues(cleaned)
    rows_out = len(cleaned)

    report = DataProcessingReport(
        rows_in=rows_in,
        rows_out=rows_out,
        missing_filled=missing_filled,
        stationarity_pvalues=pvals,
    )
    return cleaned, report

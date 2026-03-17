from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True)
class PeerModelSpec:
    key: str
    label: str
    proxy_label: str
    source_url: str
    qlib_annualized_return: float
    qlib_information_ratio: float
    qlib_max_drawdown: float
    is_surrogate: bool


class _StackedProxy:
    def __init__(self, random_state: int) -> None:
        self.base_models = [
            RandomForestClassifier(n_estimators=140, max_depth=6, min_samples_leaf=20, n_jobs=-1, random_state=random_state),
            ExtraTreesClassifier(n_estimators=140, max_depth=6, min_samples_leaf=15, n_jobs=-1, random_state=random_state + 17),
            HistGradientBoostingClassifier(max_iter=220, max_depth=6, learning_rate=0.05, min_samples_leaf=25, random_state=random_state + 31),
        ]
        self.meta = LogisticRegression(max_iter=1000, random_state=random_state + 47)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y = pd.Series(y).astype(int)
        n_splits = min(4, max(2, len(X) // 300))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof = np.full((len(X), len(self.base_models)), np.nan)

        for tr_idx, val_idx in tscv.split(X):
            xtr, xval = X.iloc[tr_idx], X.iloc[val_idx]
            ytr = y.iloc[tr_idx]
            for j, m in enumerate(self.base_models):
                fitted = clone(m)
                fitted.fit(xtr, ytr)
                oof[val_idx, j] = fitted.predict_proba(xval)[:, 1]

        valid = ~np.isnan(oof).any(axis=1)
        if valid.sum() >= 80:
            self.meta.fit(oof[valid], y.iloc[valid])

        for m in self.base_models:
            m.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        stacked = np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_models])
        return self.meta.predict_proba(stacked)[:, 1]


def default_peer_model_names() -> list[str]:
    return [
        "doubleensemble",
        "lightgbm",
        "mlp",
        "tft",
        "xgboost",
        "catboost",
        "tra",
        "linear",
        "gats",
        "alstm",
    ]


def get_top_peer_specs() -> dict[str, PeerModelSpec]:
    readme_url = "https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/README.md"
    return {
        "doubleensemble": PeerModelSpec(
            key="doubleensemble",
            label="DoubleEnsemble",
            proxy_label="Stacked DoubleEnsemble surrogate",
            source_url=readme_url,
            qlib_annualized_return=0.1158,
            qlib_information_ratio=1.3432,
            qlib_max_drawdown=-0.0920,
            is_surrogate=True,
        ),
        "lightgbm": PeerModelSpec(
            key="lightgbm",
            label="LightGBM",
            proxy_label="HistGradientBoosting surrogate",
            source_url=readme_url,
            qlib_annualized_return=0.0901,
            qlib_information_ratio=1.0164,
            qlib_max_drawdown=-0.1038,
            is_surrogate=True,
        ),
        "mlp": PeerModelSpec(
            key="mlp",
            label="MLP",
            proxy_label="MLPClassifier",
            source_url=readme_url,
            qlib_annualized_return=0.0895,
            qlib_information_ratio=1.1408,
            qlib_max_drawdown=-0.1103,
            is_surrogate=False,
        ),
        "tft": PeerModelSpec(
            key="tft",
            label="TFT",
            proxy_label="Residual MLP surrogate",
            source_url=readme_url,
            qlib_annualized_return=0.0847,
            qlib_information_ratio=0.8131,
            qlib_max_drawdown=-0.1824,
            is_surrogate=True,
        ),
        "xgboost": PeerModelSpec(
            key="xgboost",
            label="XGBoost",
            proxy_label="GradientBoosting surrogate",
            source_url=readme_url,
            qlib_annualized_return=0.0780,
            qlib_information_ratio=0.9070,
            qlib_max_drawdown=-0.1168,
            is_surrogate=True,
        ),
        "catboost": PeerModelSpec(
            key="catboost",
            label="CatBoost",
            proxy_label="AdaBoost+shallow trees surrogate",
            source_url=readme_url,
            qlib_annualized_return=0.0765,
            qlib_information_ratio=0.8032,
            qlib_max_drawdown=-0.1092,
            is_surrogate=True,
        ),
        "tra": PeerModelSpec(
            key="tra",
            label="TRA",
            proxy_label="Mixture-tree routing surrogate",
            source_url=readme_url,
            qlib_annualized_return=0.0718,
            qlib_information_ratio=1.0835,
            qlib_max_drawdown=-0.0760,
            is_surrogate=True,
        ),
        "linear": PeerModelSpec(
            key="linear",
            label="Linear",
            proxy_label="LogisticRegression",
            source_url=readme_url,
            qlib_annualized_return=0.0692,
            qlib_information_ratio=0.9209,
            qlib_max_drawdown=-0.1509,
            is_surrogate=False,
        ),
        "gats": PeerModelSpec(
            key="gats",
            label="GATs",
            proxy_label="Feature-attention MLP surrogate",
            source_url=readme_url,
            qlib_annualized_return=0.0497,
            qlib_information_ratio=0.7338,
            qlib_max_drawdown=-0.0777,
            is_surrogate=True,
        ),
        "alstm": PeerModelSpec(
            key="alstm",
            label="ALSTM",
            proxy_label="Attention-LSTM surrogate (tabular MLP)",
            source_url=readme_url,
            qlib_annualized_return=0.0470,
            qlib_information_ratio=0.6992,
            qlib_max_drawdown=-0.1072,
            is_surrogate=True,
        ),
    }


def build_peer_model(model_name: str, random_state: int):
    name = model_name.lower().strip()

    if name == "doubleensemble":
        return _StackedProxy(random_state=random_state)
    if name == "lightgbm":
        return HistGradientBoostingClassifier(
            max_iter=240,
            max_depth=7,
            learning_rate=0.04,
            min_samples_leaf=24,
            random_state=random_state,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(96, 48),
            alpha=1e-4,
            learning_rate_init=0.003,
            max_iter=250,
            random_state=random_state,
        )
    if name == "tft":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=3e-4,
            learning_rate_init=0.002,
            max_iter=260,
            random_state=random_state,
        )
    if name == "xgboost":
        return GradientBoostingClassifier(
            n_estimators=260,
            max_depth=3,
            learning_rate=0.04,
            min_samples_leaf=20,
            random_state=random_state,
        )
    if name == "catboost":
        return AdaBoostClassifier(
            n_estimators=220,
            learning_rate=0.03,
            random_state=random_state,
        )
    if name == "tra":
        return RandomForestClassifier(
            n_estimators=260,
            max_depth=8,
            min_samples_leaf=18,
            criterion="entropy",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "linear":
        return LogisticRegression(C=0.8, max_iter=3000, random_state=random_state)
    if name == "gats":
        return MLPClassifier(
            hidden_layer_sizes=(72, 36),
            alpha=4e-4,
            learning_rate_init=0.0025,
            max_iter=220,
            random_state=random_state,
        )
    if name == "alstm":
        return MLPClassifier(
            hidden_layer_sizes=(112, 56),
            alpha=5e-4,
            learning_rate_init=0.002,
            max_iter=280,
            random_state=random_state,
        )

    raise ValueError(f"Unsupported peer model: {model_name}")

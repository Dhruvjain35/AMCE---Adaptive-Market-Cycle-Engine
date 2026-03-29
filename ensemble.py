from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit

from schema import ModelConfig


def _positive_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = getattr(model, "predict_proba")(X)
        arr = np.asarray(raw)
        if arr.ndim == 1:
            return arr
        return arr[:, 1]
    if hasattr(model, "decision_function"):
        score = np.asarray(getattr(model, "decision_function")(X))
        return 1.0 / (1.0 + np.exp(-score))
    pred = np.asarray(getattr(model, "predict")(X))
    return np.clip(pred.astype(float), 0.0, 1.0)


@dataclass
class EnsembleStack:
    config: ModelConfig

    def __post_init__(self) -> None:
        self.templates: list[object] = [
            LogisticRegression(C=self.config.logistic_c, max_iter=2000, random_state=self.config.random_state),
            LogisticRegression(
                C=self.config.logistic_l1_c,
                max_iter=3000,
                solver="liblinear",
                penalty="l1",
                random_state=self.config.random_state + 3,
            ),
            CalibratedClassifierCV(
                estimator=RidgeClassifier(alpha=self.config.ridge_alpha),
                method="sigmoid",
                cv=3,
            ),
            RandomForestClassifier(
                n_estimators=self.config.rf_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1,
            ),
            GradientBoostingClassifier(
                n_estimators=self.config.gb_estimators,
                max_depth=self.config.gb_max_depth,
                learning_rate=self.config.gb_learning_rate,
                random_state=self.config.random_state,
            ),
            HistGradientBoostingClassifier(
                max_depth=6,
                max_iter=240,
                learning_rate=0.04,
                min_samples_leaf=20,
                random_state=self.config.random_state + 5,
            ),
        ]
        self._add_optional_boosters()
        self.base_models: list[object] = []
        self.meta_model: LogisticRegression | None = None
        self.fitted_model_names: list[str] = []

    def _add_optional_boosters(self) -> None:
        if self.config.use_lightgbm_if_available:
            try:
                from lightgbm import LGBMClassifier  # type: ignore

                self.templates.append(
                    LGBMClassifier(
                        n_estimators=220,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.config.random_state + 11,
                        verbose=-1,
                    )
                )
            except Exception:
                pass

        if self.config.use_xgboost_if_available:
            try:
                from xgboost import XGBClassifier  # type: ignore

                self.templates.append(
                    XGBClassifier(
                        n_estimators=260,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        random_state=self.config.random_state + 17,
                    )
                )
            except Exception:
                pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y = pd.Series(y).astype(int)
        if y.nunique() < 2:
            raise ValueError("Training target has only one class.")

        n_splits = min(5, max(2, len(X) // 250))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof = np.full(len(X), np.nan)

        for tr_idx, val_idx in tscv.split(X):
            xtr, xval = X.iloc[tr_idx], X.iloc[val_idx]
            ytr = y.iloc[tr_idx]
            fold_probs = []
            for template in self.templates:
                m = clone(template)
                try:
                    m.fit(xtr, ytr)
                    fold_probs.append(_positive_proba(m, xval))
                except Exception:
                    fold_probs.append(np.full(len(xval), float(ytr.mean())))
            oof[val_idx] = np.mean(fold_probs, axis=0)

        valid = ~np.isnan(oof)
        if valid.sum() > 100:
            self.meta_model = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
            self.meta_model.fit(oof[valid].reshape(-1, 1), y.iloc[valid])

        self.base_models = []
        self.fitted_model_names = []
        for template in self.templates:
            model = clone(template)
            try:
                model.fit(X, y)
                self.base_models.append(model)
                self.fitted_model_names.append(type(model).__name__)
            except Exception:
                continue
        if not self.base_models:
            raise ValueError("No base models were successfully fitted.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_probs = np.mean([_positive_proba(m, X) for m in self.base_models], axis=0)
        if self.meta_model is None:
            return base_probs
        return self.meta_model.predict_proba(base_probs.reshape(-1, 1))[:, 1]

    def predict_proba_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Return calibrated probabilities and model disagreement uncertainty.
        Uncertainty is measured as std-dev of base-model probabilities.
        """
        base_matrix = np.column_stack([_positive_proba(m, X) for m in self.base_models])
        base_mean = base_matrix.mean(axis=1)
        if self.meta_model is None:
            calibrated = base_mean
        else:
            calibrated = self.meta_model.predict_proba(base_mean.reshape(-1, 1))[:, 1]
        uncertainty = base_matrix.std(axis=1)
        return calibrated, uncertainty

    def predict_component_probas(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return per-base-model probabilities for ensemble diagnostics.
        """
        data: dict[str, np.ndarray] = {}
        for i, model in enumerate(self.base_models):
            name = self.fitted_model_names[i] if i < len(self.fitted_model_names) else f"Model{i+1}"
            key = f"Ens_{name}_{i+1}"
            data[key] = _positive_proba(model, X)
        return pd.DataFrame(data, index=X.index)

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        scores = pd.Series(0.0, index=feature_names, dtype=float)
        used = 0
        for model in self.base_models:
            values: np.ndarray | None = None
            if hasattr(model, "feature_importances_"):
                values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
            elif hasattr(model, "coef_"):
                coef = np.asarray(getattr(model, "coef_"), dtype=float)
                values = np.abs(coef[0] if coef.ndim > 1 else coef)

            if values is None or len(values) != len(feature_names):
                continue
            s = pd.Series(values, index=feature_names, dtype=float)
            denom = float(np.abs(s).sum())
            if denom <= 0:
                continue
            scores = scores.add(np.abs(s) / denom, fill_value=0.0)
            used += 1

        if used == 0:
            return scores
        return (scores / used).sort_values(ascending=False)


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    std = np.std(returns, ddof=1)
    if std <= 0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(252))


def learn_threshold(
    probabilities: np.ndarray,
    risk_returns: pd.Series,
    safe_returns: pd.Series,
    cfg: ModelConfig,
) -> float:
    grid = np.arange(cfg.threshold_grid_low, cfg.threshold_grid_high + 1e-8, cfg.threshold_grid_step)
    best = cfg.threshold_grid_low
    best_score = -1e9

    risk = np.asarray(risk_returns)
    safe = np.asarray(safe_returns)

    for th in grid:
        sig = (probabilities >= th).astype(float)
        pos = pd.Series(sig).shift(1).fillna(1.0).to_numpy()
        ret = np.where(pos >= 0.5, risk, safe)
        score = _sharpe(ret)
        turnover = np.abs(np.diff(pos)).mean() if len(pos) > 1 else 0.0

        objective = score - 0.05 * turnover
        if objective > best_score:
            best_score = objective
            best = float(th)

    return best

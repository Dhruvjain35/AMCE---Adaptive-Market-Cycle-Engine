from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression


@dataclass
class FeatureSelectionReport:
    total_features: int
    selected_features: int
    selected_list: list[str]
    method_scores: dict[str, dict[str, float]]


def _score_to_rank(score_map: dict[str, float]) -> dict[str, float]:
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    out: dict[str, float] = {}
    n = len(ranked)
    for i, (name, _) in enumerate(ranked):
        out[name] = (n - i) / max(1, n)
    return out


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 60,
    random_state: int = 42,
    max_rows: int = 1200,
    permutation_repeats: int = 3,
) -> FeatureSelectionReport:
    X = X.copy()
    y = pd.Series(y).astype(int)
    if len(X) > max_rows:
        sampled_idx = X.index[-max_rows:]
        X = X.loc[sampled_idx]
        y = y.loc[sampled_idx]

    rf = RandomForestClassifier(n_estimators=160, max_depth=6, min_samples_leaf=15, n_jobs=-1, random_state=random_state)
    rf.fit(X, y)

    # permutation importance
    perm = permutation_importance(rf, X, y, n_repeats=permutation_repeats, random_state=random_state, n_jobs=-1)
    perm_scores = {c: float(s) for c, s in zip(X.columns, perm.importances_mean)}

    # mutual information
    mi = mutual_info_classif(X, y, random_state=random_state)
    mi_scores = {c: float(s) for c, s in zip(X.columns, mi)}

    # recursive feature elimination
    lr_rfe = LogisticRegression(max_iter=2000, C=0.8, random_state=random_state)
    rfe = RFE(estimator=lr_rfe, n_features_to_select=max(10, min(top_k, X.shape[1] // 2)), step=0.2)
    rfe.fit(X, y)
    rfe_scores = {c: float(1.0 / r) for c, r in zip(X.columns, rfe.ranking_)}

    # L1 regularization
    lr_l1 = LogisticRegression(max_iter=3000, C=0.2, penalty="l1", solver="liblinear", random_state=random_state)
    lr_l1.fit(X, y)
    l1_scores = {c: float(abs(v)) for c, v in zip(X.columns, lr_l1.coef_[0])}

    rank_perm = _score_to_rank(perm_scores)
    rank_mi = _score_to_rank(mi_scores)
    rank_rfe = _score_to_rank(rfe_scores)
    rank_l1 = _score_to_rank(l1_scores)

    combined = {}
    for c in X.columns:
        combined[c] = (
            0.30 * rank_perm.get(c, 0.0)
            + 0.25 * rank_mi.get(c, 0.0)
            + 0.25 * rank_rfe.get(c, 0.0)
            + 0.20 * rank_l1.get(c, 0.0)
        )

    selected = [k for k, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    return FeatureSelectionReport(
        total_features=X.shape[1],
        selected_features=len(selected),
        selected_list=selected,
        method_scores={
            "permutation": perm_scores,
            "mutual_info": mi_scores,
            "rfe": rfe_scores,
            "l1": l1_scores,
            "combined": combined,
        },
    )

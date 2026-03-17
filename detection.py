from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def regime_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["Risk"].pct_change()
    vol_21 = ret.rolling(21).std()
    mom_21 = df["Risk"].pct_change(21)
    trend_63 = df["Risk"] / df["Risk"].rolling(63).mean() - 1.0
    vix_mom = df["VIX"].pct_change(21) if "VIX" in df.columns else 0.0
    autocorr_21 = ret.rolling(21).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)

    feats = pd.DataFrame(
        {
            "ret_1d": ret,
            "vol_21": vol_21,
            "mom_21": mom_21,
            "trend_63": trend_63,
            "vix_mom_21": vix_mom,
            "autocorr_21": autocorr_21,
        },
        index=df.index,
    )
    return feats.replace([np.inf, -np.inf], np.nan)


@dataclass
class RegimeClassifier:
    method: str
    n_states: int
    scaler: StandardScaler
    model: Any | None
    state_labels: dict[int, str]
    vol_bins: tuple[float, float] | None = None

    def _predict_states(self, features: pd.DataFrame) -> np.ndarray:
        x = self.scaler.transform(features)
        if self.method == "volatility":
            if self.vol_bins is None:
                return np.zeros(len(features), dtype=int)
            q1, q2 = self.vol_bins
            v = features["vol_21"].to_numpy()
            return np.digitize(v, bins=[q1, q2]).astype(int)
        if self.model is None:
            return np.zeros(len(features), dtype=int)
        return np.asarray(self.model.predict(x)).astype(int)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        feats = regime_feature_matrix(df).dropna()
        if feats.empty:
            return pd.Series("Unknown", index=df.index, dtype="object")

        states = self._predict_states(feats)
        labels = pd.Series([self.state_labels.get(int(s), f"State_{int(s)}") for s in states], index=feats.index, dtype="object")
        return labels.reindex(df.index).ffill().bfill().fillna("Unknown")


def _build_state_labels(states: np.ndarray, feats: pd.DataFrame) -> dict[int, str]:
    diagnostics = []
    for s in sorted(np.unique(states)):
        subset = feats.iloc[states == s]
        if len(subset) == 0:
            continue
        diagnostics.append(
            {
                "state": int(s),
                "vol": float(subset["vol_21"].mean()),
                "trend": float(subset["trend_63"].mean()),
                "autocorr": float(subset["autocorr_21"].mean()),
            }
        )

    if not diagnostics:
        return {0: "Unknown"}

    vol_values = np.array([d["vol"] for d in diagnostics])
    v25 = float(np.nanpercentile(vol_values, 25))
    v75 = float(np.nanpercentile(vol_values, 75))

    labels: dict[int, str] = {}
    for d in diagnostics:
        if d["vol"] >= v75:
            labels[d["state"]] = "HighVol"
        elif d["vol"] <= v25 and d["trend"] > 0:
            labels[d["state"]] = "LowVol"
        elif d["trend"] > 0 and d["autocorr"] > 0:
            labels[d["state"]] = "Trending"
        elif d["trend"] < 0 and d["autocorr"] < 0:
            labels[d["state"]] = "MeanReverting"
        else:
            labels[d["state"]] = "Neutral"
    return labels


def fit_regime_classifier(
    train_df: pd.DataFrame,
    method: str = "gmm",
    n_states: int = 4,
    random_state: int = 42,
) -> RegimeClassifier:
    method = method.lower().strip()
    feats = regime_feature_matrix(train_df).dropna()
    if len(feats) < 120:
        scaler = StandardScaler().fit(np.zeros((2, len(feats.columns) if len(feats.columns) else 1)))
        return RegimeClassifier(
            method="volatility",
            n_states=1,
            scaler=scaler,
            model=None,
            state_labels={0: "Neutral"},
            vol_bins=None,
        )

    scaler = StandardScaler()
    x = scaler.fit_transform(feats)

    if method == "kmeans":
        model: Any | None = KMeans(n_clusters=n_states, n_init=20, random_state=random_state)
        states = model.fit_predict(x)
        vol_bins = None
    elif method == "gmm":
        model = GaussianMixture(n_components=n_states, covariance_type="full", random_state=random_state)
        states = model.fit_predict(x)
        vol_bins = None
    elif method == "hmm":
        # Optional dependency path.
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore

            model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=300, random_state=random_state)
            model.fit(x)
            states = model.predict(x)
        except Exception:
            model = GaussianMixture(n_components=n_states, covariance_type="full", random_state=random_state)
            states = model.fit_predict(x)
        vol_bins = None
    elif method == "volatility":
        model = None
        q1 = float(feats["vol_21"].quantile(0.33))
        q2 = float(feats["vol_21"].quantile(0.67))
        vol_bins = (q1, q2)
        states = np.digitize(feats["vol_21"].to_numpy(), bins=[q1, q2]).astype(int)
    else:
        raise ValueError(f"Unsupported regime method: {method}")

    state_labels = _build_state_labels(np.asarray(states), feats)
    return RegimeClassifier(
        method=method,
        n_states=n_states,
        scaler=scaler,
        model=model,
        state_labels=state_labels,
        vol_bins=vol_bins,
    )

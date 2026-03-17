from __future__ import annotations

import numpy as np
import pandas as pd


def generate_alpha_library(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate a broad alpha library with 100+ candidate signals.
    Features are assembled in a single concat to avoid DataFrame fragmentation.
    """
    out = df.copy()
    feature_cols: list[str] = []
    new_features: dict[str, pd.Series] = {}

    r_ret = out["Risk"].pct_change()
    s_ret = out["Safe"].pct_change()
    rel_ret = r_ret - s_ret

    new_features["R_ret"] = r_ret
    new_features["S_ret"] = s_ret
    new_features["Rel_ret"] = rel_ret

    mom_windows = [3, 5, 10, 21, 42, 63, 84, 126, 252]
    vol_windows = [5, 10, 21, 42, 63]
    z_windows = [10, 21, 42, 63, 126]

    # Momentum and relative strength
    for w in mom_windows:
        c = f"mom_{w}"
        new_features[c] = out["Risk"].pct_change(w)
        feature_cols.append(c)

        c = f"safe_mom_{w}"
        new_features[c] = out["Safe"].pct_change(w)
        feature_cols.append(c)

        c = f"rel_strength_{w}"
        new_features[c] = (out["Risk"] / out["Safe"]).pct_change(w)
        feature_cols.append(c)

    # Momentum acceleration
    for s in [5, 10, 21]:
        for l in [42, 63, 126]:
            c = f"mom_accel_{s}_{l}"
            new_features[c] = out["Risk"].pct_change(s) - out["Risk"].pct_change(l)
            feature_cols.append(c)

    # Mean reversion / z-score deviations
    for w in z_windows:
        mean = out["Risk"].rolling(w).mean()
        std = out["Risk"].rolling(w).std()

        c = f"risk_z_{w}"
        new_features[c] = pd.Series(np.where(std > 0, (out["Risk"] - mean) / std, 0.0), index=out.index)
        feature_cols.append(c)

        c = f"bollinger_revert_{w}"
        new_features[c] = -new_features[f"risk_z_{w}"]
        feature_cols.append(c)

        c = f"yield_z_{w}"
        y_mean = out["Yield"].rolling(w).mean()
        y_std = out["Yield"].rolling(w).std()
        new_features[c] = pd.Series(np.where(y_std > 0, (out["Yield"] - y_mean) / y_std, 0.0), index=out.index)
        feature_cols.append(c)

    # RSI extremes
    for p in [7, 14, 21, 28]:
        delta = out["Risk"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
        rs = np.where(loss > 0, gain / loss, 100.0)
        rsi = pd.Series(100 - (100 / (1 + rs)), index=out.index)

        c = f"rsi_{p}"
        new_features[c] = rsi
        feature_cols.append(c)

        c = f"rsi_extreme_{p}"
        new_features[c] = pd.Series(np.where(rsi > 70, -1.0, np.where(rsi < 30, 1.0, 0.0)), index=out.index)
        feature_cols.append(c)

    # Volatility and ATR proxies
    for w in vol_windows:
        c = f"realized_vol_{w}"
        new_features[c] = r_ret.rolling(w).std()
        feature_cols.append(c)

        c = f"vol_cluster_{w}"
        new_features[c] = new_features[f"realized_vol_{w}"].rolling(w).mean()
        feature_cols.append(c)

        c = f"atr_proxy_{w}"
        new_features[c] = r_ret.abs().rolling(w).mean()
        feature_cols.append(c)

        c = f"vol_ratio_rs_{w}"
        s_vol = s_ret.rolling(w).std()
        new_features[c] = pd.Series(np.where(s_vol > 0, r_ret.rolling(w).std() / s_vol, np.nan), index=out.index)
        feature_cols.append(c)

    # Volume proxies (when explicit volume is unavailable)
    for w in [5, 10, 21, 42]:
        c = f"activity_spike_proxy_{w}"
        baseline = r_ret.abs().rolling(w).mean()
        new_features[c] = pd.Series(np.where(baseline > 0, r_ret.abs() / baseline, 0.0), index=out.index)
        feature_cols.append(c)

        c = f"vwap_dev_proxy_{w}"
        ma = out["Risk"].rolling(w).mean()
        new_features[c] = pd.Series(np.where(ma > 0, out["Risk"] / ma - 1, np.nan), index=out.index)
        feature_cols.append(c)

    # Technical structure
    ma_windows = [10, 20, 50, 100, 200]
    for w in ma_windows:
        c = f"ma_dist_{w}"
        ma = out["Risk"].rolling(w).mean()
        new_features[c] = pd.Series(np.where(ma > 0, out["Risk"] / ma - 1, np.nan), index=out.index)
        feature_cols.append(c)

    for s in [10, 20, 50]:
        for l in [100, 200]:
            c = f"ma_spread_{s}_{l}"
            new_features[c] = new_features[f"ma_dist_{s}"] - new_features[f"ma_dist_{l}"]
            feature_cols.append(c)

    for w in [20, 42, 63, 126]:
        c = f"breakout_pos_{w}"
        hh = out["Risk"].rolling(w).max()
        ll = out["Risk"].rolling(w).min()
        rng = hh - ll
        new_features[c] = pd.Series(np.where(rng > 0, (out["Risk"] - ll) / rng, 0.5), index=out.index)
        feature_cols.append(c)

    # Statistical signals
    for w in [21, 42, 63, 126]:
        c = f"rolling_sharpe_{w}"
        mu = r_ret.rolling(w).mean()
        sd = r_ret.rolling(w).std()
        new_features[c] = pd.Series(np.where(sd > 0, (mu / sd) * np.sqrt(252), 0.0), index=out.index)
        feature_cols.append(c)

        c = f"autocorr_{w}"
        new_features[c] = r_ret.rolling(w).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
        feature_cols.append(c)

        c = f"hurst_proxy_{w}"
        std_short = out["Risk"].diff().rolling(max(2, w // 2)).std()
        std_long = out["Risk"].diff().rolling(w).std()
        ratio = pd.Series(np.where(std_long > 0, std_short / std_long, np.nan), index=out.index)
        new_features[c] = pd.Series(np.where(ratio > 0, np.log(ratio) / np.log(0.5), np.nan), index=out.index)
        feature_cols.append(c)

    # Cross-asset and market breadth proxies
    for w in [10, 21, 42, 63, 126]:
        c = f"corr_risk_safe_{w}"
        new_features[c] = r_ret.rolling(w).corr(s_ret)
        feature_cols.append(c)

        c = f"yield_change_{w}"
        new_features[c] = out["Yield"].diff(w)
        feature_cols.append(c)

        c = f"vix_change_{w}"
        new_features[c] = out["VIX"].pct_change(w)
        feature_cols.append(c)

        c = f"beta_to_benchmark_{w}"
        cov = r_ret.rolling(w).cov(s_ret)
        var = s_ret.rolling(w).var()
        new_features[c] = pd.Series(np.where(var > 0, cov / var, np.nan), index=out.index)
        feature_cols.append(c)

        c = f"breadth_proxy_{w}"
        new_features[c] = (r_ret > 0).rolling(w).mean()
        feature_cols.append(c)

    feature_df = pd.DataFrame(new_features, index=out.index).replace([np.inf, -np.inf], np.nan)
    out = pd.concat([out, feature_df], axis=1)
    feature_cols = [c for c in feature_cols if c in feature_df.columns]
    return out, feature_cols

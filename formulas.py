from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return pd.Series(100 - (100 / (1 + rs)), index=series.index)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window)

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        coef = np.polyfit(x, values, 1)
        return coef[0]

    return series.rolling(window).apply(slope, raw=True)


def _donchian_position(series: pd.Series, window: int) -> pd.Series:
    hh = series.rolling(window).max()
    ll = series.rolling(window).min()
    rng = hh - ll
    return np.where(rng > 0, (series - ll) / rng, 0.5)


def _adx_proxy(price: pd.Series, period: int) -> pd.Series:
    diff = price.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    tr = diff.abs().rolling(period).mean()
    plus_dm = up.rolling(period).mean()
    minus_dm = down.rolling(period).mean()
    plus_di = np.divide(plus_dm, tr, out=np.zeros_like(plus_dm, dtype=float), where=tr.to_numpy() > 0)
    minus_di = np.divide(minus_dm, tr, out=np.zeros_like(minus_dm, dtype=float), where=tr.to_numpy() > 0)
    di_sum = plus_di + minus_di
    dx = np.divide(np.abs(plus_di - minus_di), di_sum, out=np.zeros_like(di_sum, dtype=float), where=di_sum > 0)
    return pd.Series(dx, index=price.index).rolling(period).mean() * 100


def _updown_vol_ratio(returns: pd.Series, window: int) -> pd.Series:
    up = np.maximum(returns, 0.0)
    down = np.minimum(returns, 0.0)
    up_semivar = np.sqrt((up**2).rolling(window).mean())
    down_semivar = np.sqrt((down**2).rolling(window).mean())
    return np.where(down_semivar > 0, up_semivar / down_semivar, np.nan)


def _drawdown(price: pd.Series, window: int) -> pd.Series:
    rolling_max = price.rolling(window).max()
    return np.where(rolling_max > 0, price / rolling_max - 1.0, np.nan)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return np.where(std > 0, (series - mean) / std, 0.0)


def _expected_shortfall(returns: pd.Series, window: int, q: float = 0.05) -> pd.Series:
    def es(values: pd.Series) -> float:
        cutoff = values.quantile(q)
        tail = values[values <= cutoff]
        return float(tail.mean()) if len(tail) else np.nan

    return returns.rolling(window).apply(lambda x: es(pd.Series(x)), raw=False)


def _rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    return series.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=False)


def compute_formula(df: pd.DataFrame, formula_id: str, params: dict) -> pd.Series:
    src = params.get("source", "Risk")
    src2 = params.get("source2", "Safe")
    window = int(params.get("window", 21))
    short = int(params.get("short", 21))
    long = int(params.get("long", 63))

    if formula_id == "pct_change":
        return df[src].pct_change(window)
    if formula_id == "ma_dist":
        ma = df[src].rolling(window).mean()
        return np.where(ma > 0, df[src] / ma - 1.0, np.nan)
    if formula_id == "ema_dist":
        ema = df[src].ewm(span=window, adjust=False).mean()
        return np.where(ema > 0, df[src] / ema - 1.0, np.nan)
    if formula_id == "rolling_slope":
        return _rolling_slope(df[src], window)
    if formula_id == "donchian_position":
        return pd.Series(_donchian_position(df[src], window), index=df.index)
    if formula_id == "adx_proxy":
        return _adx_proxy(df[src], window)
    if formula_id == "residual_mom":
        return (df["Risk"].pct_change(window) - df["Safe"].pct_change(window))
    if formula_id == "mom_accel":
        return df[src].pct_change(short) - df[src].pct_change(long)
    if formula_id == "reversal":
        return -(df[src].pct_change(short) - df[src].pct_change(long))
    if formula_id == "rsi":
        return _rsi(df[src], window)
    if formula_id == "stoch_k":
        hh = df[src].rolling(window).max()
        ll = df[src].rolling(window).min()
        denom = hh - ll
        return np.where(denom > 0, 100 * (df[src] - ll) / denom, np.nan)
    if formula_id == "stoch_d":
        k = compute_formula(df, "stoch_k", {"source": src, "window": window})
        return pd.Series(k, index=df.index).rolling(3).mean()
    if formula_id == "rolling_vol":
        return df[src].pct_change().rolling(window).std()
    if formula_id == "downside_vol":
        ret = df[src].pct_change()
        down = np.minimum(ret, 0.0)
        return np.sqrt((down**2).rolling(window).mean())
    if formula_id == "upside_vol":
        ret = df[src].pct_change()
        up = np.maximum(ret, 0.0)
        return np.sqrt((up**2).rolling(window).mean())
    if formula_id == "updown_vol_ratio":
        ret = df[src].pct_change()
        return pd.Series(_updown_vol_ratio(ret, window), index=df.index)
    if formula_id == "atr_proxy":
        return df[src].pct_change().abs().rolling(window).mean()
    if formula_id == "vol_of_vol":
        vol = df[src].pct_change().rolling(window).std()
        return vol.rolling(window).std()
    if formula_id == "risk_safe_vol_ratio":
        rvol = df["Risk"].pct_change().rolling(window).std()
        svol = df["Safe"].pct_change().rolling(window).std()
        return np.where(svol > 0, rvol / svol, np.nan)
    if formula_id == "drawdown":
        return pd.Series(_drawdown(df[src], window), index=df.index)
    if formula_id == "rolling_quantile":
        q = float(params.get("quantile", 0.05))
        return df[src].pct_change().rolling(window).quantile(q)
    if formula_id == "expected_shortfall":
        ret = df[src].pct_change()
        q = float(params.get("quantile", 0.05))
        return _expected_shortfall(ret, window, q)
    if formula_id == "rolling_skew":
        return df[src].pct_change().rolling(window).skew()
    if formula_id == "rolling_kurt":
        return df[src].pct_change().rolling(window).kurt()
    if formula_id == "level":
        return df[src]
    if formula_id == "diff":
        return df[src].diff(window)
    if formula_id == "trend_deviation":
        ma = df[src].rolling(window).mean()
        return df[src] - ma
    if formula_id == "zscore":
        return pd.Series(_zscore(df[src], window), index=df.index)
    if formula_id == "vix_term_proxy":
        ma = df[src].rolling(window).mean()
        return np.where(ma > 0, df[src] / ma - 1, np.nan)
    if formula_id == "rolling_corr":
        return df[src].pct_change().rolling(window).corr(df[src2].pct_change())
    if formula_id == "relative_strength":
        rs = df[src] / df[src2]
        return rs.pct_change(window)
    if formula_id == "dispersion_proxy":
        r = df["Risk"].pct_change()
        s = df["Safe"].pct_change()
        return pd.concat([r, s], axis=1).rolling(window).std().mean(axis=1)
    if formula_id == "positive_day_ratio":
        r = (df[src].pct_change() > 0).astype(float)
        return r.rolling(window).mean()
    if formula_id == "carry_proxy":
        ret_safe = df["Safe"].pct_change().rolling(window).mean() * 252
        return df["Yield"] / 100 - ret_safe
    if formula_id == "cash_vs_safe_adv":
        cash = (df["Yield"] / 100) / 252
        safe = df["Safe"].pct_change()
        return (cash - safe).rolling(window).mean()
    if formula_id == "defensive_spread":
        return (df["Safe"].pct_change(window) - df["Risk"].pct_change(window))
    if formula_id == "risk_off_pressure":
        vix_z = pd.Series(_zscore(df["VIX"], window), index=df.index)
        trend = pd.Series(np.where(df["Risk"].rolling(window).mean() > 0, df["Risk"] / df["Risk"].rolling(window).mean() - 1, np.nan), index=df.index)
        return vix_z - trend
    if formula_id == "yield_vol_interaction":
        ychg = df["Yield"].diff(window)
        vol = df["Risk"].pct_change().rolling(window).std()
        return ychg * vol
    if formula_id == "flight_to_quality":
        return (df["Safe"].pct_change() - df["Risk"].pct_change()).rolling(window).mean()
    if formula_id == "hedge_demand":
        safe_mom = df["Safe"].pct_change(window)
        vix_chg = df["VIX"].pct_change(window)
        return safe_mom + vix_chg
    if formula_id == "rolling_autocorr":
        return _rolling_autocorr(df[src].pct_change(), window, lag=int(params.get("lag", 1)))

    raise ValueError(f"Unknown formula_id: {formula_id}")

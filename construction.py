from __future__ import annotations

import numpy as np
import pandas as pd

from schema import RiskConfig


def apply_portfolio_constraints(
    weights: pd.Series,
    max_position_size: float = 0.30,
    leverage_limit: float = 1.0,
) -> pd.Series:
    out = weights.astype(float).clip(lower=0.0)
    target_sum = float(max(0.0, leverage_limit))
    if target_sum <= 0:
        return out * 0.0
    if out.sum() <= 0:
        out[:] = 1.0 / len(out)

    out = out / out.sum() * target_sum

    for _ in range(20):
        over = out > max_position_size
        if not over.any():
            break
        excess = float((out[over] - max_position_size).sum())
        out.loc[over] = max_position_size

        under = out < max_position_size - 1e-12
        if not under.any() or excess <= 0:
            break
        room = (max_position_size - out[under]).clip(lower=0.0)
        room_sum = float(room.sum())
        if room_sum <= 0:
            break
        out.loc[under] = out[under] + room / room_sum * excess

    out = out.clip(lower=0.0, upper=max_position_size)
    rem = target_sum - float(out.sum())
    if rem > 1e-12:
        under = out < max_position_size - 1e-12
        room = (max_position_size - out[under]).clip(lower=0.0)
        room_sum = float(room.sum())
        if room_sum > 0:
            out.loc[under] = out[under] + room / room_sum * rem

    gross = float(out.abs().sum())
    if gross > target_sum and gross > 0:
        out = out * (target_sum / gross)
    return out


def mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 4.0,
    max_position_size: float = 0.35,
    leverage_limit: float = 1.0,
) -> pd.Series:
    mu = expected_returns.astype(float)
    cov = covariance.loc[mu.index, mu.index].astype(float)
    inv_cov = np.linalg.pinv(cov.to_numpy())
    raw = pd.Series(inv_cov @ mu.to_numpy(), index=mu.index)
    raw = raw / max(risk_aversion, 1e-8)
    if raw.sum() <= 0:
        raw = pd.Series(1.0 / len(mu), index=mu.index)
    else:
        raw = raw / raw.sum()
    return apply_portfolio_constraints(raw, max_position_size=max_position_size, leverage_limit=leverage_limit)


def risk_parity_weights(
    covariance: pd.DataFrame,
    max_position_size: float = 0.35,
    leverage_limit: float = 1.0,
    iterations: int = 200,
) -> pd.Series:
    assets = covariance.columns
    cov = covariance.to_numpy(dtype=float)
    n = len(assets)
    w = np.full(n, 1.0 / n, dtype=float)

    for _ in range(iterations):
        port_var = float(w.T @ cov @ w)
        if port_var <= 0:
            break
        mrc = cov @ w
        rc = w * mrc / np.sqrt(port_var)
        target = np.full(n, rc.mean())
        step = 0.12
        w = w * np.exp(-step * (rc - target))
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()

    out = pd.Series(w, index=assets)
    return apply_portfolio_constraints(out, max_position_size=max_position_size, leverage_limit=leverage_limit)


def signal_weighted_allocation(
    signals: pd.Series,
    volatility: pd.Series | None = None,
    max_position_size: float = 0.35,
    leverage_limit: float = 1.0,
) -> pd.Series:
    s = signals.astype(float).clip(lower=0.0)
    if volatility is not None:
        v = volatility.reindex(s.index).astype(float)
        inv_v = np.where(v > 0, 1.0 / v, 0.0)
        s = s * pd.Series(inv_v, index=s.index)
    if s.sum() <= 0:
        s = pd.Series(1.0 / len(s), index=s.index)
    else:
        s = s / s.sum()
    return apply_portfolio_constraints(s, max_position_size=max_position_size, leverage_limit=leverage_limit)


def fractional_kelly_allocation(
    edge: pd.Series,
    variance: pd.Series,
    fraction: float = 0.25,
    max_leverage: float = 1.0,
    max_position_size: float = 0.35,
) -> pd.Series:
    edge = edge.astype(float)
    variance = variance.reindex(edge.index).astype(float)
    raw = pd.Series(np.where(variance > 1e-10, edge / variance, 0.0), index=edge.index)
    raw = raw.clip(lower=0.0) * max(0.0, fraction)
    gross = float(raw.sum())
    if gross > max_leverage and gross > 0:
        raw = raw * (max_leverage / gross)
    if raw.sum() <= 0:
        raw = pd.Series(1.0 / len(raw), index=raw.index)
    else:
        raw = raw / raw.sum()
    return apply_portfolio_constraints(raw, max_position_size=max_position_size, leverage_limit=max_leverage)


def probabilities_to_exposure(
    probabilities: np.ndarray,
    risk_returns: pd.Series,
    defensive_returns: pd.Series | None,
    cfg: RiskConfig,
    threshold: float,
) -> pd.Series:
    probs = np.asarray(probabilities)
    idx = risk_returns.index

    if cfg.threshold_band <= 0:
        raw = (probs >= threshold).astype(float)
    else:
        lo = threshold - cfg.threshold_band
        hi = threshold + cfg.threshold_band
        raw = np.clip((probs - lo) / max(hi - lo, 1e-8), 0.0, 1.0)

    ann_vol = risk_returns.rolling(21).std() * np.sqrt(252)
    vol_scale = np.where(ann_vol > 0, cfg.target_annual_vol / ann_vol, 1.0)
    vol_scale = np.clip(vol_scale, 0.0, 2.0)
    target = np.clip(raw * vol_scale, 0.0, cfg.max_exposure)

    if defensive_returns is None:
        defensive_returns = pd.Series(0.0, index=idx)

    out = np.zeros(len(target))
    out[0] = float(target[0]) if len(target) else 0.0

    equity = 1.0
    peak = 1.0
    rolling_var = risk_returns.rolling(cfg.tail_var_window).quantile(0.01)
    daily_vol = risk_returns.rolling(21).std()

    for i in range(1, len(target)):
        desired = float(target[i])
        prev = out[i - 1]
        delta = desired - prev
        bounded_delta = float(np.clip(delta, -cfg.max_daily_turnover, cfg.max_daily_turnover))
        current = prev + bounded_delta

        drawdown = equity / peak - 1.0
        if drawdown <= cfg.drawdown_de_risk_at:
            current *= cfg.drawdown_de_risk_factor

        # Volatility-based stop loss and tail risk overlay.
        sigma = float(daily_vol.iloc[i]) if i < len(daily_vol) else 0.0
        stop_level = -cfg.stop_loss_sigma * sigma if sigma > 0 else -np.inf
        if float(risk_returns.iloc[i]) <= stop_level:
            current *= 0.5

        tail_var = float(rolling_var.iloc[i]) if i < len(rolling_var) else 0.0
        if not np.isnan(tail_var) and tail_var < cfg.tail_var_floor:
            ratio = abs(cfg.tail_var_floor) / max(abs(tail_var), 1e-8)
            scale = float(np.clip(ratio, cfg.tail_risk_min_scale, 1.0))
            current *= scale

        current = float(np.clip(current, 0.0, cfg.max_exposure))
        out[i] = current

        gross_ret = current * float(risk_returns.iloc[i]) + (1 - current) * float(defensive_returns.iloc[i])
        equity *= 1 + gross_ret
        peak = max(peak, equity)

    return pd.Series(out, index=idx, name="Exposure")

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=1)
    return np.where(sd > 0, (mu / sd) * np.sqrt(252), np.nan)


def _alpha_beta(strategy: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    x = benchmark.reindex(strategy.index).dropna()
    y = strategy.reindex(x.index).dropna()
    x = x.reindex(y.index)
    if len(y) < 30:
        return 0.0, 0.0
    x_var = float(x.var(ddof=1))
    if x_var <= 0:
        return 0.0, 0.0
    cov = float(np.cov(y.to_numpy(), x.to_numpy(), ddof=1)[0, 1])
    beta = cov / x_var
    alpha_daily = float(y.mean()) - beta * float(x.mean())
    return alpha_daily * 252, beta


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
) -> dict[str, float]:
    r = returns.dropna()
    if len(r) == 0:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "calmar": 0.0,
            "information_ratio": 0.0,
            "win_rate": 0.0,
            "turnover_annual": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
        }

    mean_d = float(r.mean())
    std_d = float(r.std(ddof=1))
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else 0.0

    downside = r[r < 0]
    downside_std = float(np.sqrt((downside**2).mean())) if len(downside) else 0.0
    sortino = (mean_d * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())

    total_return = float(eq.iloc[-1] - 1)
    ann_return = float((1 + mean_d) ** 252 - 1)
    cagr = float(eq.iloc[-1] ** (252 / max(len(r), 1)) - 1)
    vol = float(std_d * np.sqrt(252))
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0
    win_rate = float((r > 0).mean())
    turnover_annual = float(turnover.dropna().mean() * 252) if turnover is not None and len(turnover.dropna()) else 0.0

    information_ratio = 0.0
    alpha = 0.0
    beta = 0.0
    if benchmark_returns is not None:
        b = benchmark_returns.reindex(r.index).dropna()
        aligned = r.reindex(b.index).dropna()
        b = b.reindex(aligned.index)
        if len(aligned) > 1:
            active = aligned - b
            active_std = float(active.std(ddof=1))
            information_ratio = float((active.mean() / active_std) * np.sqrt(252)) if active_std > 0 else 0.0
            alpha, beta = _alpha_beta(aligned, b)

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "total_return": total_return,
        "annual_return": ann_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "volatility": vol,
        "calmar": calmar,
        "information_ratio": information_ratio,
        "win_rate": win_rate,
        "turnover_annual": turnover_annual,
        "alpha": alpha,
        "beta": beta,
    }


def compute_tercile_metrics(returns: pd.Series) -> list[dict[str, float]]:
    r = returns.dropna()
    if len(r) < 30:
        return []

    size = len(r) // 3
    slices = [r.iloc[:size], r.iloc[size : 2 * size], r.iloc[2 * size :]]
    out = []
    for i, sub in enumerate(slices, start=1):
        m = compute_metrics(sub)
        m["tercile"] = float(i)
        out.append(m)
    return out

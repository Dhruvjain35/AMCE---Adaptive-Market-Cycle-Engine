from __future__ import annotations

import numpy as np
import pandas as pd

from schema import BacktestConfig
from amce_types import BacktestResult


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


# ---------------------------------------------------------------------------
# Backtest engine (originally in amce/backtest/engine.py)
# ---------------------------------------------------------------------------

def _rebalance_mask(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    if freq == "daily":
        return pd.Series(True, index=index)
    if freq == "weekly":
        return pd.Series(index.weekday == 0, index=index)
    if freq == "monthly":
        month = pd.Series(index.month, index=index)
        return month != month.shift(1)
    raise ValueError(f"Unsupported rebalance frequency: {freq}")


def _apply_rebalance(exposure: pd.Series, freq: str) -> pd.Series:
    allowed = _rebalance_mask(exposure.index, freq)
    out = exposure.copy()
    out.loc[~allowed] = np.nan
    return out.ffill().fillna(0.0)


def run_backtest(df: pd.DataFrame, exposure: pd.Series, cfg: BacktestConfig) -> BacktestResult:
    out = df.copy()
    out["R_ret"] = out["Risk"].pct_change().fillna(0.0)
    out["S_ret"] = out["Safe"].pct_change().fillna(0.0)
    out["Cash_ret"] = (out["Yield"] / 100.0 / 252.0).fillna(0.0)

    if "Yield_Trend_63" in out.columns:
        yield_trend = out["Yield_Trend_63"] > 0
    else:
        yield_trend = out["Yield"] > out["Yield"].rolling(63).mean()

    defensive_ret = np.where(yield_trend, out["Cash_ret"], out["S_ret"])
    out["Defensive_ret"] = defensive_ret

    desired = exposure.reindex(out.index).ffill().fillna(0.0)
    lagged = desired.shift(cfg.execution_lag_days).fillna(0.0)
    out["Exposure"] = _apply_rebalance(lagged, cfg.rebalance_frequency)

    out["Gross"] = out["Exposure"] * out["R_ret"] + (1 - out["Exposure"]) * out["Defensive_ret"]

    out["Turnover"] = out["Exposure"].diff().abs().fillna(0.0)
    if "Volume" in out.columns:
        adv = out["Volume"].rolling(cfg.liquidity_lookback_days).mean()
        rel_adv = np.where(adv > 0, out["Volume"] / adv, 1.0)
        cap = np.clip(cfg.max_adv_participation * rel_adv, 0.01, 1.0)
        if cfg.allow_partial_fills:
            out["Turnover"] = np.minimum(out["Turnover"], cap)
        else:
            out["Turnover"] = np.where(out["Turnover"] <= cap, out["Turnover"], 0.0)

    friction = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10000.0
    out["Cost"] = out["Turnover"] * friction
    out["Net"] = out["Gross"] - out["Cost"]

    out["Benchmark_ret"] = out["R_ret"]
    if cfg.benchmark_apply_costs:
        out["Benchmark_ret"] = out["Benchmark_ret"] - out["Turnover"] * friction

    out["Eq_Strat"] = (1 + out["Net"]).cumprod()
    out["Eq_Bench"] = (1 + out["Benchmark_ret"]).cumprod()
    out["DD_Strat"] = out["Eq_Strat"] / out["Eq_Strat"].cummax() - 1
    out["DD_Bench"] = out["Eq_Bench"] / out["Eq_Bench"].cummax() - 1

    return BacktestResult(
        frame=out,
        metrics=compute_metrics(out["Net"], benchmark_returns=out["Benchmark_ret"], turnover=out["Turnover"]),
        benchmark_metrics=compute_metrics(out["Benchmark_ret"]),
    )

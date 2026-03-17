"""Legacy compatibility wrapper for backtesting and stats."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from amce.backtest import compute_metrics, run_backtest as run_backtest_engine
from amce.config.schema import BacktestConfig



def backtest(data: pd.DataFrame, cost_bps: float, slip_bps: float) -> pd.DataFrame:
    exposure = data.get("Signal", pd.Series(0.0, index=data.index)).astype(float)
    cfg = BacktestConfig(
        transaction_cost_bps=cost_bps,
        slippage_bps=slip_bps,
        execution_lag_days=1,
        rebalance_frequency="daily",
    )
    result = run_backtest_engine(data, exposure, cfg)
    return result.frame



def compute_stats(rets: pd.Series) -> Tuple[float, float, float, float, float]:
    m = compute_metrics(rets)
    return m["sharpe"], m["sortino"], m["total_return"], m["annual_return"], m["max_drawdown"]



def compute_drawdown_durations(equity_series: pd.Series) -> dict:
    underwater = equity_series < equity_series.cummax()
    changes = underwater.astype(int).diff()

    starts = changes[changes == 1].index.tolist()
    ends = changes[changes == -1].index.tolist()

    if not starts:
        return {"max_duration": 0, "avg_duration": 0, "num_drawdowns": 0}
    if len(ends) < len(starts):
        ends.append(equity_series.index[-1])

    durations = [(e - s).days for s, e in zip(starts[: len(ends)], ends)]
    if not durations:
        return {"max_duration": 0, "avg_duration": 0, "num_drawdowns": 0}

    return {
        "max_duration": max(durations),
        "avg_duration": float(np.mean(durations)),
        "num_drawdowns": len(durations),
    }

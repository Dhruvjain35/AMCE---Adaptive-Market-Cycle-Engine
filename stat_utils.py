"""Legacy compatibility wrapper for statistical utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from stats import bootstrap_alpha_ci, permutation_test_sharpe



def permutation_test(
    pos: np.ndarray,
    risk_ret: np.ndarray,
    safe_ret: np.ndarray,
    actual_sharpe: float,
    n_perms: int = 1000,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    return permutation_test_sharpe(pos.astype(float), risk_ret, safe_ret, actual_sharpe, n_trials=n_perms, seed=seed)



def monte_carlo_simulation(
    returns: np.ndarray,
    n_sims: int = 500,
    benchmark_final: float = 1.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sims = rng.choice(returns, size=(n_sims, len(returns)), replace=True)
    sims_cum = np.cumprod(1 + sims, axis=1)

    p5 = np.percentile(sims_cum, 5, axis=0)
    p50 = np.percentile(sims_cum, 50, axis=0)
    p95 = np.percentile(sims_cum, 95, axis=0)

    prob_beat = (sims_cum[:, -1] > benchmark_final).mean() * 100
    running_max = np.maximum.accumulate(sims_cum, axis=1)
    drawdowns = sims_cum / running_max - 1
    prob_severe_dd = (drawdowns.min(axis=1) < -0.40).mean() * 100

    return {
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "prob_beat": prob_beat,
        "prob_severe_dd": prob_severe_dd,
        "sims_cum": sims_cum,
    }


__all__ = ["permutation_test", "bootstrap_alpha_ci", "monte_carlo_simulation"]

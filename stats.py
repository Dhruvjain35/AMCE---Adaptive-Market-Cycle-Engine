from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def permutation_test_sharpe(
    exposure: np.ndarray,
    risk_ret: np.ndarray,
    defensive_ret: np.ndarray,
    actual_sharpe: float,
    n_trials: int = 1000,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    sampled = np.empty(n_trials)

    for i in range(n_trials):
        shuffled = rng.permutation(exposure)
        ret = shuffled * risk_ret + (1 - shuffled) * defensive_ret
        std = np.std(ret, ddof=1)
        sampled[i] = (np.mean(ret) / std) * np.sqrt(252) if std > 0 else 0.0

    p_val = float((sampled >= actual_sharpe).mean())
    return p_val, sampled


def bootstrap_alpha_ci(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_boot: int = 500,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    y = strategy_returns.dropna()
    x = benchmark_returns.reindex(y.index).dropna()
    idx = y.index.intersection(x.index)
    y = y.loc[idx]
    x = x.loc[idx]

    if len(y) < 30:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0, "p_alpha": 1.0, "ci_low": 0.0, "ci_high": 0.0}

    xreg = sm.add_constant(x)
    model = sm.OLS(y, xreg).fit()

    alpha = float(model.params["const"] * 252)
    beta = float(model.params.iloc[1]) if len(model.params) > 1 else 0.0
    rsq = float(model.rsquared)
    p_alpha = float(model.pvalues["const"])

    rng = np.random.default_rng(seed)
    block = 21
    n = len(y)
    n_blocks = max(1, n // block)
    samples = []

    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block), size=n_blocks)
        idxs = np.concatenate([np.arange(s, min(n, s + block)) for s in starts])[:n]

        yb = y.iloc[idxs].reset_index(drop=True)
        xb = xreg.iloc[idxs].reset_index(drop=True)
        try:
            m = sm.OLS(yb, xb).fit()
            samples.append(float(m.params["const"] * 252))
        except Exception:
            continue

    if not samples:
        ci_low = ci_high = alpha
    else:
        tail = (1 - ci_level) / 2
        ci_low = float(np.percentile(samples, tail * 100))
        ci_high = float(np.percentile(samples, (1 - tail) * 100))

    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": rsq,
        "p_alpha": p_alpha,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def bootstrap_superiority_probability(
    strategy_returns: pd.Series,
    peer_returns: pd.Series,
    n_boot: int = 300,
    block_size: int = 21,
    seed: int = 42,
) -> float:
    """
    Estimate P(Sharpe_strategy > Sharpe_peer) using block bootstrap.
    """
    s = strategy_returns.dropna()
    p = peer_returns.reindex(s.index).dropna()
    idx = s.index.intersection(p.index)
    s = s.loc[idx]
    p = p.loc[idx]

    if len(s) < 60:
        return 0.0

    rng = np.random.default_rng(seed)
    n = len(s)
    wins = 0
    valid = 0
    n_blocks = max(1, n // block_size)

    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block_size), size=n_blocks)
        sample_idx = np.concatenate([np.arange(st, min(n, st + block_size)) for st in starts])[:n]
        ss = s.iloc[sample_idx].to_numpy()
        pp = p.iloc[sample_idx].to_numpy()

        s_std = np.std(ss, ddof=1)
        p_std = np.std(pp, ddof=1)
        if s_std <= 0 or p_std <= 0:
            continue
        sharpe_s = (np.mean(ss) / s_std) * np.sqrt(252)
        sharpe_p = (np.mean(pp) / p_std) * np.sqrt(252)
        wins += sharpe_s > sharpe_p
        valid += 1

    if valid == 0:
        return 0.0
    return float(wins / valid)

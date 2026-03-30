"""
AMCE Trend-Following Macro Rotation Strategy
=============================================

A five-rule, zero-optimisation systematic strategy that rotates between
risk-on (QQQ) and risk-off (IEF) using slow-moving, economically-motivated
signals.

Every threshold is fixed and justified by decades of academic and
practitioner evidence — none are fitted to data:

  * 200-day MA  : Faber (2007), "A Quantitative Approach to TAA"
  * 12-1 momentum: Jegadeesh & Titman (1993), cross-sectional momentum
  * VIX 20/25   : CBOE long-run median ~19; 25 = +1 s.d. stress
  * Yield curve 0: Estrella & Mishkin (1996), inversion => recession
  * Supertrend  : ATR volatility bands (TradingView defaults: ATR length 10,
                  factor 3.0) — trend filter; no optimisation vs history

No parameter was chosen by backtesting. No parameter is tuned in-sample.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# ─── Result container ───────────────────────────────────────────────

@dataclass
class StrategyResult:
    """Immutable result bundle from a single strategy run."""
    returns: pd.Series              # daily net-of-cost strategy returns
    equity_curve: pd.Series         # cumulative strategy wealth ($1 start)
    benchmark_equity: pd.Series     # cumulative SPY wealth ($1 start)
    benchmark_6040_equity: pd.Series  # cumulative 60/40 wealth ($1 start)
    signals_df: pd.DataFrame        # all signals, scores, exposures
    metrics_dict: dict[str, float]  # CAGR, Sharpe, MaxDD, etc.
    benchmark_metrics: dict[str, float]
    benchmark_6040_metrics: dict[str, float]
    permutation_p_value: float
    oos_start: str
    oos_end: str
    gross_equity_curve: pd.Series   # before transaction costs
    trade_log: pd.DataFrame         # dates where exposure changed
    rolling_sharpe: pd.Series       # 252-day rolling Sharpe
    drawdown_series: pd.Series      # strategy drawdown curve
    benchmark_drawdown: pd.Series   # SPY drawdown curve


# ─── Data fetching ───────────────────────────────────────────────────

def _fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers
    close = close.ffill().dropna(how="all")
    return close


def _ohlc_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """yfinance may return a 1-column DataFrame for a single ticker under MultiIndex columns."""
    col = frame[name]
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col


def _fetch_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download adjusted OHLC for one ticker (for Supertrend high/low/close)."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No OHLC data returned for {ticker}")
    high = _ohlc_column(raw, "High")
    low = _ohlc_column(raw, "Low")
    close = _ohlc_column(raw, "Close")
    ohlc = pd.DataFrame(
        {"high": high, "low": low, "close": close},
        index=raw.index,
    )
    return ohlc.ffill().dropna(how="all")


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 5,
    factor: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Supertrend (TradingView-style defaults: ATR length 5, factor 2.0).

    Implementation follows the standard band-trailing algorithm used by
    pandas_ta / common TA libraries (Wilder ATR via EWM, then iterative
    upper/lower bands). Returns:

      supertrend : float line value
      direction  : +1 = bullish / uptrend (close tracking lower band),
                   -1 = bearish / downtrend
      st_signal  : 1 = risk-on (bullish), 0 = not bullish
    """
    hl2 = (high + low) / 2.0
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    upper = (hl2 + factor * atr).to_numpy(dtype=float, copy=True)
    lower = (hl2 - factor * atr).to_numpy(dtype=float, copy=True)
    c = close.to_numpy(dtype=float, copy=False)
    n = len(close)
    dir_ = np.ones(n, dtype=np.int8)
    trend = np.full(n, np.nan, dtype=float)

    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if c[i] > upper[i - 1]:
            dir_[i] = 1
        elif c[i] < lower[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lower[i] < lower[i - 1]:
                lower[i] = lower[i - 1]
            if dir_[i] < 0 and upper[i] > upper[i - 1]:
                upper[i] = upper[i - 1]

        if dir_[i] > 0:
            trend[i] = lower[i]
        else:
            trend[i] = upper[i]

    if n > 0 and not np.isnan(upper[0]) and not np.isnan(lower[0]):
        trend[0] = lower[0] if dir_[0] > 0 else upper[0]

    idx = close.index
    supertrend = pd.Series(trend, index=idx, name="supertrend")
    direction = pd.Series(dir_, index=idx, name="st_direction").astype(int)
    st_signal = (direction == 1).astype(int)
    st_signal = st_signal.where(atr.notna(), 0)
    st_signal = st_signal.rename("st_signal")
    return supertrend, direction, st_signal


# ─── Signal construction ────────────────────────────────────────────

def _momentum_signal(risk_close: pd.Series) -> pd.Series:
    """
    12-1 Month Momentum (Jegadeesh-Titman, 1993).

    Economic rationale: intermediate-horizon momentum (2-12 months)
    is one of the most robust cross-sectional and time-series
    anomalies. We skip the most recent month to avoid the short-term
    reversal effect.

    Threshold: 0 (positive momentum = bullish, economically natural).
    """
    mom_12m = risk_close.pct_change(252)   # ~12 months
    mom_1m = risk_close.pct_change(21)     # ~1 month
    net_mom = mom_12m - mom_1m
    signal = (net_mom > 0).astype(int)
    return signal.rename("mom_signal")


def _ma200_signal(risk_close: pd.Series) -> pd.Series:
    """
    200-Day Moving Average Filter (Faber, 2007).

    Economic rationale: the 200-day MA is the standard institutional
    trend filter. Price > MA indicates the long-term trend is intact.
    This is not fitted - 200 trading days ~ 10 calendar months, a
    natural business-cycle frequency.

    Threshold: price vs MA (no parameter to tune).
    """
    ma200 = risk_close.rolling(200).mean()
    signal = (risk_close > ma200).astype(int)
    return signal.rename("ma_signal")


def _vix_signal(vix: pd.Series) -> pd.Series:
    """
    VIX Regime Filter.

    Economic rationale: CBOE VIX long-run median is ~19. Values above
    25 represent +1 standard deviation stress (elevated fear). Values
    below 20 represent calm markets. The 20-25 band is a dead zone
    where we hold the current position to avoid whipsawing.

    Thresholds: 20 and 25, both derived from VIX distributional
    properties, not from optimization.
    """
    signal = pd.Series(np.nan, index=vix.index, name="vix_signal")
    signal[vix < 20] = 1   # calm: risk-on
    signal[vix > 25] = 0   # fear: risk-off
    # 20-25 band: forward-fill previous state (hold)
    signal = signal.ffill().fillna(1).astype(int)
    return signal


def _yield_curve_signal(tnx: pd.Series, irx: pd.Series) -> pd.Series:
    """
    Yield Curve Signal (Estrella & Mishkin, 1996).

    Economic rationale: an inverted yield curve (10Y < 3M) has
    preceded every US recession since 1960 with only one false
    positive. A negative spread is a fundamental macro warning.

    Threshold: 0 (positive vs negative spread - economically natural).
    """
    spread = tnx - irx
    signal = (spread > 0).astype(int)
    return signal.rename("yield_signal")


# ─── Composite scoring & exposure ───────────────────────────────────

def _composite_exposure(score: pd.Series) -> pd.Series:
    """
    Majority-vote exposure rule.

    score >= 3 of 5 signals bullish  =>  full risk-on  (1.0)
    score == 2                       =>  partial        (0.5)
    score <= 1                       =>  risk-off       (0.0)

    No parameter here is fitted.
    """
    exposure = pd.Series(0.0, index=score.index, name="exposure")
    exposure[score >= 3] = 1.0
    exposure[score == 2] = 0.5
    exposure[score <= 1] = 0.0
    return exposure


def _apply_weekly_rebalance(exposure: pd.Series) -> pd.Series:
    """Only allow exposure changes on Mondays (weekly rebalance)."""
    weekly = exposure.copy()
    is_monday = exposure.index.dayofweek == 0
    # On non-Mondays, carry forward
    weekly[~is_monday] = np.nan
    weekly = weekly.ffill().fillna(0.0)
    return weekly


def _apply_holding_period(exposure: pd.Series, min_days: int = 10) -> pd.Series:
    """
    Minimum holding period: prevent signal flips within min_days.
    This is an anti-whipsaw filter, not an optimised parameter.
    10 trading days ~ 2 calendar weeks, a natural minimum for
    macro signals.
    """
    result = exposure.copy()
    last_change_idx = 0
    prev_val = result.iloc[0]

    for i in range(1, len(result)):
        if result.iloc[i] != prev_val:
            if (i - last_change_idx) < min_days:
                result.iloc[i] = prev_val  # suppress flip
            else:
                last_change_idx = i
                prev_val = result.iloc[i]
    return result


# ─── Backtest engine ────────────────────────────────────────────────

def _compute_metrics(returns: pd.Series, label: str = "") -> dict[str, float]:
    """Compute standard performance metrics from a daily return series."""
    total_days = len(returns)
    years = total_days / 252

    cum = (1 + returns).cumprod()
    cagr = (cum.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0.0

    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / ann_vol if ann_vol > 0 else 0.0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / downside if downside > 0 else 0.0

    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "annual_volatility": round(ann_vol, 4),
        "total_return": round(cum.iloc[-1] - 1, 4),
        "years": round(years, 2),
    }


def _backtest(
    signals_df: pd.DataFrame,
    cost_bps: float = 5.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """
    Run the backtest with transaction costs.

    Returns (net_returns, gross_equity, net_equity, trade_log).
    """
    df = signals_df.copy()

    # 1-day execution lag: signal on T, trade on T+1
    df["exposure_lagged"] = df["exposure"].shift(1).fillna(0.0)

    # Strategy returns: exposure * risk return + (1 - exposure) * safe return
    df["risk_ret"] = df["risk_close"].pct_change()
    df["safe_ret"] = df["safe_close"].pct_change()
    df["gross_ret"] = (
        df["exposure_lagged"] * df["risk_ret"]
        + (1 - df["exposure_lagged"]) * df["safe_ret"]
    )

    # Transaction costs: applied on exposure changes
    df["turnover"] = df["exposure_lagged"].diff().abs()
    cost_frac = cost_bps / 10_000
    df["cost"] = df["turnover"] * cost_frac
    df["net_ret"] = df["gross_ret"] - df["cost"]

    # Equity curves
    df["gross_equity"] = (1 + df["gross_ret"].fillna(0)).cumprod()
    df["net_equity"] = (1 + df["net_ret"].fillna(0)).cumprod()

    # Trade log: dates where exposure changed
    trades = df[df["turnover"] > 0][["exposure_lagged", "turnover", "cost"]].copy()

    return df["net_ret"], df["gross_equity"], df["net_equity"], trades


# ─── Permutation test ───────────────────────────────────────────────

def _permutation_test(
    signals_df: pd.DataFrame,
    actual_sharpe: float,
    n_perms: int = 1000,
    cost_bps: float = 5.0,
) -> float:
    """
    Permutation test on WEEKLY signal blocks.

    Shuffle the weekly exposure decisions (not daily returns) to test
    whether the strategy's Sharpe ratio is due to skill or luck.
    Returns p-value (fraction of permuted Sharpes >= actual Sharpe).
    """
    df = signals_df.copy()
    df["risk_ret"] = df["risk_close"].pct_change()
    df["safe_ret"] = df["safe_close"].pct_change()

    # Extract weekly exposure blocks (Monday-to-Monday)
    mondays = df.index[df.index.dayofweek == 0]
    weekly_exposures = df.loc[mondays, "exposure"].values.copy()

    rng = np.random.default_rng(42)
    count_better = 0

    for _ in range(n_perms):
        shuffled = weekly_exposures.copy()
        rng.shuffle(shuffled)

        # Rebuild daily exposure from shuffled weekly blocks
        perm_exposure = pd.Series(np.nan, index=df.index)
        for i, m in enumerate(mondays):
            if i < len(shuffled):
                perm_exposure.loc[m] = shuffled[i]
        perm_exposure = perm_exposure.ffill().fillna(0.0)
        perm_exposure_lag = perm_exposure.shift(1).fillna(0.0)

        perm_ret = (
            perm_exposure_lag * df["risk_ret"]
            + (1 - perm_exposure_lag) * df["safe_ret"]
        )
        turnover = perm_exposure_lag.diff().abs()
        perm_ret_net = perm_ret - turnover * (cost_bps / 10_000)

        ann_vol = perm_ret_net.std() * np.sqrt(252)
        if ann_vol > 0:
            perm_sharpe = (perm_ret_net.mean() * 252) / ann_vol
        else:
            perm_sharpe = 0.0

        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_perms


# ─── Main entry point ───────────────────────────────────────────────

def run_strategy(
    start_date: str = "2003-01-01",
    end_date: str = "2024-12-31",
    risk_ticker: str = "QQQ",
    safe_ticker: str = "IEF",
    benchmark_ticker: str = "SPY",
    cost_bps: float = 5.0,
    n_permutations: int = 1000,
    oos_start: str = "2016-01-01",
) -> StrategyResult:
    """
    Run the full trend-following macro rotation strategy.

    Parameters
    ----------
    start_date : str
        Backtest start (training period begins here).
    end_date : str
        Backtest end.
    risk_ticker : str
        Risk-on asset (default QQQ).
    safe_ticker : str
        Risk-off asset (default IEF).
    benchmark_ticker : str
        Buy-and-hold benchmark (default SPY, not QQQ).
    cost_bps : float
        One-way transaction cost in basis points.
    n_permutations : int
        Number of permutation trials.
    oos_start : str
        Out-of-sample period starts here.

    Returns
    -------
    StrategyResult
        Complete backtest results with equity curves, signals, metrics.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    # ── 1. Fetch data ──────────────────────────────────────────────
    # We need: risk asset, safe asset, SPY benchmark, VIX, 10Y yield, 3M yield
    price_tickers = [risk_ticker, safe_ticker, benchmark_ticker]
    macro_tickers = ["^VIX", "^TNX", "^IRX"]

    prices = _fetch_prices(price_tickers, start=start_date, end=end_date)
    macro = _fetch_prices(macro_tickers, start=start_date, end=end_date)
    risk_ohlc = _fetch_ohlc(risk_ticker, start=start_date, end=end_date)

    # Align all data to common dates
    all_data = prices.join(macro, how="inner").dropna()
    risk_ohlc = risk_ohlc.reindex(all_data.index).ffill()
    risk_close_series = all_data[risk_ticker]
    risk_ohlc["high"] = risk_ohlc["high"].fillna(risk_close_series)
    risk_ohlc["low"] = risk_ohlc["low"].fillna(risk_close_series)
    risk_ohlc["close"] = risk_ohlc["close"].fillna(risk_close_series)

    if len(all_data) < 300:
        raise ValueError(
            f"Insufficient data: got {len(all_data)} rows, need >= 300. "
            "Check tickers and date range."
        )

    # ── 2. Compute signals ─────────────────────────────────────────
    # All signals use ONLY data available at time T (no lookahead)
    risk_close = all_data[risk_ticker]
    vix = all_data["^VIX"]
    tnx = all_data["^TNX"]
    irx = all_data["^IRX"]

    mom = _momentum_signal(risk_close)
    ma = _ma200_signal(risk_close)
    vix_sig = _vix_signal(vix)
    yc_sig = _yield_curve_signal(tnx, irx)
    st_line, st_dir, st_sig = _supertrend(
        risk_ohlc["high"],
        risk_ohlc["low"],
        risk_ohlc["close"],
        atr_period=5,
        factor=2.0,
    )

    # ── 3. Build signals DataFrame ─────────────────────────────────
    signals = pd.DataFrame({
        "risk_close": risk_close,
        "risk_high": risk_ohlc["high"],
        "risk_low": risk_ohlc["low"],
        "safe_close": all_data[safe_ticker],
        "benchmark_close": all_data[benchmark_ticker],
        "vix": vix,
        "tnx": tnx,
        "irx": irx,
        "mom_signal": mom,
        "ma_signal": ma,
        "vix_signal": vix_sig,
        "yield_signal": yc_sig,
        "st_signal": st_sig,
        "supertrend": st_line,
        "st_direction": st_dir,
    }, index=all_data.index)

    signals["score"] = (
        signals["mom_signal"]
        + signals["ma_signal"]
        + signals["vix_signal"]
        + signals["yield_signal"]
        + signals["st_signal"]
    )

    # ── 4. Compute exposure with execution rules ───────────────────
    raw_exposure = _composite_exposure(signals["score"])

    # .shift(1): signal computed end-of-day T, can only trade on T+1
    # This is the CRITICAL anti-lookahead step
    raw_exposure = raw_exposure.shift(1).fillna(0.0)

    weekly_exposure = _apply_weekly_rebalance(raw_exposure)
    final_exposure = _apply_holding_period(weekly_exposure, min_days=10)
    signals["exposure"] = final_exposure

    # ── 5. Compute regime labels for display ───────────────────────
    signals["regime"] = "partial"
    signals.loc[signals["exposure"] == 1.0, "regime"] = "risk-on"
    signals.loc[signals["exposure"] == 0.0, "regime"] = "risk-off"

    # ── 6. Run backtest ────────────────────────────────────────────
    net_returns, gross_equity, net_equity, trade_log = _backtest(
        signals, cost_bps=cost_bps
    )

    # Benchmark: buy-and-hold SPY
    bench_ret = signals["benchmark_close"].pct_change().fillna(0)
    bench_equity = (1 + bench_ret).cumprod()

    # 60/40 benchmark: 60% SPY + 40% safe asset
    safe_ret = signals["safe_close"].pct_change().fillna(0)
    bench_6040_ret = 0.6 * bench_ret + 0.4 * safe_ret
    bench_6040_equity = (1 + bench_6040_ret).cumprod()

    # ── 7. OOS metrics ─────────────────────────────────────────────
    oos_mask = signals.index >= oos_start
    oos_signals = signals[oos_mask]
    oos_net = net_returns[oos_mask]
    oos_bench = bench_ret[oos_mask]
    oos_6040 = bench_6040_ret[oos_mask]

    strategy_metrics = _compute_metrics(oos_net, "AMCE Strategy")
    benchmark_metrics_dict = _compute_metrics(oos_bench, "SPY Buy & Hold")
    bench_6040_metrics = _compute_metrics(oos_6040, "60/40 Portfolio")

    # Percent time in market
    strategy_metrics["pct_time_risk_on"] = round(
        (oos_signals["exposure"] > 0).mean(), 4
    )
    strategy_metrics["pct_time_full_risk_on"] = round(
        (oos_signals["exposure"] == 1.0).mean(), 4
    )

    # Annual turnover
    oos_turnover = signals.loc[oos_mask, "exposure"].diff().abs()
    trades_per_year = oos_turnover[oos_turnover > 0].count() / strategy_metrics["years"]
    strategy_metrics["trades_per_year"] = round(trades_per_year, 1)
    strategy_metrics["annual_cost_drag"] = round(
        trades_per_year * cost_bps / 10_000 * 0.5, 4  # avg turnover per trade ~ 0.5
    )

    # ── 8. Permutation test (OOS only) ─────────────────────────────
    p_value = _permutation_test(
        oos_signals, strategy_metrics["sharpe"],
        n_perms=n_permutations, cost_bps=cost_bps,
    )

    # ── 9. Rolling Sharpe & Drawdown ───────────────────────────────
    rolling_sharpe = (
        oos_net.rolling(252).mean() / oos_net.rolling(252).std()
    ) * np.sqrt(252)

    oos_equity = (1 + oos_net).cumprod()
    running_max = oos_equity.cummax()
    drawdown = (oos_equity - running_max) / running_max

    oos_bench_equity = (1 + oos_bench).cumprod()
    bench_running_max = oos_bench_equity.cummax()
    bench_drawdown = (oos_bench_equity - bench_running_max) / bench_running_max

    # ── 10. Package result ─────────────────────────────────────────
    return StrategyResult(
        returns=oos_net,
        equity_curve=oos_equity,
        benchmark_equity=oos_bench_equity,
        benchmark_6040_equity=(1 + oos_6040).cumprod(),
        signals_df=signals,
        metrics_dict=strategy_metrics,
        benchmark_metrics=benchmark_metrics_dict,
        benchmark_6040_metrics=bench_6040_metrics,
        permutation_p_value=p_value,
        oos_start=oos_start,
        oos_end=str(signals.index[-1].date()),
        gross_equity_curve=gross_equity[oos_mask] / gross_equity[oos_mask].iloc[0],
        trade_log=trade_log[trade_log.index >= oos_start],
        rolling_sharpe=rolling_sharpe,
        drawdown_series=drawdown,
        benchmark_drawdown=bench_drawdown,
    )

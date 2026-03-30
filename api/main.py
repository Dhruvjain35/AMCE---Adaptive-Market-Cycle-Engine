"""
AMCE API — FastAPI backend for the Adaptive Market Cycle Engine.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root so amce package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amce.strategy.engine import run_strategy

app = FastAPI(
    title="AMCE API",
    description="Adaptive Market Cycle Engine — trend-following macro rotation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    risk_asset: str = Field(default="QQQ", description="Risk-on ticker")
    safe_asset: str = Field(default="IEF", description="Risk-off ticker")
    start_year: int = Field(default=2005, ge=2003, le=2025)
    end_year: int = Field(default=2024, ge=2010, le=2030)


def _sanitize(obj):
    """Recursively replace NaN/inf with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _series_to_records(s: pd.Series, value_name: str = "value") -> list[dict]:
    """Convert a pandas Series with DatetimeIndex to JSON-safe records."""
    df = s.reset_index()
    df.columns = ["date", value_name]
    df["date"] = df["date"].astype(str)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(df.notna(), None)
    return df.to_dict("records")


def _permutation_null_sharpes(
    oos_df: pd.DataFrame,
    n_perms: int = 1000,
    cost_bps: float = 5.0,
) -> list[float]:
    """
    Mirror amce.strategy.engine._permutation_test but return all permuted Sharpes.
    Same RNG seed and shuffle logic as the engine — does not modify engine.py.
    """
    df = oos_df.copy()
    df["risk_ret"] = df["risk_close"].pct_change()
    df["safe_ret"] = df["safe_close"].pct_change()
    mondays = df.index[df.index.dayofweek == 0]
    weekly_exposures = df.loc[mondays, "exposure"].values.copy()
    rng = np.random.default_rng(42)
    sharpes: list[float] = []
    for _ in range(n_perms):
        shuffled = weekly_exposures.copy()
        rng.shuffle(shuffled)
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
            perm_sharpe = float((perm_ret_net.mean() * 252) / ann_vol)
        else:
            perm_sharpe = 0.0
        sharpes.append(perm_sharpe)
    return sharpes


def _signals_to_records(sdf: pd.DataFrame) -> list[dict]:
    out = sdf.reset_index()
    first = out.columns[0]
    if first != "date":
        out = out.rename(columns={first: "date"})
    out["date"] = out["date"].astype(str)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(out.notna(), None)
    return out.to_dict("records")


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """Run the full AMCE strategy analysis and return all results."""
    try:
        result = run_strategy(
            start_date=f"{req.start_year}-01-01",
            end_date=f"{req.end_year}-12-31",
            risk_ticker=req.risk_asset,
            safe_ticker=req.safe_asset,
            benchmark_ticker="SPY",
            cost_bps=5.0,
            n_permutations=1000,
            oos_start="2016-01-01",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    sm = result.metrics_dict
    bm = result.benchmark_metrics
    bm6040 = result.benchmark_6040_metrics
    sdf = result.signals_df
    oos_signals = sdf[sdf.index >= result.oos_start].copy()
    null_sharpes = _permutation_null_sharpes(oos_signals, n_perms=1000, cost_bps=5.0)
    null_arr = np.array(null_sharpes, dtype=float)
    null_mean = float(null_arr.mean())
    null_std = float(null_arr.std(ddof=1)) if len(null_arr) > 1 else 0.0
    actual_sharpe = float(sm["sharpe"])
    beaten = sum(1 for x in null_sharpes if x < actual_sharpe)
    beaten_frac = beaten / len(null_sharpes) if null_sharpes else 0.0
    top_percent = round(100 * (1 - beaten_frac), 2)

    last = sdf.iloc[-1]
    last_date = str(sdf.index[-1].date())

    risk_price = float(last["risk_close"])
    ma200_val = float(sdf["risk_close"].rolling(200).mean().iloc[-1])
    mom_12m = float(sdf["risk_close"].pct_change(252).iloc[-1])
    mom_1m = float(sdf["risk_close"].pct_change(21).iloc[-1])
    vix_val = float(last["vix"])
    tnx_val = float(last["tnx"])
    irx_val = float(last["irx"])
    spread_bps = round((tnx_val - irx_val) * 100, 1)

    regime_raw = str(last["regime"])
    if regime_raw == "risk-on":
        overall = "RISK-ON"
    elif regime_raw == "risk-off":
        overall = "RISK-OFF"
    else:
        overall = "PARTIAL"
    score = int(last["score"])

    oos_signals["regime_block"] = (
        oos_signals["regime"] != oos_signals["regime"].shift(1)
    ).cumsum()

    regimes: list[dict] = []
    for _, block in oos_signals.groupby("regime_block"):
        regimes.append({
            "start": str(block.index[0].date()),
            "end": str(block.index[-1].date()),
            "regime": str(block["regime"].iloc[0]),
        })

    equity_curve = _series_to_records(result.equity_curve, "amce")
    bench_equity = _series_to_records(result.benchmark_equity, "spy")
    bench_6040 = _series_to_records(result.benchmark_6040_equity, "sixtyforty")
    gross_equity = _series_to_records(result.gross_equity_curve, "gross")

    eq_map: dict[str, dict] = {}
    for r in equity_curve:
        eq_map[r["date"]] = {"date": r["date"], "amce": r["amce"]}
    for r in bench_equity:
        if r["date"] in eq_map:
            eq_map[r["date"]]["spy"] = r["spy"]
    for r in bench_6040:
        if r["date"] in eq_map:
            eq_map[r["date"]]["sixtyforty"] = r["sixtyforty"]
    for r in gross_equity:
        if r["date"] in eq_map:
            eq_map[r["date"]]["gross"] = r["gross"]
    equity_records = list(eq_map.values())

    rolling_sharpe = _series_to_records(result.rolling_sharpe, "sharpe")

    dd_amce = _series_to_records(result.drawdown_series, "amce_dd")
    dd_bench = _series_to_records(result.benchmark_drawdown, "spy_dd")
    dd_map: dict[str, dict] = {}
    for r in dd_amce:
        dd_map[r["date"]] = {"date": r["date"], "amce_dd": r["amce_dd"]}
    for r in dd_bench:
        if r["date"] in dd_map:
            dd_map[r["date"]]["spy_dd"] = r["spy_dd"]
    drawdown_records = list(dd_map.values())

    covid_pct_cash = None
    covid_mask = (sdf.index >= "2020-02-01") & (sdf.index <= "2020-04-30")
    if covid_mask.any():
        covid_pct_cash = round(float((1.0 - sdf.loc[covid_mask, "exposure"]).mean() * 100), 1)

    gross_cagr = round(
        float((result.gross_equity_curve.iloc[-1] ** (1 / sm["years"])) - 1),
        4,
    )

    hac_p = round(result.hac_mean_return_pvalue, 4)
    perm_p = round(result.permutation_p_value, 4)
    perm_block = {
        # Headline significance: HAC on OOS mean daily return (serial-correlation robust)
        "p_value": hac_p,
        "hac_mean_return_p_value": hac_p,
        # Timing-shuffle null (same weekly blocks, random order) — Sharpe vs permutation
        "timing_shuffle_p_value": perm_p,
        "strategy_sharpe": sm["sharpe"],
        "null_distribution": null_sharpes,
        "null_mean": round(null_mean, 4),
        "null_std": round(null_std, 4),
        "percentile": round(beaten_frac * 100, 1),
        "top_percent": top_percent,
    }

    educational_stats = {
        "covid_pct_in_cash_feb_apr_2020": covid_pct_cash,
        "null_sharpe_mean": round(null_mean, 4),
        "null_sharpe_std": round(null_std, 4),
        "hac_mean_return_p_value": hac_p,
        "permutation_p_value": perm_p,
        "random_strategies_beaten_pct": round(beaten_frac * 100, 1),
    }

    signals_records = _signals_to_records(
        oos_signals[
            [
                "risk_close",
                "safe_close",
                "benchmark_close",
                "vix",
                "tnx",
                "irx",
                "mom_signal",
                "ma_signal",
                "vix_signal",
                "yield_signal",
                "st_signal",
                "supertrend",
                "st_direction",
                "score",
                "exposure",
                "regime",
            ]
        ]
    )

    st_val = float(last["supertrend"]) if pd.notna(last["supertrend"]) else 0.0
    st_dir = int(last["st_direction"]) if pd.notna(last["st_direction"]) else 0

    current_signal = {
        "overall": overall,
        "score": score,
        "date": last_date,
        "components": {
            "momentum": {
                "signal": bool(last["mom_signal"]),
                "reading": f"12M: {mom_12m * 100:+.1f}%, 1M: {mom_1m * 100:+.1f}%",
                "value": round(mom_12m - mom_1m, 4),
                "mom_12m": round(mom_12m, 4),
                "mom_1m": round(mom_1m, 4),
            },
            "ma200": {
                "signal": bool(last["ma_signal"]),
                "reading": f"Price ${risk_price:.2f}, MA ${ma200_val:.2f}",
                "value": round((risk_price / ma200_val - 1) * 100, 2),
                "price": round(risk_price, 2),
                "ma": round(ma200_val, 2),
            },
            "vix": {
                "signal": bool(last["vix_signal"]),
                "reading": f"VIX: {vix_val:.1f}",
                "value": round(vix_val, 2),
            },
            "yield_curve": {
                "signal": bool(last["yield_signal"]),
                "reading": f"Spread: {spread_bps:+.0f}bps",
                "value": round(tnx_val - irx_val, 4),
                "tnx": round(tnx_val, 2),
                "irx": round(irx_val, 2),
                "spread_bps": spread_bps,
            },
            "supertrend": {
                "signal": bool(last["st_signal"]),
                "reading": f"ST: ${st_val:.2f} | dir {st_dir:+d}",
                "value": round(st_val, 4),
                "direction": st_dir,
                "atr_period": 5,
                "factor": 2.0,
            },
        },
    }

    payload = {
        "metrics": sm,
        "benchmark_metrics": bm,
        "benchmark_6040_metrics": bm6040,
        "oos_start": result.oos_start,
        "oos_end": result.oos_end,
        "equity_curve": equity_records,
        "signals": signals_records,
        "rolling_sharpe": rolling_sharpe,
        "drawdown": drawdown_records,
        "regimes": regimes,
        "regime_history": regimes,
        "current_signal": current_signal,
        "permutation": perm_block,
        "permutation_result": perm_block,
        "wealth": {
            "initial": 10000,
            "amce_final": round(result.equity_curve.iloc[-1] * 10000, 0),
            "spy_final": round(result.benchmark_equity.iloc[-1] * 10000, 0),
            "sixtyforty_final": round(result.benchmark_6040_equity.iloc[-1] * 10000, 0),
            "advantage": round(
                (result.equity_curve.iloc[-1] - result.benchmark_equity.iloc[-1]) * 10000,
                0,
            ),
        },
        "costs": {
            "trades_per_year": sm.get("trades_per_year", 0),
            "cost_bps": 5,
            "annual_drag_bps": round(sm.get("annual_cost_drag", 0) * 10000, 1),
            "gross_cagr": gross_cagr,
            "net_cagr": sm["cagr"],
        },
        "educational_stats": educational_stats,
        "risk_asset": req.risk_asset,
        "safe_asset": req.safe_asset,
    }

    return _sanitize(payload)


@app.get("/health")
async def health():
    return {"status": "ok"}

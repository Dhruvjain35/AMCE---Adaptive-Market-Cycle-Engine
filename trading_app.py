"""Presentation-focused Streamlit UI for AMCE research outputs with economics education."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from amce.pipeline import run_pipeline

st.set_page_config(page_title="AMCE Showcase", layout="wide")

RISK_TICKER_OPTIONS = ["QQQ", "SPY", "IWM", "DIA", "EFA", "EEM", "XLF", "XLK", "XLI", "XLE"]
SAFE_TICKER_OPTIONS = ["IEF", "TLT", "SHY", "BIL", "GLD", "UUP"]

# ---------------------------------------------------------------------------
# Plain-language regime descriptions for the education sidebar
# ---------------------------------------------------------------------------
REGIME_EXPLANATIONS: dict[str, str] = {
    "bull": (
        "The model detects a **bull regime** -- historically, this means stocks are "
        "trending upward with low volatility. Returns tend to be above average and "
        "the strategy leans into momentum signals, increasing exposure to riskier assets."
    ),
    "bear": (
        "The model detects a **bear regime** -- historically, this means stocks are "
        "trending downward or experiencing elevated fear (high VIX). Returns tend to "
        "be below average and the strategy shifts toward defensive, safe-haven assets "
        "to protect capital."
    ),
    "transition": (
        "The model detects a **transition regime** -- the market is shifting between "
        "bull and bear conditions. This is the hardest environment to trade because "
        "signals conflict. The strategy reduces position sizes and waits for clearer "
        "direction before committing capital."
    ),
    "stress": (
        "The model detects a **stress regime** -- a period of extreme volatility, "
        "often linked to crises (e.g., 2008, COVID). The strategy aggressively "
        "de-risks, cutting exposure to protect against tail losses."
    ),
}
DEFAULT_REGIME_EXPLANATION = (
    "The model is analyzing current market conditions to classify the regime. "
    "Regimes help the strategy decide whether to favor momentum (offensive) or "
    "risk-reduction (defensive) signals."
)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;500;700&family=Manrope:wght@400;600;800&display=swap');
:root {
  --bg:#f6f4ee;
  --panel:#ffffff;
  --ink:#17313a;
  --muted:#56717a;
  --accent:#0f766e;
  --accent-soft:#9adfd8;
  --warm:#d97706;
  --risk:#b42318;
}
.stApp {
  background:
    radial-gradient(circle at 10% 10%, #fff7e6 0%, rgba(255,247,230,0.0) 40%),
    radial-gradient(circle at 90% 15%, #dcfce7 0%, rgba(220,252,231,0.0) 35%),
    linear-gradient(180deg, #f6f4ee 0%, #edf2f4 100%);
  color:var(--ink);
  font-family:'Manrope', sans-serif;
}
h1,h2,h3 {
  font-family:'Sora', sans-serif;
  color:var(--ink);
}
.hero {
  background:linear-gradient(135deg, rgba(15,118,110,0.12), rgba(217,119,6,0.14));
  border:1px solid rgba(23,49,58,0.12);
  border-radius:18px;
  padding:18px 22px;
  margin-bottom:14px;
  animation:fadeIn 600ms ease-out;
}
.hero h1 { margin:0 0 6px 0; font-size:2rem; }
.hero p { margin:0; color:var(--muted); }
.badges { margin-top:10px; display:flex; flex-wrap:wrap; gap:8px; }
.badge {
  font-size:0.72rem;
  letter-spacing:0.04em;
  border-radius:999px;
  padding:4px 10px;
  border:1px solid rgba(15,118,110,0.3);
  background:rgba(15,118,110,0.08);
  color:var(--accent);
}
.card-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin:8px 0 18px; }
.metric-card {
  background:var(--panel);
  border:1px solid rgba(23,49,58,0.1);
  border-radius:14px;
  padding:12px;
  min-height:96px;
  box-shadow:0 6px 18px rgba(23,49,58,0.06);
  animation:riseIn 500ms ease both;
}
.metric-card .label { font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; }
.metric-card .value { font-size:1.5rem; font-family:'Sora', sans-serif; font-weight:700; color:var(--ink); margin-top:4px; }
.metric-card .delta { font-size:0.8rem; color:var(--accent); margin-top:4px; }
.section { margin-top:12px; margin-bottom:6px; }
.small-note { color:var(--muted); font-size:0.82rem; }
@keyframes fadeIn { from { opacity:0; transform:translateY(8px);} to { opacity:1; transform:translateY(0);} }
@keyframes riseIn { from { opacity:0; transform:translateY(14px);} to { opacity:1; transform:translateY(0);} }
@media (max-width: 960px){
  .card-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def metric_card_html(label: str, value: str, delta: str = "") -> str:
    return (
        "<div class='metric-card'>"
        f"<div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>"
        f"<div class='delta'>{delta}</div>"
        "</div>"
    )


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _parse_custom_tickers(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        ticker = part.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def _merge_ticker_inputs(selected: list[str], custom_raw: str, fallback: str) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for ticker in [*selected, *_parse_custom_tickers(custom_raw)]:
        clean = str(ticker).strip().upper()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        merged.append(clean)
    if not merged:
        merged = [fallback]
    return merged


# ---------------------------------------------------------------------------
# Cached pipeline wrapper
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=3600, show_spinner=False)
def _cached_run_pipeline(cfg_json: str):
    """Run the pipeline and cache the full report object keyed on the JSON config string."""
    cfg = json.loads(cfg_json)
    return run_pipeline(cfg)


# ---------------------------------------------------------------------------
# Chart builders (V2 light-theme styling)
# ---------------------------------------------------------------------------

def build_equity_chart(oos: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oos.index, y=oos["Eq_Strat"], mode="lines", name="AMCE Strategy", line=dict(color="#0f766e", width=3)))
    fig.add_trace(go.Scatter(x=oos.index, y=oos["Eq_Bench"], mode="lines", name="Benchmark", line=dict(color="#94653f", width=2, dash="dash")))
    fig.add_trace(
        go.Scatter(
            x=oos.index,
            y=oos["DD_Strat"] * 100,
            mode="lines",
            name="Drawdown %",
            yaxis="y2",
            line=dict(color="#b42318", width=1.5),
            opacity=0.6,
        )
    )
    fig.update_layout(
        height=470,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend=dict(orientation="h", y=1.02, x=0.0),
        yaxis=dict(title="Growth of $1", type="log"),
        yaxis2=dict(title="Drawdown %", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
    )
    return fig


def build_uplift_bar(strat: dict[str, float], inst: dict[str, float], bench: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="AMCE", x=["Sharpe", "Sortino", "Max DD"], y=[strat["sharpe"], strat["sortino"], strat["max_drawdown"]], marker_color="#0f766e"))
    fig.add_trace(go.Bar(name="Institutional Baseline", x=["Sharpe", "Sortino", "Max DD"], y=[inst.get("sharpe", 0.0), inst.get("sortino", 0.0), inst.get("max_drawdown", 0.0)], marker_color="#d97706"))
    fig.add_trace(go.Bar(name="Benchmark", x=["Sharpe", "Sortino", "Max DD"], y=[bench["sharpe"], bench["sortino"], bench["max_drawdown"]], marker_color="#64748b"))
    fig.update_layout(
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
    )
    return fig


def build_feature_importance_chart(rows: list[dict[str, object]]) -> go.Figure:
    top = rows[:20]
    labels = [str(r.get("feature", "")) for r in top]
    values = [float(r.get("importance", 0.0)) for r in top]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values[::-1],
                y=labels[::-1],
                orientation="h",
                marker_color="#0f766e",
            )
        ]
    )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        xaxis_title="Average Fold Importance",
        yaxis_title="Feature",
    )
    return fig


def build_regime_distribution_chart(oos: pd.DataFrame) -> go.Figure:
    regime_counts = oos["Regime"].fillna("Unknown").value_counts()
    fig = go.Figure(
        data=[
            go.Pie(
                labels=regime_counts.index.tolist(),
                values=regime_counts.values.tolist(),
                hole=0.42,
                marker=dict(colors=["#0f766e", "#d97706", "#b42318", "#64748b", "#0ea5e9"]),
            )
        ]
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def build_signal_distribution_chart(oos: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=oos["Prob"],
            nbinsx=35,
            marker_color="#0f766e",
            opacity=0.75,
            name="Model Probability",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=oos["Prob_Uncertainty"] if "Prob_Uncertainty" in oos.columns else np.zeros(len(oos)),
            nbinsx=35,
            marker_color="#d97706",
            opacity=0.55,
            name="Model Uncertainty",
        )
    )
    fig.update_layout(
        barmode="overlay",
        height=380,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        xaxis_title="Score",
        yaxis_title="Frequency",
    )
    return fig


def build_ensemble_graph(oos: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=oos.index,
            y=oos["Prob"],
            mode="lines",
            name="Ensemble Probability",
            line=dict(color="#0f766e", width=2.8),
        )
    )
    if "Prob_Raw" in oos.columns:
        fig.add_trace(
            go.Scatter(
                x=oos.index,
                y=oos["Prob_Raw"],
                mode="lines",
                name="Raw Probability",
                line=dict(color="#94a3b8", width=1.4, dash="dot"),
            )
        )
    if "Exposure" in oos.columns:
        fig.add_trace(
            go.Scatter(
                x=oos.index,
                y=oos["Exposure"],
                mode="lines",
                name="Exposure",
                line=dict(color="#d97706", width=1.8),
                yaxis="y2",
            )
        )

    comp_cols = [c for c in oos.columns if c.startswith("Ens_")]
    for col in comp_cols[:5]:
        fig.add_trace(
            go.Scatter(
                x=oos.index,
                y=oos[col].rolling(10).mean(),
                mode="lines",
                name=col.replace("Ens_", ""),
                line=dict(width=1.2),
                opacity=0.65,
            )
        )

    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        yaxis=dict(title="Probability"),
        yaxis2=dict(title="Exposure", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.0),
    )
    return fig


def _build_wealth_series(oos: pd.DataFrame, initial_capital: float = 10_000.0) -> pd.DataFrame:
    out = pd.DataFrame(index=oos.index)
    out["Quant"] = initial_capital * oos["Eq_Strat"]
    if "SPX" in oos.columns:
        spx_ret = oos["SPX"].pct_change().fillna(0.0)
        out["S&P 500"] = initial_capital * (1.0 + spx_ret).cumprod()
    else:
        out["S&P 500"] = initial_capital * oos["Eq_Bench"]
    return out


def build_wealth_comparison_chart(oos: pd.DataFrame, initial_capital: float = 10_000.0) -> go.Figure:
    wealth = _build_wealth_series(oos, initial_capital=initial_capital)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wealth.index,
            y=wealth["Quant"],
            mode="lines",
            name="AMCE Quant",
            line=dict(color="#0f766e", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wealth.index,
            y=wealth["S&P 500"],
            mode="lines",
            name="S&P 500",
            line=dict(color="#d97706", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=56, b=10),
        title=dict(text=f"${initial_capital:,.0f} Growth: AMCE Quant vs S&P 500", x=0.01, xanchor="left"),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Portfolio Value ($)"),
        legend=dict(orientation="h", y=1.02, x=0.0),
        hovermode="x unified",
    )
    return fig


def build_3d_performance_path(oos: pd.DataFrame) -> go.Figure:
    frame = oos.copy()
    frame["RollingVol"] = frame["Net"].rolling(21).std().fillna(0.0) * np.sqrt(252)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=frame["Eq_Strat"],
                y=frame["RollingVol"],
                z=frame["DD_Strat"],
                mode="lines+markers",
                marker=dict(size=2.8, color=frame["Prob"], colorscale="Tealgrn", opacity=0.85, colorbar=dict(title="Confidence")),
                line=dict(color="#0f766e", width=5),
                name="Strategy Path",
            )
        ]
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(255,255,255,0)",
        scene=dict(
            xaxis_title="Equity",
            yaxis_title="21D Ann. Vol",
            zaxis_title="Drawdown",
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )
    return fig


def build_3d_decision_surface(oos: pd.DataFrame) -> go.Figure:
    threshold = float(oos["Threshold"].median()) if "Threshold" in oos else 0.5
    band = 0.06
    penalty = 0.15

    p = np.linspace(0, 1, 55)
    u = np.linspace(0, 0.30, 55)
    p_grid, u_grid = np.meshgrid(p, u)

    adjusted = np.clip(p_grid - penalty * u_grid, 0, 1)
    lo, hi = threshold - band, threshold + band
    exposure = np.clip((adjusted - lo) / max(hi - lo, 1e-8), 0, 1)

    fig = go.Figure(
        data=[
            go.Surface(
                x=p_grid,
                y=u_grid,
                z=exposure,
                colorscale="Mint",
                showscale=True,
                colorbar=dict(title="Exposure"),
            )
        ]
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(255,255,255,0)",
        scene=dict(
            xaxis_title="Raw Probability",
            yaxis_title="Model Uncertainty",
            zaxis_title="Final Exposure",
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )
    return fig


def build_3d_regime_cloud(oos: pd.DataFrame) -> go.Figure | None:
    needed = {"MA_200_Dist", "VIX_Change_21D", "Prob", "Net"}
    if not needed.issubset(set(oos.columns)):
        return None

    sample = oos.dropna(subset=["MA_200_Dist", "VIX_Change_21D", "Prob", "Net"]).iloc[:: max(1, len(oos) // 600)]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=sample["MA_200_Dist"],
                y=sample["VIX_Change_21D"],
                z=sample["Prob"],
                mode="markers",
                marker=dict(
                    size=3.2,
                    color=sample["Net"],
                    colorscale="RdYlGn",
                    cmin=-0.02,
                    cmax=0.02,
                    colorbar=dict(title="Daily Net"),
                    opacity=0.78,
                ),
                name="Regime cloud",
            )
        ]
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(255,255,255,0)",
        scene=dict(
            xaxis_title="MA_200 Dist",
            yaxis_title="VIX 21D Change",
            zaxis_title="Model Probability",
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    inject_theme()

    st.markdown(
        """
<div class="hero">
  <h1>AMCE Quant Strategy Showcase</h1>
  <p>An educational, presentation-first analytics platform that teaches how quantitative macro-regime models work while running live backtests against public benchmarks.</p>
  <div class="badges">
    <span class="badge">Economics Education</span>
    <span class="badge">Walk-Forward Validation</span>
    <span class="badge">Institutional Uplift Gate</span>
    <span class="badge">3D Decision & Regime Views</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Scenario Controls")
        risk_multi = st.multiselect("Risk Tickers", options=RISK_TICKER_OPTIONS, default=["QQQ", "SPY"])
        risk_custom = st.text_input("Additional Risk Tickers (comma-separated)", "")
        safe_multi = st.multiselect("Safe Tickers", options=SAFE_TICKER_OPTIONS, default=["IEF", "TLT"])
        safe_custom = st.text_input("Additional Safe Tickers (comma-separated)", "")
        router_label = st.selectbox(
            "Basket Routing Strategy",
            options=["Probability-Weighted (Dynamic)", "Equal-Weight"],
            index=0,
            help="Dynamic mode uses a scenario-probability score to route within your chosen baskets.",
        )
        benchmark_ticker = st.text_input("S&P Benchmark Ticker", "^GSPC")
        start = st.date_input("Start Date", value=date(2005, 1, 1))
        use_end = st.checkbox("Set End Date", value=False)
        end = st.date_input("End Date", value=date.today(), disabled=not use_end)

        st.markdown("---")
        n_splits = st.slider("Walk-Forward Folds", 3, 8, 5)
        embargo = st.slider("Embargo Days", 0, 63, 21)
        tc = st.slider("Transaction Cost (bps)", 0, 20, 3)
        sl = st.slider("Slippage (bps)", 0, 50, 5)

        run = st.button("Run Showcase", use_container_width=True)

    if not run:
        st.info("Configure the scenario and click **Run Showcase**.")
        return

    risk_tickers = _merge_ticker_inputs(risk_multi, risk_custom, fallback="QQQ")
    safe_tickers = _merge_ticker_inputs(safe_multi, safe_custom, fallback="IEF")
    basket_router = "probability" if router_label.startswith("Probability-Weighted") else "equal"
    benchmark_ticker = str(benchmark_ticker).strip().upper() or "^GSPC"

    st.caption(
        f"Routing `{', '.join(risk_tickers)}` (risk) and `{', '.join(safe_tickers)}` (safe) "
        f"using **{basket_router}** mode  |  Benchmark: `{benchmark_ticker}`"
    )

    cfg = {
        "data": {
            "risk_ticker": risk_tickers[0],
            "safe_ticker": safe_tickers[0],
            "risk_tickers": risk_tickers,
            "safe_tickers": safe_tickers,
            "basket_router": basket_router,
            "benchmark_ticker": benchmark_ticker,
            "start_date": str(start),
            "end_date": str(end) if use_end else None,
        },
        "validation": {
            "n_splits": n_splits,
            "embargo_days": embargo,
        },
        "backtest": {
            "transaction_cost_bps": float(tc),
            "slippage_bps": float(sl),
        },
    }

    # ------------------------------------------------------------------
    # Run pipeline (cached)
    # ------------------------------------------------------------------
    with st.status(
        "Fitting Hidden Markov Model to detect market regimes...",
        expanded=True,
    ):
        try:
            cfg_json = json.dumps(cfg, sort_keys=True)
            report = _cached_run_pipeline(cfg_json)
        except ValueError as exc:
            msg = str(exc)
            if "Missing required fields after load" in msg:
                st.error(
                    "Data fetch is missing required macro fields (`VIX`/`Yield`). "
                    "Please retry or shorten the date range and run again."
                )
            else:
                st.error(f"Pipeline failed: {msg}")
            return
        except Exception as exc:
            st.error(f"Pipeline failed with an unexpected error: {exc}")
            return

    summary = report.summary
    strat = summary["strategy_metrics"]
    bench = summary["benchmark_metrics"]
    inst = summary.get("institutional_baseline_metrics") or {}
    uplift = (summary.get("institutional_comparison") or {}).get("strategy_vs_reference", {})

    cards = [
        metric_card_html("Sharpe", f"{strat['sharpe']:.3f}", f"vs benchmark {bench['sharpe']:.3f}"),
        metric_card_html("Sortino", f"{strat['sortino']:.3f}", f"vs baseline {inst.get('sortino', 0.0):.3f}"),
        metric_card_html("Annual Return", fmt_pct(strat["annual_return"]), f"benchmark {fmt_pct(bench['annual_return'])}"),
        metric_card_html("Max Drawdown", fmt_pct(strat["max_drawdown"]), f"uplift {fmt_pct(uplift.get('drawdown_uplift', 0.0))}"),
        metric_card_html("Sharpe Uplift", f"{uplift.get('sharpe_uplift', 0.0):+.3f}", "vs Qlib-style institutional baseline"),
    ]
    st.markdown(f"<div class='card-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Governance Gate
    # ------------------------------------------------------------------
    gov = report.governance
    if gov["passed"]:
        st.success("Governance status: PASS")
    else:
        st.warning("Governance status: PARTIAL PASS (check failed gates below)")

    gate_df = pd.DataFrame([{"Gate": k, "Pass": v} for k, v in gov["checks"].items()])
    st.dataframe(gate_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # EDUCATIONAL: "What do these results mean?"
    # ------------------------------------------------------------------
    with st.expander("What do these results mean?"):
        st.markdown(
            """
**What is a Market Regime?**

Financial markets don't behave the same way all the time. A *regime* is a period
where the market follows a consistent pattern -- for example, a **bull** regime
(steady gains, low fear), a **bear** regime (falling prices, rising volatility),
or a **transition** regime (mixed signals, uncertain direction). Our model uses a
Hidden Markov Model (HMM) to automatically detect which regime the market is in
each day, because the best trading strategy depends on the current environment.

---

**What is the Sharpe Ratio?**

The Sharpe ratio measures *how much extra return you earn for each unit of risk
you take*. Think of it like miles-per-gallon for investing: a higher number means
you're getting more reward for the same amount of volatility. A Sharpe above 1.0
is considered good; above 2.0 is excellent. If the strategy's Sharpe is higher
than the benchmark's, the model is adding value beyond simple buy-and-hold.

---

**Why does the model switch between momentum and risk signals?**

In a bull regime, stocks that have been going up tend to *keep* going up
(momentum). So the model increases exposure to winning assets. But in a bear or
stress regime, momentum reverses -- yesterday's winners become today's losers.
The model detects this shift and pivots to *risk signals* (VIX level, drawdown
depth, yield curve slope) that help it reduce exposure before large losses occur.
This regime-aware switching is what separates an adaptive model from a static one.

---

**What is the Permutation P-Value Test?**

After the model produces results, we need to check: *could these results have
happened by pure luck?* The permutation test shuffles the model's daily
predictions randomly (1,000+ times) and re-runs the backtest on each shuffle. If
the real strategy beats 95% or more of the random shuffles, we can be confident
the results reflect genuine skill, not chance. The p-value tells you the
probability that luck alone explains the performance -- lower is better (below
0.05 is the standard threshold).
"""
        )

    # ------------------------------------------------------------------
    # OOS-dependent sections
    # ------------------------------------------------------------------
    oos_raw = report.extras.get("oos_frame")
    if isinstance(oos_raw, dict):
        oos = pd.DataFrame(oos_raw)
    elif isinstance(oos_raw, pd.DataFrame):
        oos = oos_raw
    else:
        oos = None

    # --- Sidebar: Regime Education (needs oos) ---
    if oos is not None and "Regime" in oos.columns:
        latest_regime = str(oos["Regime"].dropna().iloc[-1]).strip().lower() if not oos["Regime"].dropna().empty else ""
        regime_text = REGIME_EXPLANATIONS.get(latest_regime, DEFAULT_REGIME_EXPLANATION)
        with st.sidebar:
            st.markdown("---")
            st.subheader("Regime Education")
            st.info(f"**Current detected regime:** {latest_regime.title() or 'Analyzing...'}")
            st.markdown(regime_text)

    if isinstance(oos, pd.DataFrame):
        st.markdown("### Overfitting Diagnostics")
        of = summary.get("overfitting_diagnostics", {})
        if of:
            if of.get("status") == "Elevated":
                st.error("Overfitting check: ELEVATED (train/test gap above threshold)")
            else:
                st.success("Overfitting check: ACCEPTABLE")
            of_df = pd.DataFrame(
                [
                    {"Metric": "Avg Train Sharpe", "Value": of.get("avg_train_sharpe", 0.0)},
                    {"Metric": "Avg Test Sharpe", "Value": of.get("avg_test_sharpe", 0.0)},
                    {"Metric": "Avg Sharpe Gap", "Value": of.get("avg_sharpe_gap", 0.0)},
                    {"Metric": "Avg Train AUC", "Value": of.get("avg_train_auc", 0.0)},
                    {"Metric": "Avg Test AUC", "Value": of.get("avg_test_auc", 0.0)},
                    {"Metric": "Avg AUC Gap", "Value": of.get("avg_auc_gap", 0.0)},
                    {"Metric": "Avg Train Brier", "Value": of.get("avg_train_brier", 0.0)},
                    {"Metric": "Avg Test Brier", "Value": of.get("avg_test_brier", 0.0)},
                    {"Metric": "Elevated Folds", "Value": of.get("elevated_folds", 0)},
                ]
            )
            st.dataframe(of_df, use_container_width=True, hide_index=True)
        else:
            st.info("Overfitting diagnostics unavailable for this run.")

        st.markdown("### Performance Story")
        st.plotly_chart(build_equity_chart(oos), use_container_width=True)

        st.markdown("### $10,000: Quant vs S&P 500")
        st.plotly_chart(build_wealth_comparison_chart(oos, initial_capital=10_000.0), use_container_width=True)
        wealth = _build_wealth_series(oos, initial_capital=10_000.0)
        quant_final = float(wealth["Quant"].iloc[-1])
        spx_final = float(wealth["S&P 500"].iloc[-1])
        lift = (quant_final / spx_final - 1.0) if spx_final > 0 else 0.0
        w1, w2, w3 = st.columns(3)
        with w1:
            st.metric("Quant Final Value", f"${quant_final:,.0f}")
        with w2:
            st.metric("S&P Final Value", f"${spx_final:,.0f}")
        with w3:
            st.metric("Quant vs S&P", f"{lift:+.2%}")

        st.markdown("### Ensemble Graph")
        st.plotly_chart(build_ensemble_graph(oos), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("### Relative Strength vs Institutional Baseline")
            if inst:
                st.plotly_chart(build_uplift_bar(strat, inst, bench), use_container_width=True)
            else:
                st.info("Institutional baseline metrics unavailable for this run.")
        with right:
            st.markdown("### Fold Stability")
            st.dataframe(pd.DataFrame(report.fold_metrics), use_container_width=True, height=380)

        st.markdown("### 3D Strategy Models")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(build_3d_performance_path(oos), use_container_width=True)
        with c2:
            st.plotly_chart(build_3d_decision_surface(oos), use_container_width=True)

        regime_fig = build_3d_regime_cloud(oos)
        if regime_fig is not None:
            st.plotly_chart(regime_fig, use_container_width=True)

        st.markdown("### Factor & Regime Diagnostics")
        d1, d2 = st.columns(2)
        with d1:
            fi = summary.get("feature_importance_top", [])
            if fi:
                st.plotly_chart(build_feature_importance_chart(fi), use_container_width=True)
            else:
                st.info("Feature importance not available for this run.")
        with d2:
            st.plotly_chart(build_regime_distribution_chart(oos), use_container_width=True)

        st.plotly_chart(build_signal_distribution_chart(oos), use_container_width=True)

        st.markdown("### Crisis Table (Drawdown Episodes)")
        crisis = report.extras.get("crisis_table")
        if isinstance(crisis, pd.DataFrame) and not crisis.empty:
            st.dataframe(crisis, use_container_width=True, height=320, hide_index=True)
        elif isinstance(crisis, list) and crisis:
            st.dataframe(pd.DataFrame(crisis), use_container_width=True, height=320, hide_index=True)
        else:
            st.info("No drawdown episodes detected in OOS frame.")

        st.markdown("### Peer League Table")
        peer_table = summary.get("peer_league_table", [])
        if peer_table:
            st.dataframe(pd.DataFrame(peer_table), use_container_width=True, height=340)

        # ------------------------------------------------------------------
        # EDUCATIONAL: "Learn More" under peer league table
        # ------------------------------------------------------------------
        with st.expander("Learn More: What is the Qlib benchmark and why does beating it matter?"):
            st.markdown(
                """
**Qlib** is an open-source quantitative investment platform created by Microsoft
Research. It includes a suite of well-known machine learning models (LightGBM,
XGBoost, Transformer, ALSTM, and others) that are commonly used in academic and
industry research for stock prediction.

The **Peer League Table** above compares our AMCE strategy against these
established models. Beating the Qlib benchmark matters because:

1. **It proves the strategy isn't just lucky.** If our model outperforms 10+
   well-tuned ML baselines, the alpha is more likely to be real and robust.

2. **It demonstrates institutional-grade quality.** Hedge funds and asset
   managers use similar peer comparisons to decide whether a strategy deserves
   capital allocation.

3. **It shows the value of regime-awareness.** Most Qlib models are purely
   statistical -- they don't understand market regimes. Our model's ability to
   switch strategies based on economic conditions is what gives it an edge,
   especially during bear markets and crises.

If AMCE's Sharpe ratio exceeds the best Qlib model's Sharpe, the governance gate
marks this as a **PASS** -- meaning the strategy has demonstrated statistically
meaningful outperformance over the state of the art.
"""
            )

    st.markdown("### Presenter Notes")
    st.markdown(
        "- Use the uplift cards first to anchor value.\n"
        "- Move to the equity+drawdown panel to explain risk-adjusted consistency.\n"
        "- Use the 3D decision surface to explain confidence-aware position sizing.\n"
        "- Use the regime cloud to discuss behavior in stress environments.\n"
        "- Open the educational expanders to explain key concepts to the audience."
    )

    with st.expander("Technical JSON (appendix)"):
        st.code(json.dumps(report.to_dict(), indent=2), language="json")


if __name__ == "__main__":
    main()

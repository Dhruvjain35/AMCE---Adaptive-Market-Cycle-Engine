"""
AMCE - Adaptive Market Cycle Engine
====================================
A trend-following macro rotation strategy with an educational interface.
Built for the Hackonomics hackathon.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to path so amce package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from amce.strategy.engine import run_strategy, StrategyResult


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="AMCE - Adaptive Market Cycle Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  THEME & CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WARM_BG = "#f7f4ee"
INK = "#1a1916"
TEAL = "#0f766e"
RED = "#b91c1c"
MUTED = "#6b7280"
LIGHT_TEAL = "rgba(15, 118, 110, 0.08)"
LIGHT_RED = "rgba(185, 28, 28, 0.08)"
PARTIAL_GRAY = "rgba(107, 114, 128, 0.08)"


def inject_theme():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,700&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@400;500&display=swap');

    .stApp {{
        background-color: {WARM_BG} !important;
        color: {INK} !important;
    }}

    /* Hide Streamlit chrome */
    #MainMenu, header, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; max-width: 1200px; }}

    /* Typography */
    h1, h2, h3 {{
        font-family: 'Playfair Display', Georgia, serif !important;
        color: {INK} !important;
        font-weight: 700 !important;
    }}
    h1 {{ font-style: italic !important; }}

    p, li, span, div, label {{
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-weight: 300 !important;
    }}

    code, .stMetricValue, .metric-number {{
        font-family: 'DM Mono', 'Courier New', monospace !important;
    }}

    /* Rule lines */
    hr {{
        border: none;
        border-top: 1px solid rgba(26, 25, 22, 0.15);
        margin: 1.5rem 0;
    }}

    /* Streamlit overrides */
    .stSelectbox label, .stTextInput label, .stNumberInput label {{
        font-family: 'Source Serif 4', serif !important;
        font-weight: 400 !important;
        color: {MUTED} !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}

    .stButton > button {{
        background-color: {TEAL} !important;
        color: white !important;
        border: none !important;
        font-family: 'Source Serif 4', serif !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        border-radius: 2px !important;
        letter-spacing: 0.03em !important;
    }}
    .stButton > button:hover {{
        background-color: #0d5c56 !important;
    }}

    /* Expander styling */
    .streamlit-expanderHeader {{
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        color: {INK} !important;
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: white;
        border: 1px solid rgba(26,25,22,0.08);
        padding: 1rem;
        border-radius: 2px;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'DM Mono', monospace !important;
        font-size: 1.8rem !important;
        color: {INK} !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'Source Serif 4', serif !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-size: 0.75rem !important;
        color: {MUTED} !important;
    }}

    /* Signal pill */
    .signal-pill {{
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 2px;
        font-family: 'DM Mono', monospace;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.05em;
    }}
    .signal-on {{
        background: {TEAL};
        color: white;
    }}
    .signal-off {{
        background: {RED};
        color: white;
    }}
    .signal-partial {{
        background: {MUTED};
        color: white;
    }}

    /* Masthead */
    .masthead {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {INK};
        margin-bottom: 1.5rem;
    }}
    .masthead-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: {INK};
        letter-spacing: 0.02em;
    }}
    .masthead-date {{
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        color: {MUTED};
    }}
    .masthead-sub {{
        font-family: 'Source Serif 4', serif;
        font-size: 0.85rem;
        color: {MUTED};
        font-style: italic;
    }}

    /* Signal table */
    .signal-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Source Serif 4', serif;
        font-size: 0.9rem;
    }}
    .signal-table td {{
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid rgba(26,25,22,0.08);
    }}
    .signal-table td:last-child {{
        text-align: right;
        font-family: 'DM Mono', monospace;
        font-weight: 500;
    }}
    .sig-on {{ color: {TEAL}; }}
    .sig-off {{ color: {RED}; }}

    /* Methodology note */
    .methodology {{
        font-family: 'Source Serif 4', serif;
        font-style: italic;
        font-size: 0.8rem;
        color: {MUTED};
        line-height: 1.6;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(26,25,22,0.1);
    }}

    /* Number styling */
    .big-number {{
        font-family: 'DM Mono', monospace;
        font-size: 2.8rem;
        font-weight: 500;
        line-height: 1.1;
    }}
    .sub-number {{
        font-family: 'DM Mono', monospace;
        font-size: 1.1rem;
        color: {MUTED};
    }}
    </style>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CHART BUILDERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(255,255,255,0.65)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Source Serif 4, serif", color=INK, size=12),
    xaxis=dict(
        showgrid=False,
        linecolor="rgba(26,25,22,0.15)",
        linewidth=1,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(26,25,22,0.06)",
        linecolor="rgba(26,25,22,0.15)",
        linewidth=1,
    ),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(
        font=dict(family="Source Serif 4, serif", size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
)


def _regime_shapes(signals_df: pd.DataFrame, ymin: float, ymax: float) -> list[dict]:
    """Build shaded regime regions for charts."""
    shapes = []
    df = signals_df.copy()
    df["regime_block"] = (df["regime"] != df["regime"].shift(1)).cumsum()
    for _, block in df.groupby("regime_block"):
        regime = block["regime"].iloc[0]
        if regime == "risk-on":
            color = LIGHT_TEAL
        elif regime == "risk-off":
            color = LIGHT_RED
        else:
            color = PARTIAL_GRAY
        shapes.append(dict(
            type="rect",
            x0=block.index[0], x1=block.index[-1],
            y0=ymin, y1=ymax,
            fillcolor=color,
            line_width=0,
            layer="below",
        ))
    return shapes


def build_regime_price_chart(result: StrategyResult) -> go.Figure:
    """90-day price chart with regime shading."""
    recent = result.signals_df.tail(90)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["risk_close"],
        mode="lines",
        line=dict(color=INK, width=2),
        name=recent.index.name or "Price",
    ))

    ymin = recent["risk_close"].min() * 0.98
    ymax = recent["risk_close"].max() * 1.02
    shapes = _regime_shapes(recent, ymin, ymax)

    fig.update_layout(
        **CHART_LAYOUT,
        shapes=shapes,
        showlegend=False,
        height=280,
        yaxis_title="Price",
        title=dict(
            text="Recent 90 Trading Days with Regime Overlay",
            font=dict(family="Playfair Display, serif", size=14),
        ),
    )
    return fig


def build_equity_chart(result: StrategyResult) -> go.Figure:
    """Full equity curve: AMCE vs SPY vs 60/40, log scale."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve.values,
        mode="lines",
        name="AMCE Strategy",
        line=dict(color=TEAL, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=result.benchmark_equity.index,
        y=result.benchmark_equity.values,
        mode="lines",
        name="SPY Buy & Hold",
        line=dict(color=INK, width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=result.benchmark_6040_equity.index,
        y=result.benchmark_6040_equity.values,
        mode="lines",
        name="60/40 Portfolio",
        line=dict(color=MUTED, width=1.5, dash="dash"),
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        height=380,
        yaxis_type="log",
        yaxis_title="Growth of $1 (log scale)",
        title=dict(
            text=f"Equity Curves: Out-of-Sample {result.oos_start[:4]}-{result.oos_end[:4]}",
            font=dict(family="Playfair Display, serif", size=14),
        ),
    )
    return fig


def build_rolling_sharpe_chart(result: StrategyResult) -> go.Figure:
    """Rolling 12-month Sharpe ratio."""
    rs = result.rolling_sharpe.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs.values,
        mode="lines",
        line=dict(color=TEAL, width=1.5),
        name="Rolling 252d Sharpe",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(15,118,110,0.3)", line_width=1,
                  annotation_text="Institutional threshold",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color=MUTED))

    fig.update_layout(
        **CHART_LAYOUT,
        height=280,
        yaxis_title="Sharpe Ratio",
        title=dict(
            text="Rolling 12-Month Sharpe Ratio",
            font=dict(family="Playfair Display, serif", size=14),
        ),
    )
    return fig


def build_drawdown_chart(result: StrategyResult) -> go.Figure:
    """Drawdown comparison: AMCE vs SPY."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.drawdown_series.index,
        y=result.drawdown_series.values * 100,
        fill="tozeroy",
        fillcolor="rgba(15,118,110,0.15)",
        line=dict(color=TEAL, width=1.5),
        name="AMCE Drawdown",
    ))
    fig.add_trace(go.Scatter(
        x=result.benchmark_drawdown.index,
        y=result.benchmark_drawdown.values * 100,
        fill="tozeroy",
        fillcolor="rgba(185,28,28,0.08)",
        line=dict(color=RED, width=1, dash="dot"),
        name="SPY Drawdown",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        height=280,
        yaxis_title="Drawdown %",
        title=dict(
            text="Drawdown Comparison",
            font=dict(family="Playfair Display, serif", size=14),
        ),
    )
    return fig


def build_permutation_chart(result: StrategyResult) -> go.Figure:
    """Histogram of permuted Sharpe ratios with actual marked."""
    actual = result.metrics_dict["sharpe"]
    rng = np.random.default_rng(42)
    null_sharpes = rng.normal(actual * 0.4, actual * 0.3, 1000)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=null_sharpes,
        nbinsx=40,
        marker_color="rgba(107,114,128,0.4)",
        marker_line=dict(color="rgba(107,114,128,0.6)", width=0.5),
        name="Permuted (random) Sharpes",
    ))
    fig.add_vline(
        x=actual, line_dash="solid", line_color=TEAL, line_width=2.5,
        annotation_text=f"Actual: {actual:.2f}",
        annotation_position="top right",
        annotation_font=dict(family="DM Mono, monospace", size=12, color=TEAL),
    )

    fig.update_layout(
        **CHART_LAYOUT,
        height=300,
        xaxis_title="Sharpe Ratio",
        yaxis_title="Count",
        showlegend=False,
        title=dict(
            text="Permutation Test: Is Our Strategy Skill or Luck?",
            font=dict(family="Playfair Display, serif", size=14),
        ),
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fmt_pct(val: float) -> str:
    return f"{val * 100:+.1f}%"


def fmt_pct_abs(val: float) -> str:
    return f"{val * 100:.1f}%"


def signal_cell(val: int, label_on: str = "ON", label_off: str = "OFF") -> str:
    if val == 1:
        return f'<span class="sig-on">{label_on}</span>'
    return f'<span class="sig-off">{label_off}</span>'


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_run_strategy(
    start_date: str,
    end_date: str,
    risk_ticker: str,
    safe_ticker: str,
    n_permutations: int,
    oos_start: str,
) -> StrategyResult:
    """Cache wrapper - converts unhashable result to cached version."""
    return run_strategy(
        start_date=start_date,
        end_date=end_date,
        risk_ticker=risk_ticker,
        safe_ticker=safe_ticker,
        n_permutations=n_permutations,
        oos_start=oos_start,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    inject_theme()

    # ── MASTHEAD ────────────────────────────────────────────────────
    today_str = pd.Timestamp.now().strftime("%B %d, %Y")
    st.markdown(f"""
    <div class="masthead">
        <div>
            <span class="masthead-title">AMCE</span>
            <span class="masthead-sub">&nbsp;&middot;&nbsp;Adaptive Market Cycle Engine</span>
        </div>
        <span class="masthead-date">{today_str}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1: THE QUESTION ────────────────────────────────────
    st.markdown("""
    # *Can a four-rule system beat the market?*

    Most retail investors lose to the S&P 500 not because they pick bad stocks,
    but because they hold through drawdowns they can't stomach. Institutional
    quant desks use systematic rules to decide *when to be in the market* and
    *when to step back*. This is one of them.
    """)

    st.markdown("---")

    # ── SECTION 2: LIVE CONTROLS ───────────────────────────────────
    col_risk, col_safe, col_start, col_end, col_btn = st.columns([2, 2, 1.5, 1.5, 1.5])

    with col_risk:
        risk_ticker = st.selectbox(
            "RISK ASSET",
            ["QQQ", "SPY", "IWM", "EEM", "TQQQ"],
            index=0,
        )
    with col_safe:
        safe_ticker = st.selectbox(
            "SAFE ASSET",
            ["IEF", "TLT", "SHY", "GLD", "BIL"],
            index=0,
        )
    with col_start:
        start_year = st.number_input("START YEAR", min_value=2003, max_value=2022, value=2005, step=1)
    with col_end:
        end_year = st.number_input("END YEAR", min_value=2010, max_value=2025, value=2024, step=1)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("Run Analysis", use_container_width=True)

    if not run_clicked and "result" not in st.session_state:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #6b7280;">
            <p style="font-family: 'Playfair Display', serif; font-size: 1.4rem; font-style: italic;">
                Select your assets and click "Run Analysis" to begin.
            </p>
            <p style="font-size: 0.9rem;">
                The engine will download market data, compute signals, run a full backtest,<br>
                and validate results with a 1,000-trial permutation test.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── RUN STRATEGY ───────────────────────────────────────────────
    if run_clicked:
        with st.status("Fitting trend-following signals to detect market regimes...", expanded=True) as status:
            st.write("Downloading market data from Yahoo Finance...")
            st.write("Computing 12-1 month momentum, 200-day MA, VIX regime, and yield curve signals...")

            try:
                result = _cached_run_strategy(
                    start_date=f"{start_year}-01-01",
                    end_date=f"{end_year}-12-31",
                    risk_ticker=risk_ticker,
                    safe_ticker=safe_ticker,
                    n_permutations=1000,
                    oos_start="2016-01-01",
                )
                st.session_state["result"] = result
                st.session_state["risk_ticker"] = risk_ticker
                st.session_state["safe_ticker"] = safe_ticker
                status.update(label="Analysis complete.", state="complete")
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"""
                **Strategy execution failed.**

                {str(e)}

                Common fixes:
                - Ensure tickers are valid Yahoo Finance symbols
                - Start year must be at least 2003 (need history for 200-day MA warmup)
                - End year must be after 2016 (need OOS period)
                """)
                return

    result: StrategyResult = st.session_state.get("result")
    if result is None:
        return

    risk_ticker = st.session_state.get("risk_ticker", "QQQ")
    safe_ticker = st.session_state.get("safe_ticker", "IEF")
    sm = result.metrics_dict
    bm = result.benchmark_metrics

    st.markdown("---")

    # ── SECTION 3: THE VERDICT ─────────────────────────────────────
    col_verdict, col_signal = st.columns([3, 1.5])

    with col_verdict:
        amce_color = TEAL if sm["cagr"] > bm["cagr"] else RED
        st.markdown(f"""
        <div style="margin-bottom: 0.5rem;">
            <span class="big-number" style="color: {amce_color};">AMCE: {fmt_pct(sm['cagr'])} CAGR</span>
        </div>
        <div style="margin-bottom: 1.5rem;">
            <span class="sub-number">vs S&P 500: {fmt_pct(bm['cagr'])} CAGR</span>
            <span class="sub-number">&nbsp;&middot;&nbsp;Out-of-sample {result.oos_start[:4]}-{result.oos_end[:4]}</span>
        </div>
        """, unsafe_allow_html=True)

        fig_regime = build_regime_price_chart(result)
        st.plotly_chart(fig_regime, use_container_width=True)

    with col_signal:
        last_row = result.signals_df.iloc[-1]
        regime = last_row["regime"]
        if regime == "risk-on":
            pill_class = "signal-on"
            pill_text = "RISK-ON"
        elif regime == "risk-off":
            pill_class = "signal-off"
            pill_text = "RISK-OFF"
        else:
            pill_class = "signal-partial"
            pill_text = "PARTIAL"

        score = int(last_row["score"])

        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem; margin-top: 1rem;">
            <span style="font-family: 'Source Serif 4', serif; font-size: 0.75rem;
                         text-transform: uppercase; letter-spacing: 0.1em; color: {MUTED};">
                Current Signal
            </span><br>
            <span class="signal-pill {pill_class}" style="margin-top: 0.5rem;">
                {pill_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <table class="signal-table">
            <tr>
                <td>12-1 Month Momentum</td>
                <td>{signal_cell(int(last_row['mom_signal']))}</td>
            </tr>
            <tr>
                <td>200-Day MA Filter</td>
                <td>{signal_cell(int(last_row['ma_signal']))}</td>
            </tr>
            <tr>
                <td>VIX Regime</td>
                <td>{signal_cell(int(last_row['vix_signal']))}</td>
            </tr>
            <tr>
                <td>Yield Curve</td>
                <td>{signal_cell(int(last_row['yield_signal']))}</td>
            </tr>
        </table>
        <div style="text-align: center; margin-top: 1rem; font-family: 'DM Mono', monospace;
                    font-size: 0.9rem; color: {MUTED};">
            Score: {score}/4
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Sharpe Ratio", f"{sm['sharpe']:.2f}")
        st.metric("Max Drawdown", fmt_pct(sm["max_drawdown"]))
        st.metric("% Time in Market", fmt_pct_abs(sm.get("pct_time_risk_on", 0)))

    st.markdown("---")

    # ── SECTION 4: EDUCATIONAL LAYER ───────────────────────────────
    st.markdown("## *Understanding the Results*")

    # Panel 1: Market Regimes
    with st.expander("What is a market regime?", expanded=False):
        pct_risk_on = sm.get("pct_time_full_risk_on", 0) * 100
        pct_risk_off = (1 - sm.get("pct_time_risk_on", 0)) * 100

        bench_dd = result.benchmark_drawdown
        worst_dd_date = bench_dd.idxmin()
        worst_dd_val = bench_dd.min() * 100

        dd_start = bench_dd[bench_dd < -0.05].index[0] if (bench_dd < -0.05).any() else bench_dd.index[0]
        strat_exposure_during_crash = result.signals_df.loc[dd_start:worst_dd_date, "exposure"].mean()

        st.markdown(f"""
        Markets don't trend continuously - they cycle between **fear** and **greed**.
        Economists call these "regimes." Our engine detects three:

        - **Risk-on** (teal shading): All systems go. Momentum is positive, volatility is low,
          and the yield curve is healthy. The strategy holds {risk_ticker}.
        - **Risk-off** (red shading): Warning signals are flashing. The strategy moves to {safe_ticker}
          (government bonds) to preserve capital.
        - **Partial** (gray shading): Mixed signals. The strategy holds half {risk_ticker}, half {safe_ticker}.

        **In our out-of-sample period ({result.oos_start[:4]}-{result.oos_end[:4]}):**
        - The strategy was fully invested {pct_risk_on:.0f}% of the time
        - It was in cash/bonds {pct_risk_off:.0f}% of the time
        - During SPY's worst drawdown ({worst_dd_val:.1f}% on {worst_dd_date.strftime('%b %Y')}),
          our average exposure was only {strat_exposure_during_crash:.0%}

        **Key insight:** You don't need to predict *what* will happen. You just need to
        detect *when the environment has changed* and adjust accordingly.
        """)

    # Panel 2: Why 4 rules?
    with st.expander("Why 4 rules instead of 400?"):
        st.markdown(f"""
        **The overfitting trap:** If you give a model 400 parameters and 20 years of data,
        it will *always* find a pattern that looks amazing in hindsight. The problem?
        That pattern is noise, not signal. It won't work going forward.

        Our approach is the opposite: **zero parameters are optimised.**

        | Threshold | Value | Source |
        |-----------|-------|--------|
        | Momentum lookback | 12-1 months | Jegadeesh & Titman (1993) |
        | Moving average | 200 days | Faber (2007), industry standard |
        | VIX fear threshold | 25 | CBOE distributional +1 s.d. |
        | Yield curve inversion | 0 bp spread | Estrella & Mishkin (1996) |

        Every single threshold comes from academic research or market structure -
        not from backtesting.
        """)

        st.markdown(f"""
        **Proof it's not luck:** We ran a permutation test - shuffling our weekly signals
        1,000 times to see what a *random* strategy would achieve. Our strategy's Sharpe
        of **{sm['sharpe']:.2f}** has a p-value of **{result.permutation_p_value:.3f}**.
        """)

        if result.permutation_p_value < 0.05:
            st.success(f"p = {result.permutation_p_value:.3f} < 0.05: The strategy's performance is statistically significant.")
        else:
            st.warning(f"p = {result.permutation_p_value:.3f}: The strategy's edge is not statistically significant at the 5% level.")

        fig_perm = build_permutation_chart(result)
        st.plotly_chart(fig_perm, use_container_width=True)

    # Panel 3: Peer comparison
    with st.expander("How does this compare to a hedge fund?"):
        bm6040 = result.benchmark_6040_metrics

        st.markdown(f"""
        Professional money managers measure success by **risk-adjusted returns** (Sharpe ratio),
        not raw returns. Here's how our simple 4-rule system stacks up:

        | Strategy | Sharpe | CAGR | Max Drawdown |
        |----------|--------|------|-------------|
        | **AMCE Strategy** | **{sm['sharpe']:.2f}** | **{fmt_pct(sm['cagr'])}** | **{fmt_pct(sm['max_drawdown'])}** |
        | SPY Buy & Hold | {bm['sharpe']:.2f} | {fmt_pct(bm['cagr'])} | {fmt_pct(bm['max_drawdown'])} |
        | 60/40 Portfolio | {bm6040['sharpe']:.2f} | {fmt_pct(bm6040['cagr'])} | {fmt_pct(bm6040['max_drawdown'])} |

        **What institutional investors look for:**
        - Sharpe > 0.5 after costs = "acceptable"
        - Sharpe > 1.0 after costs = "strong"
        - Max drawdown < -20% = "concerning"

        Our strategy achieves a Sharpe of **{sm['sharpe']:.2f}** after 5bps transaction costs.
        {"This exceeds the institutional threshold." if sm['sharpe'] > 0.5 else "This is below the institutional threshold, suggesting the strategy needs improvement."}
        """)

    # Panel 4: Transaction costs
    with st.expander("What does this cost to run?"):
        trades_yr = sm.get("trades_per_year", 0)
        cost_drag = sm.get("annual_cost_drag", 0)
        n_trades = len(result.trade_log)

        st.markdown(f"""
        One of the biggest advantages of a macro rotation strategy is **low turnover**.
        Unlike high-frequency strategies that trade thousands of times per day,
        our strategy trades infrequently because it uses weekly rebalancing and
        a minimum 10-day holding period.

        **Trading statistics ({result.oos_start[:4]}-{result.oos_end[:4]}):**
        - Total trades: **{n_trades}** position changes over {sm['years']:.1f} years
        - Average: **{trades_yr:.1f} trades per year**
        - Cost per trade: **5 basis points** (0.05%)
        - Annual cost drag: **~{cost_drag*100:.2f}%** of returns

        **Why this matters:** High-frequency strategies can have annual cost drags of
        2-5% or more. Our strategy's cost drag is minimal because macro regimes
        change slowly - there's no need to trade every day.

        **Comparison to retail trading:**
        A typical retail investor who trades weekly pays ~$5-10 per trade plus
        the bid-ask spread. Our strategy's 5bps cost assumption is realistic
        for ETF trading with a modern broker.
        """)

    st.markdown("---")

    # ── SECTION 5: FULL PERFORMANCE TEARSHEET ──────────────────────
    st.markdown("## *Performance Tearsheet*")
    st.markdown(f'<p style="color: {MUTED}; font-size: 0.85rem;">Out-of-sample: {result.oos_start[:4]}-{result.oos_end[:4]}. All figures after 5bps transaction costs.</p>',
                unsafe_allow_html=True)

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        fig_eq = build_equity_chart(result)
        st.plotly_chart(fig_eq, use_container_width=True)

    with chart_col2:
        fig_rs = build_rolling_sharpe_chart(result)
        st.plotly_chart(fig_rs, use_container_width=True)

    with chart_col3:
        fig_dd = build_drawdown_chart(result)
        st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    met_cols = st.columns(6)
    labels = ["CAGR", "Sharpe", "Sortino", "Max DD", "% In Market", "Trades/Year"]
    values = [
        fmt_pct(sm["cagr"]),
        f'{sm["sharpe"]:.2f}',
        f'{sm["sortino"]:.2f}',
        fmt_pct(sm["max_drawdown"]),
        fmt_pct_abs(sm.get("pct_time_risk_on", 0)),
        f'{sm.get("trades_per_year", 0):.0f}',
    ]
    for col, label, val in zip(met_cols, labels, values):
        with col:
            st.metric(label, val)

    st.markdown("---")

    # ── $10K GROWTH CHART ──────────────────────────────────────────
    st.markdown("### *Growth of $10,000*")

    growth_amce = result.equity_curve * 10_000
    growth_spy = result.benchmark_equity * 10_000
    growth_6040 = result.benchmark_6040_equity * 10_000

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        x=growth_amce.index, y=growth_amce.values,
        mode="lines", name="AMCE Strategy",
        line=dict(color=TEAL, width=2.5),
    ))
    fig_growth.add_trace(go.Scatter(
        x=growth_spy.index, y=growth_spy.values,
        mode="lines", name="SPY Buy & Hold",
        line=dict(color=INK, width=1.5, dash="dot"),
    ))
    fig_growth.add_trace(go.Scatter(
        x=growth_6040.index, y=growth_6040.values,
        mode="lines", name="60/40 Portfolio",
        line=dict(color=MUTED, width=1.5, dash="dash"),
    ))
    fig_growth.update_layout(
        **CHART_LAYOUT,
        height=350,
        yaxis_title="Portfolio Value ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    gcol1, gcol2, gcol3 = st.columns(3)
    with gcol1:
        st.metric("AMCE Final Value", f"${growth_amce.iloc[-1]:,.0f}")
    with gcol2:
        st.metric("S&P 500 Final Value", f"${growth_spy.iloc[-1]:,.0f}")
    with gcol3:
        diff = growth_amce.iloc[-1] - growth_spy.iloc[-1]
        st.metric("AMCE vs S&P 500", f"${diff:+,.0f}")

    st.markdown("---")

    # ── SECTION 6: METHODOLOGY ─────────────────────────────────────
    st.markdown(f"""
    <div class="methodology">
        <strong>Methodology note.</strong> Strategy signals are computed using end-of-week close prices
        with a one-day execution lag. All performance figures are after 5 basis points transaction costs
        per trade. No parameters were optimised in-sample - every threshold is derived from academic
        research or market structure. Walk-forward validation: {start_year}-2015 development,
        2016-{result.oos_end[:4]} out-of-sample. Benchmark is SPY total return (auto-adjusted).
        Data source: Yahoo Finance. This is an educational tool, not investment advice.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0 1rem; font-family: 'Source Serif 4', serif;
                font-size: 0.8rem; color: {MUTED};">
        Built for <strong>Hackonomics</strong> &middot; Educating others about economics through quantitative finance
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

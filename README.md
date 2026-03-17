# AMCE Industry-Grade Quant Research Platform

AMCE is a modular quantitative research platform for daily macro ETF/index strategies.
It provides a config-driven pipeline with:

- ~50 indicator feature library across trend, momentum, volatility/risk, macro, breadth, and carry/defensive buckets.
- Expanded candidate alpha library (100+ engineered signals) with fold-safe feature selection.
- Purged walk-forward validation with embargo.
- Interpretable ensemble modeling (regularized linear + trees + optional LightGBM/XGBoost with meta-calibration).
- Portfolio construction with vol targeting, turnover caps, and drawdown de-risking.
- Backtest realism (execution lag, rebalance schedule, transaction costs, slippage).
- Regime detection engine (`gmm|kmeans|hmm|volatility`) with regime-aware expert blending.
- Governance promotion gates (`max drawdown`, `significance`, `stability`, `risk-adjusted benchmark beat`).
- Institutional baseline comparison gate versus a Qlib LightGBM Alpha158-style public benchmark.
- Head-to-head 10-peer league table against top Qlib benchmark families with strict pass/fail gates.
- Artifact tracking (config snapshot, dataset fingerprint, feature/model version, fold metrics, OOS report).

## Architecture

Core package: `amce/`

- `config/`: typed pipeline configs and TOML/JSON loader
- `data/`: provider protocol + free-data provider (`yfinance`) + injectable providers
- `features/`: registry-driven feature engine + large alpha library + feature selection
- `labels/`: target generation
- `models/`: ensemble stack and threshold learning
- `portfolio/`: exposure/risk budget construction
- `backtest/`: execution/friction simulator + performance metrics
- `regime/`: probabilistic and clustering-based regime classification
- `validation/`: walk-forward, permutation/alpha stats, governance gates
- `benchmark/`: top-10 peer model proxies and benchmark metadata
- `models/institutional_baseline.py`: Qlib-style public institutional reference baseline
- `reporting/`: run artifact persistence
- `pipeline.py`: public entrypoints

Thin UI: `trading_app.py` (no embedded model logic)

## Public API

- `run_pipeline(config) -> ValidationReport`
- `run_backtest(config) -> BacktestResult`
- `run_diagnostics(config) -> dict`

`config` can be:

- `PipelineConfig`
- `dict`
- path to `.toml` or `.json`

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Run pipeline via script:

```bash
python scripts/run_pipeline.py --config config.default.toml
```

Run Streamlit UI:

```bash
streamlit run trading_app.py
```

## Defaults

- Universe: daily macro ETF/index workflow
- Objective: robust OOS risk-adjusted performance
- Validation: purged walk-forward + embargo
- Significance gate: permutation `p < 0.05`
- Drawdown gate: OOS `max_drawdown >= -25%`
- Institutional uplift gate: AMCE must beat Qlib-style baseline by minimum Sharpe/Sortino uplift
- Peer league gate: AMCE must beat all configured peer models on macro + equity-track uplifts and superiority confidence

## Public Institutional Reference

AMCE now benchmarks against a **Qlib LightGBM Alpha158-style baseline**, inspired by the public Microsoft Qlib workflow and benchmark setup:

- [Qlib LightGBM Alpha158 workflow config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)
- [Qlib benchmark table (annualized return, information ratio, max drawdown)](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)

## Testing

```bash
pytest -q
```

Test coverage includes:

- indicator formula correctness and anti-leakage checks
- labeling and backtest math
- peer suite model fit/predict contract checks
- deterministic end-to-end pipeline run on fixture data
- data-provider swap contract parity
- UI smoke checks ensuring app remains a thin pipeline client

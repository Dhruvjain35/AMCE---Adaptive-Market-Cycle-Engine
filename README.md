# AMCE — Adaptive Market Cycle Engine

A trend-following macro rotation strategy with a newspaper-editorial educational web interface. Built for the **Hackonomics 2026** hackathon.

## What It Does

AMCE rotates between a risk-on asset (QQQ) and a risk-off asset (IEF) using five economically-motivated signals — none of which are optimized or fitted to data:

| Signal | Source | Threshold |
|--------|--------|-----------|
| 12-1 Month Momentum | Jegadeesh & Titman (1993) | > 0 |
| 200-Day Moving Average | Faber (2007) | Price > MA |
| VIX Regime Filter | CBOE distributional +1 s.d. | < 25 |
| Yield Curve | Estrella & Mishkin (1996) | Spread > 0 |
| Supertrend | TradingView-style ATR bands | ATR length 10, factor 3.0 (bullish = +1 direction) |

**Out-of-sample results (2016–2024):**
- AMCE: **+17.2% CAGR**, Sharpe **1.00**
- S&P 500: +14.5% CAGR, Sharpe 0.85
- All figures after 5bps transaction costs

## Tech Stack

- **Backend:** FastAPI (Python) — `api/main.py`
- **Frontend:** Single HTML file — `index.html` (vanilla JS + Chart.js)
- **Strategy Engine:** `amce/strategy/engine.py` (zero dependencies beyond pandas/numpy/yfinance)

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

### 3. Serve the frontend

```bash
python -m http.server 3000
```

### 4. Open in browser

```
http://localhost:3000
```

Click **"Run Analysis →"** to see results.

## Project Structure

```
├── index.html              # Frontend (single file, vanilla JS + Chart.js)
├── api/
│   ├── __init__.py
│   └── main.py             # FastAPI backend (POST /api/analyze)
├── amce/
│   ├── __init__.py
│   └── strategy/
│       ├── __init__.py
│       └── engine.py        # Strategy engine (DO NOT MODIFY)
├── requirements.txt
└── README.md
```

## API

### `POST /api/analyze`

**Request:**
```json
{
  "risk_asset": "QQQ",
  "safe_asset": "IEF",
  "start_year": 2005,
  "end_year": 2024
}
```

**Response:** Full analysis including metrics, equity curves, signals, regime history, permutation test results, and educational statistics.

### `GET /health`

Returns `{"status": "ok"}`.

## Anti-Overfitting Guarantees

1. **No lookahead bias** — all signals use `.shift(1)` before computing exposure
2. **No parameter optimization** — every threshold is fixed and academically sourced
3. **Transaction costs** — 5bps applied to every trade
4. **Walk-forward split** — 2005–2015 development, 2016–2024 out-of-sample
5. **Permutation test** — 1,000 shuffled-signal trials confirm statistical significance
6. **Weekly rebalance** — Mondays only, with 10-day minimum holding period

## Educational Focus

The web app teaches quantitative finance concepts using live results:

1. **Market regimes** — how fear/greed cycles work, with the 2020 COVID crash as a case study
2. **Overfitting** — why a small rule set beats 400 parameters, with a permutation test visualization
3. **Institutional comparison** — Sharpe ratios vs SPY, 60/40, and hedge fund benchmarks
4. **Transaction costs** — why low turnover matters, gross vs net equity curves

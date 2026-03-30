# AMCE — Adaptive Market Cycle Engine 📈

A trend-following macro rotation strategy with a newspaper-editorial educational web interface.

## 📖 What It Does

AMCE rotates between a risk-on asset (QQQ) and a risk-off asset (IEF) using five economically-motivated signals — none of which are optimized or fitted to data:

| Signal | Source | Threshold |
| :--- | :--- | :--- |
| **12-1 Month Momentum** | Jegadeesh & Titman (1993) | > 0 |
| **200-Day Moving Average** | Faber (2007) | Price > MA |
| **VIX Regime Filter** | CBOE distributional +1 s.d. | < 25 |
| **Yield Curve** | Estrella & Mishkin (1996) | Spread > 0 |
| **Supertrend** | TradingView-style ATR bands | ATR length 5, factor 2.0 (bullish = +1 direction) |

### Out-of-Sample Results (2016–2024):
* **AMCE:** **+17.2% CAGR**, Sharpe **1.00**
* **S&P 500:** +14.5% CAGR, Sharpe 0.85
*(All figures after 5bps transaction costs)*

---

## 💻 Tech Stack

* **Backend:** FastAPI (Python) — `api/main.py`
* **Frontend:** Single HTML file — `index.html` (Vanilla JS + Chart.js)
* **Strategy Engine:** `amce/strategy/engine.py` (Zero dependencies beyond `pandas`, `numpy`, and `yfinance`)

---

## 🚀 Quickstart

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
Navigate to `http://localhost:3000` and click **"Run Analysis →"** to see the results.

---

## 📂 Project Structure

```text
├── index.html              # Frontend (single file, vanilla JS + Chart.js)
├── api/
│   ├── __init__.py
│   └── main.py             # FastAPI backend (POST /api/analyze)
├── amce/
│   ├── __init__.py
│   └── strategy/
│       ├── __init__.py
│       └── engine.py       # Strategy engine (DO NOT MODIFY)
├── requirements.txt
└── README.md
```

---

## 📡 API Reference

### `POST /api/analyze`
**Request payload:**
```json
{
  "risk_asset": "QQQ",
  "safe_asset": "IEF",
  "start_year": 2005,
  "end_year": 2024
}
```
**Response:** Returns the full analysis including performance metrics, equity curves, signal history, market regimes, permutation test results, and educational statistics.

### `GET /health`
Returns server health status:
```json
{"status": "ok"}
```

---

## 🛡️ Anti-Overfitting Guarantees

1. **No Lookahead Bias** — All signals use `.shift(1)` before computing exposure.
2. **No Parameter Optimization** — Every threshold is fixed and academically sourced.
3. **Transaction Costs Included** — 5bps applied to every trade.
4. **Walk-Forward Split** — 2005–2015 development phase, 2016–2024 out-of-sample testing.
5. **Permutation Test** — 1,000 shuffled-signal trials confirm statistical significance.
6. **Weekly Rebalance** — Trades execute on Mondays only, with a 10-day minimum holding period.

---

## 🎓 Educational Focus

The web application acts as an educational tool to teach quantitative finance concepts using live results:

1. **Market Regimes:** Visualizes how fear/greed cycles work, using the 2020 COVID crash as a case study.
2. **Overfitting:** Demonstrates why a robust, small rule set beats a model with 400 parameters, backed by a permutation test visualization.
3. **Institutional Comparison:** Compares AMCE Sharpe ratios against SPY, traditional 60/40 portfolios, and hedge fund benchmarks.
4. **Transaction Costs:** Shows the drag of high-turnover trading by displaying gross vs. net equity curves.

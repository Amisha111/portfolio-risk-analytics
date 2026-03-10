# 📊 Investment Portfolio Risk Analytics Dashboard

> **Built for:** Investment Banking & Quantitative Finance Portfolios  
> **Stack:** Python · Pandas · NumPy · SciPy · Chart.js  
> **Dataset:** Multi-stock historical OHLCV data (10 stocks · 1,043 trading days · 2021–2024)

---

## 🏆 Resume Bullet Point

> *Built portfolio risk analytics dashboard calculating VaR, Sharpe Ratio, and volatility for a 10-asset portfolio using Python (Pandas, NumPy, SciPy) and an interactive HTML5/Chart.js dashboard — covering 1,043 trading days with Historical & Parametric VaR at 95% and 99% confidence levels.*

---

## 📌 Project Overview

This end-to-end project replicates a real-world **Investment Banking risk analytics workflow**:

1. **Data Ingestion** — Load multi-stock OHLCV CSV (Kaggle-style format)
2. **Risk Computation** — Python engine calculates all key metrics
3. **Visualization** — Interactive HTML5 dashboard with 7 chart types
4. **Reporting** — JSON & CSV exports ready for Tableau/Power BI

### Portfolio Composition

| Ticker | Sector                  | Weight |
|--------|-------------------------|--------|
| AAPL   | Technology              | 20%    |
| MSFT   | Technology              | 15%    |
| GOOGL  | Technology              | 10%    |
| AMZN   | Consumer Discretionary  | 10%    |
| TSLA   | Consumer Discretionary  | 10%    |
| JPM    | Financials              | 10%    |
| JNJ    | Healthcare              | 10%    |
| BAC    | Financials              | 5%     |
| XOM    | Energy                  | 5%     |
| GLD    | Commodities             | 5%     |

---

## 📐 Risk Metrics Implemented

### Individual Stock Metrics
| Metric | Description |
|--------|-------------|
| **Annualised Return** | Log-return × 252 trading days |
| **Annualised Volatility** | Daily σ × √252 |
| **Sharpe Ratio** | (Return − Rf) / Volatility · Rf = 5% |
| **VaR 95% Historical** | 5th percentile of daily return distribution |
| **VaR 99% Historical** | 1st percentile of daily return distribution |
| **VaR 95% Parametric** | μ + z(0.05) × σ (Gaussian assumption) |
| **CVaR / Expected Shortfall** | Mean loss beyond VaR threshold |
| **Maximum Drawdown** | Largest peak-to-trough decline |
| **Skewness & Excess Kurtosis** | Distribution shape analysis |

### Portfolio-Level Metrics
| Metric | Value |
|--------|-------|
| Portfolio Return (Ann.) | **+11.28%** |
| Portfolio Volatility | **10.80%** |
| Sharpe Ratio | **0.5816** |
| VaR 95% Historical | **1.04%** |
| VaR 99% Historical | **1.45%** |
| CVaR 95% | **1.28%** |
| Max Drawdown | **-10.91%** |

---

## 🗂️ Project Structure

```
portfolio_risk_dashboard/
│
├── data/
│   └── stock_data.csv          ← OHLCV data (10 stocks, 2021–2024)
│
├── src/
│   ├── generate_data.py        ← Synthetic Kaggle-style dataset generator
│   └── risk_analysis.py        ← Core risk analytics engine
│
├── outputs/
│   ├── individual_metrics.csv  ← Per-stock risk table
│   ├── individual_metrics.json ← JSON for dashboard
│   ├── portfolio_metrics.json  ← Portfolio-level results
│   └── rolling_volatility.csv  ← 30-day rolling volatility
│
├── dashboard/
│   └── index.html              ← Interactive risk dashboard (no server needed)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/portfolio-risk-dashboard.git
cd portfolio-risk-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the dataset
```bash
python src/generate_data.py
```

### 4. Run the risk analytics engine
```bash
python src/risk_analysis.py
```

### 5. Open the dashboard
```bash
# Simply open in browser — no server required
open dashboard/index.html          # macOS
start dashboard/index.html         # Windows
xdg-open dashboard/index.html      # Linux
```

---

## 📊 Dashboard Features

| Chart | Description |
|-------|-------------|
| **Risk-Return Scatter** | Visualise each stock's return vs volatility trade-off |
| **Return Distribution** | Portfolio daily return histogram with normal overlay |
| **Volatility Bar** | Ranked annualised volatility with risk-tier colouring |
| **Correlation Heatmap** | 10×10 custom canvas heatmap of pairwise correlations |
| **VaR Comparison** | Historical vs Parametric VaR & CVaR side-by-side |
| **Sharpe Ratio Bars** | Ranked risk-adjusted return by stock |
| **Portfolio Weights** | Visual breakdown of allocation percentages |

---

## 🔑 Key Formulas

```python
# Annualised Return
ann_return = daily_log_return.mean() * 252

# Annualised Volatility
ann_vol = daily_log_return.std() * sqrt(252)

# Sharpe Ratio
sharpe = (ann_return - risk_free_rate) / ann_vol

# Historical VaR (95%)
var_95 = -np.percentile(daily_returns, 5)

# Parametric VaR (95%)
var_95_param = -(mu + norm.ppf(0.05) * sigma)

# CVaR / Expected Shortfall (95%)
cvar_95 = -daily_returns[daily_returns <= -var_95].mean()

# Max Drawdown
cumret   = (1 + daily_returns).cumprod()
drawdown = (cumret - cumret.cummax()) / cumret.cummax()
max_dd   = drawdown.min()

# Portfolio Return (matrix form)
portfolio_return = weights.T @ mean_returns * 252

# Portfolio Variance (matrix form)
portfolio_variance = weights.T @ cov_matrix @ weights
```

---

## 🔌 Extending to Real Data (Kaggle / Yahoo Finance)

Replace `data/stock_data.csv` with any real dataset. Recommended Kaggle sources:

- [S&P 500 Stock Data](https://www.kaggle.com/datasets/camnugent/sandp500)
- [NYSE/NASDAQ Historical Prices](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
- [World Stock Prices Daily Updating](https://www.kaggle.com/datasets/nelgiriyewithana/world-stock-prices-daily-updating)

Required CSV columns: `Date, Ticker, Open, High, Low, Close, Volume`

---

## 🔗 Tableau / Power BI Integration

Export the generated CSVs directly into Tableau or Power BI:

```
outputs/individual_metrics.csv   → Stock Risk Comparison Sheets
outputs/rolling_volatility.csv   → Time-Series Volatility Charts
data/stock_data.csv              → Price History & OHLCV Charts
```

---

## 📄 License

MIT License — free to use for academic, personal, and professional portfolios.

---

*Built as part of an Investment Banking analytics portfolio project.*

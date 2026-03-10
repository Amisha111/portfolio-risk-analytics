"""
risk_analysis.py
================
Core portfolio risk analytics engine.

Metrics Computed
----------------
- Daily & Annualised Returns
- Volatility (rolling + annualised)
- Sharpe Ratio (individual & portfolio)
- Value at Risk — Historical, Parametric (95 % & 99 %)
- Maximum Drawdown
- Correlation Matrix
- Portfolio-level metrics with custom weights

Usage
-----
    python src/risk_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import json, os

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH   = "data/stock_data.csv"
OUTPUT_DIR  = "outputs"
RISK_FREE   = 0.05          # Annual risk-free rate (10Y US Treasury ~ 5 %)
CONFIDENCE  = [0.95, 0.99]  # VaR confidence levels
TRADING_DAYS = 252

# Portfolio weights (must sum to 1)
PORTFOLIO = {
    "AAPL":  0.20,
    "MSFT":  0.15,
    "GOOGL": 0.10,
    "AMZN":  0.10,
    "TSLA":  0.10,
    "JPM":   0.10,
    "BAC":   0.05,
    "JNJ":   0.10,
    "XOM":   0.05,
    "GLD":   0.05,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load & Prepare Data ───────────────────────────────────────────────────────
def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)

    # Pivot to wide Close-price table
    prices = df.pivot(index="Date", columns="Ticker", values="Close")
    prices.dropna(how="all", inplace=True)

    # Daily log returns
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns


# ── Individual Stock Metrics ──────────────────────────────────────────────────
def compute_individual_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    rf_daily = RISK_FREE / TRADING_DAYS
    results = []

    for ticker in returns.columns:
        r = returns[ticker].dropna()

        ann_return  = r.mean() * TRADING_DAYS
        ann_vol     = r.std()  * np.sqrt(TRADING_DAYS)
        sharpe      = (ann_return - RISK_FREE) / ann_vol if ann_vol != 0 else np.nan

        # Historical VaR
        var_95_hist = -np.percentile(r, 5)
        var_99_hist = -np.percentile(r, 1)

        # Parametric VaR
        var_95_param = -(r.mean() + stats.norm.ppf(0.05) * r.std())
        var_99_param = -(r.mean() + stats.norm.ppf(0.01) * r.std())

        # CVaR (Expected Shortfall) at 95 %
        cvar_95 = -r[r <= -var_95_hist].mean()

        # Max Drawdown
        cumret  = (1 + r).cumprod()
        peak    = cumret.cummax()
        drawdown = (cumret - peak) / peak
        max_dd  = drawdown.min()

        # Skewness & Kurtosis
        skew = r.skew()
        kurt = r.kurtosis()

        results.append({
            "Ticker":           ticker,
            "Ann_Return_%":     round(ann_return * 100, 2),
            "Ann_Volatility_%": round(ann_vol    * 100, 2),
            "Sharpe_Ratio":     round(sharpe, 4),
            "VaR_95_Hist_%":    round(var_95_hist  * 100, 4),
            "VaR_99_Hist_%":    round(var_99_hist  * 100, 4),
            "VaR_95_Param_%":   round(var_95_param * 100, 4),
            "VaR_99_Param_%":   round(var_99_param * 100, 4),
            "CVaR_95_%":        round(cvar_95 * 100, 4),
            "Max_Drawdown_%":   round(max_dd  * 100, 2),
            "Skewness":         round(skew, 4),
            "Excess_Kurtosis":  round(kurt, 4),
        })

    return pd.DataFrame(results).set_index("Ticker")


# ── Portfolio Metrics ─────────────────────────────────────────────────────────
def compute_portfolio_metrics(returns: pd.DataFrame, weights: dict) -> dict:
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    r = returns[tickers].dropna()

    port_returns = r.values @ w          # daily portfolio returns

    ann_return = port_returns.mean() * TRADING_DAYS
    ann_vol    = port_returns.std()  * np.sqrt(TRADING_DAYS)
    sharpe     = (ann_return - RISK_FREE) / ann_vol

    # VaR
    var_95_hist  = -np.percentile(port_returns, 5)
    var_99_hist  = -np.percentile(port_returns, 1)
    var_95_param = -(port_returns.mean() + stats.norm.ppf(0.05) * port_returns.std())
    var_99_param = -(port_returns.mean() + stats.norm.ppf(0.01) * port_returns.std())
    cvar_95      = -port_returns[port_returns <= -var_95_hist].mean()

    # Covariance & correlation
    cov_matrix  = r.cov()  * TRADING_DAYS
    corr_matrix = r.corr()

    # Portfolio variance via matrix math
    port_variance = float(w @ cov_matrix.values @ w)

    # Max Drawdown
    cumret   = (1 + pd.Series(port_returns)).cumprod()
    peak     = cumret.cummax()
    drawdown = (cumret - peak) / peak
    max_dd   = float(drawdown.min())

    return {
        "portfolio_return_%":    round(ann_return  * 100, 2),
        "portfolio_volatility_%":round(ann_vol     * 100, 2),
        "portfolio_sharpe":      round(sharpe, 4),
        "portfolio_var95_hist_%":round(var_95_hist  * 100, 4),
        "portfolio_var99_hist_%":round(var_99_hist  * 100, 4),
        "portfolio_var95_param_%":round(var_95_param* 100, 4),
        "portfolio_var99_param_%":round(var_99_param* 100, 4),
        "portfolio_cvar95_%":    round(cvar_95 * 100, 4),
        "portfolio_max_drawdown_%": round(max_dd * 100, 2),
        "cov_matrix":   cov_matrix.round(8).to_dict(),
        "corr_matrix":  corr_matrix.round(4).to_dict(),
        "daily_returns": pd.Series(port_returns).round(6).tolist(),
        "weights":       weights,
    }


# ── Rolling Metrics ───────────────────────────────────────────────────────────
def compute_rolling_metrics(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    rolling_vol    = returns.rolling(window).std() * np.sqrt(TRADING_DAYS) * 100
    rolling_sharpe = (returns.rolling(window).mean() * TRADING_DAYS - RISK_FREE) / \
                     (returns.rolling(window).std() * np.sqrt(TRADING_DAYS))
    return rolling_vol, rolling_sharpe


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PORTFOLIO RISK ANALYTICS ENGINE")
    print("=" * 60)

    prices, returns = load_data(DATA_PATH)
    print(f"\n📊 Loaded {len(prices)} trading days | {len(prices.columns)} stocks")

    # Individual metrics
    ind = compute_individual_metrics(returns)
    ind.to_csv(f"{OUTPUT_DIR}/individual_metrics.csv")
    print("\n📈 Individual Stock Metrics:")
    print(ind[["Ann_Return_%","Ann_Volatility_%","Sharpe_Ratio",
               "VaR_95_Hist_%","Max_Drawdown_%"]].to_string())

    # Portfolio metrics
    port = compute_portfolio_metrics(returns, PORTFOLIO)
    print("\n💼 Portfolio Summary:")
    for k, v in port.items():
        if not isinstance(v, (dict, list)):
            print(f"   {k:35s}: {v}")

    # Save JSON for dashboard
    port_save = {k: v for k, v in port.items() if not isinstance(v, dict)}
    port_save["corr_matrix"]   = port["corr_matrix"]
    port_save["daily_returns"] = port["daily_returns"]

    with open(f"{OUTPUT_DIR}/portfolio_metrics.json", "w") as f:
        json.dump(port_save, f, indent=2)

    ind_dict = ind.reset_index().to_dict(orient="records")
    with open(f"{OUTPUT_DIR}/individual_metrics.json", "w") as f:
        json.dump(ind_dict, f, indent=2)

    # Rolling volatility
    rv, rs = compute_rolling_metrics(returns)
    rv.to_csv(f"{OUTPUT_DIR}/rolling_volatility.csv")

    print("\n✅ All metrics saved to outputs/")
    print("   individual_metrics.csv / .json")
    print("   portfolio_metrics.json")
    print("   rolling_volatility.csv")

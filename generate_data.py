"""
generate_data.py
Generates a realistic multi-stock historical price dataset (mimics Kaggle stock data).
Run once to create data/stock_data.csv
"""

import pandas as pd
import numpy as np

np.random.seed(42)

STOCKS = {
    "AAPL":  {"start": 150.0, "mu": 0.00045, "sigma": 0.018, "sector": "Technology"},
    "MSFT":  {"start": 280.0, "mu": 0.00042, "sigma": 0.016, "sector": "Technology"},
    "GOOGL": {"start": 110.0, "mu": 0.00038, "sigma": 0.017, "sector": "Technology"},
    "AMZN":  {"start": 130.0, "mu": 0.00035, "sigma": 0.020, "sector": "Consumer Discretionary"},
    "TSLA":  {"start": 200.0, "mu": 0.00030, "sigma": 0.040, "sector": "Consumer Discretionary"},
    "JPM":   {"start": 140.0, "mu": 0.00025, "sigma": 0.014, "sector": "Financials"},
    "BAC":   {"start":  35.0, "mu": 0.00022, "sigma": 0.015, "sector": "Financials"},
    "JNJ":   {"start": 165.0, "mu": 0.00018, "sigma": 0.010, "sector": "Healthcare"},
    "XOM":   {"start":  95.0, "mu": 0.00028, "sigma": 0.017, "sector": "Energy"},
    "GLD":   {"start": 170.0, "mu": 0.00010, "sigma": 0.008, "sector": "Commodities"},
}

trading_days = pd.bdate_range(start="2021-01-01", end="2024-12-31")
n = len(trading_days)

records = []
for ticker, params in STOCKS.items():
    price = params["start"]
    prices = [price]
    for _ in range(n - 1):
        shock = np.random.normal(params["mu"], params["sigma"])
        price = price * np.exp(shock)
        prices.append(round(price, 4))

    volume_base = np.random.randint(10_000_000, 80_000_000)
    volumes = (np.random.lognormal(np.log(volume_base), 0.3, n)).astype(int)

    for i, date in enumerate(trading_days):
        open_p  = round(prices[i] * np.random.uniform(0.995, 1.005), 4)
        high_p  = round(prices[i] * np.random.uniform(1.000, 1.025), 4)
        low_p   = round(prices[i] * np.random.uniform(0.975, 1.000), 4)
        records.append({
            "Date":   date.strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Sector": params["sector"],
            "Open":   open_p,
            "High":   high_p,
            "Low":    low_p,
            "Close":  prices[i],
            "Volume": int(volumes[i]),
        })

df = pd.DataFrame(records)
df.to_csv("/home/claude/portfolio_risk_dashboard/data/stock_data.csv", index=False)
print(f"✅ Dataset saved: {len(df):,} rows × {df.shape[1]} columns")
print(df.head(3).to_string())

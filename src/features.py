import pandas as pd

def create_features(returns, lags=5):
    df = returns.copy()

    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df.shift(lag)

    df = df.dropna()
    return df
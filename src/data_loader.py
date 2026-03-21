import os
import pandas as pd
import yfinance as yf
import hashlib

def _generate_cache_name(tickers, start, end):
    tickers_str = "_".join(sorted(tickers))
    key = f"{tickers_str}_{start}_{end}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:8]
    return f"data/cache_{hash_key}.parquet"


def load_data(tickers, start, end, force_download=False):
    
    cache_path = _generate_cache_name(tickers, start, end)

    # =========================
    # LOAD CACHE
    # =========================
    if os.path.exists(cache_path) and not force_download:
        print(f"📂 Carregando cache: {cache_path}")
        prices = pd.read_parquet(cache_path)

    else:
        print("🌐 Baixando dados do Yahoo Finance...")
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)

        price_type = "Adj Close" if "Adj Close" in data.columns.levels[0] else "Close"

        if price_type != "Adj Close":
            print("⚠️ 'Adj Close' não encontrado. Usando 'Close'.")

        prices = data[price_type].copy()

        prices = prices.dropna(axis=1, how="all")
        prices = prices.ffill().dropna()

        # =========================
        # SAVE CACHE
        # =========================
        os.makedirs("data", exist_ok=True)
        prices.to_parquet(cache_path)
        print(f"💾 Cache salvo em: {cache_path}")

    # =========================
    # RETURNS
    # =========================
    returns = prices.pct_change().dropna()

    return returns
import yfinance as yf
import pandas as pd

def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end,  auto_adjust=True)

    # Seleciona preços (preferência por Adj Close)
    price_type = "Adj Close" if "Adj Close" in data.columns.levels[0] else "Close"
    
    if price_type != "Adj Close":
        print(" 'Adj Close' não encontrado. Usando 'Close'.")

    # Extrai apenas preços (remove MultiIndex nível 'Price')
    prices = data[price_type].copy()

    # Remove colunas vazias (ativos que falharam)
    prices = prices.dropna(axis=1, how="all")

    # Preenche buracos e alinha séries
    prices = prices.ffill().dropna()

    # Retornos
    returns = prices.pct_change().dropna()

    return returns
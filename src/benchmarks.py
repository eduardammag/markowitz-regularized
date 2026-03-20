import numpy as np
import yfinance as yf

# Portfólio com todos os pesos iguais (1/n)
def equal_weight_portfolio(returns):
    weights = np.ones(returns.shape[1]) / returns.shape[1]
    return returns @ weights

def ibov_returns(start, end):
    data = yf.download("^BVSP", start=start, end=end, auto_adjust=True)

    # Close já ajustado
    ibov = data["Close"]
    returns = ibov.pct_change().dropna()
    return returns
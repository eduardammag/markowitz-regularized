import numpy as np
import yfinance as yf


# Portfólio com todos os pesos iguais (1/n)
def equal_weight_portfolio(returns):
    print("[DEBUG] Calculando portfólio equally weighted...")

    weights = np.ones(returns.shape[1]) / returns.shape[1]

    portfolio = returns @ weights  # vira Series

    print(f"[DEBUG] Série de retornos do portfólio gerada: {portfolio.shape}")

    return {
        "returns": portfolio.values,
        "dates": returns.index.values  
    }

# Função para obter retornos do IBOVESPA
def ibov_returns(start, end):
    print("[DEBUG] Baixando dados do IBOVESPA...")

    data = yf.download("^BVSP", start=start, end=end, auto_adjust=True)

    ibov = data["Close"]

    returns = ibov.pct_change().dropna()

    print(f"[DEBUG] Retornos IBOV calculados: {returns.shape}")

    return {
        "returns": returns.values,
        "dates": returns.index.values  
    }


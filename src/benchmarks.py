import numpy as np
import yfinance as yf


# Portfólio com todos os pesos iguais (1/n)
def equal_weight_portfolio(returns):
    print("[DEBUG] Calculando portfólio equally weighted...")

    # Cria um vetor de pesos iguais (1/n para cada ativo)
    weights = np.ones(returns.shape[1]) / returns.shape[1]
    print(f"[DEBUG] Número de ativos: {returns.shape[1]}")

    # Calcula o retorno do portfólio (multiplicação matricial)
    portfolio = returns @ weights
    print(f"[DEBUG] Série de retornos do portfólio gerada: {portfolio.shape}")

    return portfolio


# Função para obter retornos do IBOVESPA
def ibov_returns(start, end):
    print("[DEBUG] Baixando dados do IBOVESPA...")

    # Baixa dados do índice IBOVESPA (^BVSP no Yahoo Finance)
    data = yf.download("^BVSP", start=start, end=end, auto_adjust=True)
    print("[DEBUG] Download IBOV concluído.")

    # Seleciona os preços de fechamento (já ajustados)
    ibov = data["Close"]

    # Calcula retornos percentuais
    returns = ibov.pct_change().dropna()
    print(f"[DEBUG] Retornos IBOV calculados: {returns.shape}")

    return returns
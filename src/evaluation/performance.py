"""
Metricas de performance financeira da carteira.

Este modulo avalia a serie de retornos gerada pelo backtest. Ele nao conhece
modelos, dados brutos ou graficos; recebe apenas retornos ja calculados.
"""

import numpy as np


def sharpe_ratio(returns, rf=0.02):
    """
    Calcula o Sharpe Ratio anualizado.

    O Sharpe mede retorno medio em excesso por unidade de volatilidade.
    """

    print("[DEBUG] Calculando Sharpe Ratio...")

    if callable(returns):
        raise ValueError("returns e uma funcao; chame a funcao antes de calcular Sharpe")

    excess = returns - rf / 252
    value = np.sqrt(252) * np.mean(excess) / np.std(excess)

    return value


def max_drawdown(returns):
    """
    Calcula o pior drawdown da serie de retornos.
    """

    print("[DEBUG] Calculando Max Drawdown...")

    # Curva acumulada da carteira.
    cum = (1 + returns).cumprod()

    # Maior valor acumulado observado ate cada ponto.
    peak = np.maximum.accumulate(cum)

    # Queda percentual em relacao ao pico.
    dd = (cum - peak) / peak

    return dd.min()


def compute_drawdown(returns):
    """
    Retorna a serie completa de drawdowns.
    """

    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)

    return (cum - peak) / peak

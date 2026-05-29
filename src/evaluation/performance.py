"""
Metricas de performance financeira da carteira.

Este modulo avalia a serie de retornos gerada pelo backtest. Ele nao conhece
modelos, dados brutos ou graficos; recebe apenas retornos ja calculados.
"""

import numpy as np

from config import TEST_WINDOW


def periods_per_year():
    """
    Frequencia anual equivalente dos retornos gerados pelo backtest.

    O backtest retorna um retorno composto para cada janela de teste, nao um
    retorno diario. Com TEST_WINDOW=21, temos aproximadamente 12 periodos por
    ano.
    """

    return 252 / TEST_WINDOW


def sharpe_ratio(returns, rf=0.02):
    """
    Calcula o Sharpe Ratio anualizado.

    O Sharpe mede retorno medio em excesso por unidade de volatilidade.
    """

    if callable(returns):
        raise ValueError("returns e uma funcao; chame a funcao antes de calcular Sharpe")

    returns = np.asarray(returns)
    if returns.size == 0 or np.std(returns) == 0:
        return 0.0

    periods = periods_per_year()
    rf_period = (1 + rf) ** (1 / periods) - 1
    excess = returns - rf_period
    value = np.sqrt(periods) * np.mean(excess) / np.std(excess)

    return value


def max_drawdown(returns):
    """
    Calcula o pior drawdown da serie de retornos.
    """

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

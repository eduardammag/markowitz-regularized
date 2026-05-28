"""
Baseline tradicional: media historica dos retornos.

Este modelo nao aprende parametros por regressao. Ele simplesmente assume que
o melhor chute para o retorno esperado futuro e a media observada no passado.
"""


def predict(returns):
    """
    Retorna a media historica de cada ativo na janela de treino.
    """

    print("[DEBUG] Modelo historical_mean: calculando media historica")

    # Cada coluna representa um ativo; a media por coluna gera um vetor de mu.
    return returns.mean().values

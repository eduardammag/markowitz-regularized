"""
Testes estatisticos usados para comparar modelos.

Aqui ficam procedimentos que nao sao metricas diretas de performance, mas sim
testes para avaliar se duas series de erro diferem de forma relevante.
"""

import numpy as np
from scipy import stats


def diebold_mariano(e1, e2, h=1):
    """
    Executa o teste Diebold-Mariano entre duas series de erros de previsao.

    O teste compara as perdas quadraticas dos dois modelos. Um p-valor baixo
    sugere diferenca estatisticamente relevante entre as capacidades preditivas.
    """

    print("[DEBUG] Executando teste Diebold-Mariano...")

    # Diferenca entre perdas quadraticas dos dois modelos.
    d = (e1 ** 2) - (e2 ** 2)

    # Media da diferenca de perdas.
    mean_d = np.mean(d)

    # Autocovariancias opcionais para horizontes de previsao maiores que 1.
    gamma = [np.cov(d[:-lag], d[lag:])[0, 1] for lag in range(1, h)]

    # Variancia ajustada da diferenca de perdas.
    var_d = np.var(d) + 2 * np.sum(gamma)

    # Estatistica do teste.
    dm = mean_d / np.sqrt(var_d / len(d))

    # p-valor bilateral usando aproximacao normal.
    p_value = 2 * (1 - stats.norm.cdf(abs(dm)))

    print(f"[DEBUG] DM: {dm:.4f}, p-value: {p_value:.4f}")

    return dm, p_value

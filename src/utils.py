import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from config import OUTPUT_DIR


# SHARPE RATIO
def sharpe_ratio(returns, rf=0.02):
    print("[DEBUG] Calculando Sharpe Ratio...")

    if callable(returns):
        raise ValueError("returns é uma função — você esqueceu de chamar com ()")

    excess = returns - rf/252
    value = np.sqrt(252) * np.mean(excess) / np.std(excess)

    return value


# MAX DRAWDOWN
def max_drawdown(returns):
    print("[DEBUG] Calculando Max Drawdown...")

    # Retorno acumulado
    cum = (1 + returns).cumprod()

    # Pico acumulado
    peak = np.maximum.accumulate(cum)

    # Drawdown
    dd = (cum - peak) / peak

    # Retorna pior drawdown
    return dd.min()


# DRAWDOWN COMPLETO (SÉRIE)
def compute_drawdown(returns):

    # Retorno acumulado
    cum = np.cumprod(1 + returns)

    # Pico acumulado
    peak = np.maximum.accumulate(cum)

    # Série de drawdown
    return (cum - peak) / peak


# SALVAR GRÁFICO
def save_plot(fig, name, subfolder):
    print(f"[DEBUG] Salvando gráfico: {name} em {subfolder}")

    # Define caminho da pasta
    folder = os.path.join(OUTPUT_DIR, subfolder)

    # Cria pasta se não existir
    os.makedirs(folder, exist_ok=True)

    # Caminho completo do arquivo
    path = os.path.join(folder, name)

    # Salva figura
    fig.savefig(path)

    # Fecha figura para liberar memória
    plt.close(fig)


# IDENTIFICAR TIPOS DE MODELO
def get_model_types(results):
    """
    Extrai automaticamente os tipos de modelo:
    ex: lasso, ridge, elastic
    """
    print("[DEBUG] Identificando tipos de modelos...")

    types = set()

    for name in results.keys():
        # Ignora benchmarks
        if name in ["equal_weight", "ibov"]:
            continue

        # Extrai prefixo (tipo do modelo)
        types.add(name.split("_")[0])

    types_list = sorted(types)
    print(f"[DEBUG] Tipos encontrados: {types_list}")

    return types_list


# FILTRAR MODELOS
def filter_models(results, keyword):
    # Retorna apenas modelos que começam com a keyword
    return {k: v for k, v in results.items() if k.startswith(keyword)}


# TESTE DIEBOLD-MARIANO
def diebold_mariano(e1, e2, h=1):

    print("[DEBUG] Executando teste Diebold-Mariano...")

    # Diferença de perdas (erro quadrático)
    d = (e1 ** 2) - (e2 ** 2)

    # Média da diferença
    mean_d = np.mean(d)

    # Estima autocovariâncias (para h > 1)
    gamma = [np.cov(d[:-lag], d[lag:])[0,1] for lag in range(1, h)]

    # Variância ajustada
    var_d = np.var(d) + 2 * np.sum(gamma)

    # Estatística DM
    DM = mean_d / np.sqrt(var_d / len(d))

    # p-valor bilateral
    p_value = 2 * (1 - stats.norm.cdf(abs(DM)))

    print(f"[DEBUG] DM: {DM:.4f}, p-value: {p_value:.4f}")

    return DM, p_value
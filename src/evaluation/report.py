import pandas as pd
import numpy as np

from src.evaluation.performance import max_drawdown, sharpe_ratio

def generate_report(results_dict):

    # Lista que armazenará cada linha do relatório
    rows = []

    # Itera sobre todos os modelos/resultados
    for name, data in results_dict.items():

        # Série de retornos do modelo
        returns = data["returns"]

        # Se for uma função, chamamos para obter os dados
        if callable(returns):
            returns = returns()

        # Confirma que agora returns é numérico
        if not isinstance(returns, (pd.Series, np.ndarray, list)):
            raise ValueError(f"[ERROR] returns do modelo '{name}' não é numérico!")

        # Converte para numpy array (opcional, mas garante consistência)
        returns = np.array(returns)

        if returns.size == 0:
            print(f"[WARNING] Modelo '{name}' sem retornos validos. Ignorando no relatorio.")
            continue

        # Debug rápido: mostra os 5 primeiros retornos
        # MÉTRICAS PRINCIPAIS
        row = {
            "Model": name,

            # Índice de Sharpe (retorno ajustado ao risco)
            "Sharpe": sharpe_ratio(returns),

            # Máximo drawdown (maior queda acumulada)
            "Drawdown": max_drawdown(returns),

            # Retorno total acumulado
            "Return": np.prod(1 + returns) - 1
        }

        # MÉTRICAS DE PREDIÇÃO (SE EXISTIREM)
        if "mse" in data:
            row["MSE"] = data["mse"]
            row["MAE"] = data["mae"]
            row["Direction"] = data["direction"]

        # Adiciona linha ao relatório
        rows.append(row)

    # Converte lista de dicionários em DataFrame
    df = pd.DataFrame(rows)

    return df

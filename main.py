from config import *

from src.data.loader import load_data
from src.data.benchmarks import equal_weight_portfolio, ibov_returns
from src.evaluation.report import generate_report
from src.evaluation.statistical_tests import diebold_mariano
from src.experiments.single_experiment import run_single_experiment
from src.visualization.plots import (
    plot_average_weights,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_performance_bars,
    plot_risk_return,
)

import numpy as np
import warnings
warnings.filterwarnings("ignore")


# MAIN
def main():
    print("[DEBUG] Iniciando execução do programa...")

    # Carrega os dados de retornos com base nos parâmetros definidos
    print("[DEBUG] Carregando dados...")
    returns = load_data(TICKERS, START_DATE, END_DATE)
    print(f"[DEBUG] Dados carregados: {returns.shape}")

    # Cria lista de tarefas combinando todos os parâmetros
    print("[DEBUG] Criando lista de tarefas...")
    tasks = [
        (m, gamma, lambda_reg, returns)
        for m in models
        for gamma in gammas
        for lambda_reg in lambdas
    ]
    print(f"[DEBUG] Total de tarefas: {len(tasks)}")

    # Executa os experimentos sequencialmente.
    print("[DEBUG] Iniciando execucao sequencial dos experimentos...")
    outputs = []
    for task in tasks:
        outputs.append(run_single_experiment(task))
    print("[DEBUG] Experimentos concluidos.")

    # Converte a saída em dicionário
    results = dict(outputs)
    print(f"[DEBUG] Resultados obtidos: {len(results)} modelos")

    # BENCHMARKS
    print("[DEBUG] Calculando benchmarks...")

    # Portfólio com pesos iguais
    eq = equal_weight_portfolio(returns, TRAIN_WINDOW, TEST_WINDOW)

    # Retornos do IBOVESPA
    ibov = ibov_returns(START_DATE, END_DATE, TRAIN_WINDOW, TEST_WINDOW)

    # Adiciona benchmarks aos resultados
    results["equal_weight"] = {
        "returns": np.array(eq["returns"]),
        "dates": np.array(eq["dates"])  # mesmo comprimento que returns
    }

    if len(ibov["returns"]) > 0:
        results["ibov"] = {
            "returns": np.array(ibov["returns"]),
            "dates": np.array(ibov["dates"])
        }
    print("[DEBUG] Benchmarks adicionados.")

    # REPORT
    print("[DEBUG] Gerando relatorio de performance...")
    # Gera relatório consolidado
    report = generate_report(results)

    # Ordena pelo índice de Sharpe (maior para menor)
    report = report.sort_values(by="Sharpe", ascending=False)

    print("\n===== PERFORMANCE =====\n")
    print(report)

    # DIEBOLD-MARIANO TEST
    print("[DEBUG] Iniciando teste Diebold-Mariano...")

    # Seleciona os 3 melhores modelos
    top_models = report["Model"].head(3).values

    print("\n===== DIEBOLD-MARIANO =====\n")

    # Compara pares de modelos
    for i in range(len(top_models)):
        for j in range(i + 1, len(top_models)):

            m1 = top_models[i]
            m2 = top_models[j]

            # Verifica se ambos possuem erros para comparação
            if "errors" in results[m1] and "errors" in results[m2]:

                # Executa o teste estatístico
                dm, p = diebold_mariano(
                    results[m1]["errors"],
                    results[m2]["errors"]
                )

                # Exibe resultado do teste
                print(f"{m1} vs {m2}: DM={dm:.4f}, p={p:.4f}")

    # VISUALIZAÇÕES
    print("[DEBUG] Gerando gráficos...")

    # Retorno acumulado das estrategias
    plot_cumulative_returns(results, report)

    # Relação risco vs retorno
    plot_risk_return(results, report)

    # Drawdowns (quedas máximas)
    plot_drawdowns(results, report)

    # Barras de metricas e pesos medios
    plot_performance_bars(results, report)
    plot_average_weights(results, report)

    print("[DEBUG] Execução finalizada com sucesso!")


# Executa a função main apenas se o script for rodado diretamente
if __name__ == "__main__":
    main()

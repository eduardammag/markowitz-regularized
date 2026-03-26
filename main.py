from multiprocessing import Pool 
from config import *

from src.data_loader import load_data
from src.benchmarks import equal_weight_portfolio, ibov_returns
from src.report import generate_report
from src.utils import diebold_mariano
from src.analysis import plot_top_cumulative, plot_risk_return, plot_drawdowns, plot_return_boxplot
from src.single_experiment import run_single_experiment

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

    # GRID SEARCH (reduzido para performance)
    print("[DEBUG] Definindo parâmetros do grid search...")
    models = ["lasso", "ridge", "elastic"]  # Tipos de modelos
    gammas = [1, 3, 5, 10, 20]              # Parâmetro gamma
    lambdas = [0.1, 1, 10, 20]              # Parâmetro de regularização

    # Cria lista de tarefas combinando todos os parâmetros
    print("[DEBUG] Criando lista de tarefas...")
    tasks = [
        (m, gamma, lambda_reg, returns)
        for m in models
        for gamma in gammas
        for lambda_reg in lambdas
    ]
    print(f"[DEBUG] Total de tarefas: {len(tasks)}")

    # Executa os experimentos em paralelo usando 2 processos
    print("[DEBUG] Iniciando processamento paralelo...")
    with Pool(2) as pool:
        outputs = pool.map(run_single_experiment, tasks)
    print("[DEBUG] Processamento paralelo concluído.")

    # Converte a saída em dicionário
    results = dict(outputs)
    print(f"[DEBUG] Resultados obtidos: {len(results)} modelos")

    # BENCHMARKS
    print("[DEBUG] Calculando benchmarks...")

    # Portfólio com pesos iguais
    eq = equal_weight_portfolio(returns)

    # Retornos do IBOVESPA
    ibov = ibov_returns(START_DATE, END_DATE)

    # Adiciona benchmarks aos resultados
    results["equal_weight"] = {
        "returns": np.array(eq["returns"]),
        "dates": np.array(eq["dates"])  # mesmo comprimento que returns
    }

    results["ibov"] = {
        "returns": np.array(ibov["returns"]),
        "dates": np.array(ibov["dates"])
    }
    print("[DEBUG] Benchmarks adicionados.")

    # REPORT
    print("[DEBUG] Gerando relatório de performance...")
    for name, data in results.items():
        returns = data["returns"]
        print(name, type(returns))
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

    # Gráfico de retornos acumulados dos melhores modelos
    plot_top_cumulative(results, report)

    # Relação risco vs retorno
    plot_risk_return(results)

    # Drawdowns (quedas máximas)
    plot_drawdowns(results, report)

    # Boxplot dos retornos
    plot_return_boxplot(results, report)

    print("[DEBUG] Execução finalizada com sucesso!")


# Executa a função main apenas se o script for rodado diretamente
if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count

from config import *
from src.data_loader import load_data
from src.ml_models import predict_returns
from src.covariance import estimate_covariance
from src.optimizer import optimize_portfolio
from src.backtest import run_backtest
from src.metrics import mse, mae, directional_accuracy
from src.benchmarks import equal_weight_portfolio, ibov_returns
from src.report import generate_report
from src.stat_tests import diebold_mariano
from src.analysis import generate_performance_table, plot_cumulative_returns, plot_drawdowns
import warnings
warnings.filterwarnings("ignore")

# =========================================
# 🚀 FUNÇÃO PARALELA (1 experimento)
# =========================================
def run_single_experiment(args):
    m, gamma, lambda_reg, returns = args

    name = f"{m}_g{gamma}_l{lambda_reg}"
    print(f"\n######## Modelo: {name} ########")

    def model_wrapper(data):
        return predict_returns(data, model_type=m)

    portfolio_returns, preds, reals = run_backtest(
        returns,
        model_wrapper,
        estimate_covariance,
        lambda mu, cov: optimize_portfolio(mu, cov, lambda_reg, gamma),
        config=__import__("config")
    )

    # =========================
    # DEBUG (opcional)
    # =========================
    print(f"\n--- DEBUG {name} RETURNS ---")
    print("Max:", np.max(portfolio_returns))
    print("Min:", np.min(portfolio_returns))
    print("Mean:", np.mean(portfolio_returns))

    # =========================
    # CUMULATIVO
    # =========================
    cum = np.cumprod(1 + portfolio_returns)

    print(f"\n--- DEBUG {name} CUMULATIVO ---")
    print("Final value:", cum[-1])

    # =========================
    # SALVAR
    # =========================
    pd.DataFrame(portfolio_returns).to_csv(f"output/{name}_returns.csv")
    pd.DataFrame(cum).to_csv(f"output/{name}_cum.csv")

    return name, {
        "returns": portfolio_returns,
        "mse": mse(reals, preds),
        "mae": mae(reals, preds),
        "direction": directional_accuracy(reals, preds),
        "errors": (reals - preds).flatten()
    }


# =========================================
# 🧠 MAIN
# =========================================
def main():

    # =========================
    # 📁 Criar pasta
    # =========================
    os.makedirs("output", exist_ok=True)

    # =========================
    # 📊 LOAD DATA
    # =========================
    returns = load_data(TICKERS, START_DATE, END_DATE)

    print("\n=== DEBUG: RETURNS ===")
    print(returns.describe())
    print("Max retorno:", returns.max().max())
    print("Min retorno:", returns.min().min())
    print("Tem NaN?", returns.isna().any().any())

    # =========================
    # GRID SEARCH
    # =========================
    models = ["lasso", "ridge", "elastic"]
    gammas = [5, 10, 20]
    lambdas = [0.01, 0.1, 1]

    tasks = [
        (m, gamma, lambda_reg, returns)
        for m in models
        for gamma in gammas
        for lambda_reg in lambdas
    ]

    print(f"\n🚀 Rodando em paralelo com {cpu_count()} CPUs...\n")

    with Pool(2) as pool:
        outputs = pool.map(run_single_experiment, tasks)

    results = dict(outputs)

    # =========================
    # BENCHMARKS
    # =========================
    print("\n######## BENCHMARKS ########")

    eq = equal_weight_portfolio(returns)
    ibov = ibov_returns(START_DATE, END_DATE)

    results["equal_weight"] = {"returns": eq.values}
    results["ibov"] = {"returns": ibov.values}

    # =========================
    # REPORT
    # =========================
    report = generate_report(results)

    report = report.sort_values(by="Sharpe", ascending=False)

    print("\n=== RESULTADOS ===")
    print(report)

    report.to_csv("output/results_summary.csv", index=False)

    # =========================
    # 📈 GRÁFICO
    # =========================
    plt.figure(figsize=(12, 7))

    for name, data in results.items():
        cum = np.cumprod(1 + data["returns"])
        plt.plot(cum, label=name)

    plt.title("Comparação de Estratégias")
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.savefig("output/comparison_plot.png")
    plt.show()

    # =========================
    # 📊 DIEBOLD-MARIANO
    # =========================
    print("\n=== TESTE DIEBOLD-MARIANO ===")

    top_models = report["Model"].head(3).values

    for i in range(len(top_models)):
        for j in range(i + 1, len(top_models)):
            m1 = top_models[i]
            m2 = top_models[j]

            if "errors" in results[m1] and "errors" in results[m2]:
                dm, p = diebold_mariano(
                    results[m1]["errors"],
                    results[m2]["errors"]
                )
                print(f"{m1} vs {m2}: DM={dm:.4f}, p-value={p:.4f}")

    import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count

from config import *
from src.data_loader import load_data
from src.ml_models import predict_returns
from src.covariance import estimate_covariance
from src.optimizer import optimize_portfolio
from src.backtest import run_backtest
from src.metrics import mse, mae, directional_accuracy
from src.benchmarks import equal_weight_portfolio, ibov_returns
from src.report import generate_report
from src.stat_tests import diebold_mariano
from src.analysis import generate_performance_table, plot_cumulative_returns, plot_drawdowns
import warnings
warnings.filterwarnings("ignore")

# =========================================
# 🚀 FUNÇÃO PARALELA (1 experimento)
# =========================================
def run_single_experiment(args):
    m, gamma, lambda_reg, returns = args

    name = f"{m}_g{gamma}_l{lambda_reg}"
    print(f"\n######## Modelo: {name} ########")

    def model_wrapper(data):
        return predict_returns(data, model_type=m)

    portfolio_returns, preds, reals = run_backtest(
        returns,
        model_wrapper,
        estimate_covariance,
        lambda mu, cov: optimize_portfolio(mu, cov, lambda_reg, gamma),
        config=__import__("config")
    )

    # =========================
    # DEBUG (opcional)
    # =========================
    print(f"\n--- DEBUG {name} RETURNS ---")
    print("Max:", np.max(portfolio_returns))
    print("Min:", np.min(portfolio_returns))
    print("Mean:", np.mean(portfolio_returns))

    # =========================
    # CUMULATIVO
    # =========================
    cum = np.cumprod(1 + portfolio_returns)

    print(f"\n--- DEBUG {name} CUMULATIVO ---")
    print("Final value:", cum[-1])

    # =========================
    # SALVAR
    # =========================
    pd.DataFrame(portfolio_returns).to_csv(f"output/{name}_returns.csv")
    pd.DataFrame(cum).to_csv(f"output/{name}_cum.csv")

    return name, {
        "returns": portfolio_returns,
        "mse": mse(reals, preds),
        "mae": mae(reals, preds),
        "direction": directional_accuracy(reals, preds),
        "errors": (reals - preds).flatten()
    }


# =========================================
# 🧠 MAIN
# =========================================
def main():

    # =========================
    # 📁 Criar pasta
    # =========================
    os.makedirs("output", exist_ok=True)

    # =========================
    # 📊 LOAD DATA
    # =========================
    returns = load_data(TICKERS, START_DATE, END_DATE)

    print("\n=== DEBUG: RETURNS ===")
    print(returns.describe())
    print("Max retorno:", returns.max().max())
    print("Min retorno:", returns.min().min())
    print("Tem NaN?", returns.isna().any().any())

    # =========================
    # GRID SEARCH
    # =========================
    models = ["lasso", "ridge", "elastic"]
    gammas = [1, 5, 10, 20]
    lambdas = [0.01, 0.1, 1]

    tasks = [
        (m, gamma, lambda_reg, returns)
        for m in models
        for gamma in gammas
        for lambda_reg in lambdas
    ]

    print(f"\n🚀 Rodando em paralelo com {cpu_count()} CPUs...\n")

    with Pool(2) as pool:
        outputs = pool.map(run_single_experiment, tasks)

    results = dict(outputs)

    # =========================
    # BENCHMARKS
    # =========================
    print("\n######## BENCHMARKS ########")

    eq = equal_weight_portfolio(returns)
    ibov = ibov_returns(START_DATE, END_DATE)

    results["equal_weight"] = {"returns": eq.values}
    results["ibov"] = {"returns": ibov.values}

    # =========================
    # REPORT
    # =========================
    report = generate_report(results)

    report = report.sort_values(by="Sharpe", ascending=False)

    print("\n=== RESULTADOS ===")
    print(report)

    report.to_csv("output/results_summary.csv", index=False)

    # =========================
    # 📈 GRÁFICO
    # =========================
    plt.figure(figsize=(12, 7))

    for name, data in results.items():
        cum = np.cumprod(1 + data["returns"])
        plt.plot(cum, label=name)

    plt.title("Comparação de Estratégias")
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.savefig("output/comparison_plot.png")
    plt.show()

    # =========================
    # 📊 DIEBOLD-MARIANO
    # =========================
    print("\n=== TESTE DIEBOLD-MARIANO ===")

    top_models = report["Model"].head(3).values

    for i in range(len(top_models)):
        for j in range(i + 1, len(top_models)):
            m1 = top_models[i]
            m2 = top_models[j]

            if "errors" in results[m1] and "errors" in results[m2]:
                dm, p = diebold_mariano(
                    results[m1]["errors"],
                    results[m2]["errors"]
                )
                print(f"{m1} vs {m2}: DM={dm:.4f}, p-value={p:.4f}")

    # =========================
    # 📊 PERFORMANCE TABLE
    # =========================

    table = generate_performance_table(results)

    print("\n=== PERFORMANCE TABLE ===")
    print(table)

    table.to_csv("output/performance_table.csv", index=False)


    # =========================
    # 📈 GRAFICOS
    # =========================

    plot_cumulative_returns(results)
    plot_drawdowns(results)

print("\nGráficos salvos na pasta output")

# =========================================
# ⚠️ IMPORTANTE (Windows)
# =========================================
if __name__ == "__main__":
    main()
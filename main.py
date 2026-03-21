from multiprocessing import Pool

from config import *
from src.data_loader import load_data
from src.benchmarks import equal_weight_portfolio, ibov_returns
from src.report import generate_report
from src.utils import diebold_mariano
from src.analysis import plot_top_cumulative, plot_risk_return, plot_drawdowns, plot_return_boxplot
from src.single_experiment import run_single_experiment

import warnings
warnings.filterwarnings("ignore")


# MAIN
def main():
    returns = load_data(TICKERS, START_DATE, END_DATE)

    # GRID SEARCH (reduzido para performance)
    models = ["lasso", "ridge", "elastic"]
    gammas = [1, 3, 5, 10, 20]
    lambdas = [0.1, 1, 10, 15, 20]

    tasks = [
        (m, gamma, lambda_reg, returns)
        for m in models
        for gamma in gammas
        for lambda_reg in lambdas
    ]

    with Pool(4) as pool:
        outputs = pool.map(run_single_experiment, tasks)

    results = dict(outputs)

    # BENCHMARKS
    eq = equal_weight_portfolio(returns)
    ibov = ibov_returns(START_DATE, END_DATE)

    results["equal_weight"] = {"returns": eq.values}
    results["ibov"] = {"returns": ibov.values}

    # REPORT
    report = generate_report(results)
    report = report.sort_values(by="Sharpe", ascending=False)

    print("\n===== PERFORMANCE =====\n")
    print(report)

    # DIEBOLD-MARIANO
    top_models = report["Model"].head(3).values

    print("\n===== DIEBOLD-MARIANO =====\n")

    for i in range(len(top_models)):
        for j in range(i + 1, len(top_models)):

            m1 = top_models[i]
            m2 = top_models[j]

            if "errors" in results[m1] and "errors" in results[m2]:

                dm, p = diebold_mariano(
                    results[m1]["errors"],
                    results[m2]["errors"]
                )

                print(f"{m1} vs {m2}: DM={dm:.4f}, p={p:.4f}")

    # VISUALIZAÇÕES
    plot_top_cumulative(results, report)
    plot_risk_return(results)
    plot_drawdowns(results, report)
    plot_return_boxplot(results, report)


if __name__ == "__main__":
    main()
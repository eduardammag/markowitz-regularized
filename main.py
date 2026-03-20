import os
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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():

    # =========================
    # 📁 Criar pasta output
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
    # MODELOS
    # =========================
    results = {}
    models = ["lasso", "ridge", "elastic"]

    for m in models:
        print(f"\n######## Modelo: {m} ########")

        def model_wrapper(data):
            return predict_returns(data, model_type=m)

        portfolio_returns, preds, reals = run_backtest(
            returns,
            model_wrapper,
            estimate_covariance,
            lambda mu, cov: optimize_portfolio(mu, cov, LAMBDA_REG),
            config=__import__("config")
        )

        # =========================
        # DEBUG PORTFOLIO RETURNS
        # =========================
        print(f"\n--- DEBUG {m} RETURNS ---")
        print("Max:", np.max(portfolio_returns))
        print("Min:", np.min(portfolio_returns))
        print("Mean:", np.mean(portfolio_returns))

        if np.max(portfolio_returns) > 1:
            print("🚨 ALERTA: retorno > 100% em um período")

        if np.min(portfolio_returns) < -1:
            print("🚨 ALERTA: retorno < -100% (impossível!)")

        # =========================
        # DEBUG PREDIÇÕES
        # =========================
        print(f"\n--- DEBUG {m} PREDIÇÕES ---")
        print("Pred max:", np.max(preds))
        print("Pred min:", np.min(preds))
        print("Real max:", np.max(reals))
        print("Real min:", np.min(reals))

        # =========================
        # CUMULATIVO DEBUG
        # =========================
        cum = np.cumprod(1 + portfolio_returns)

        print(f"\n--- DEBUG {m} CUMULATIVO ---")
        print("Final value:", cum[-1])

        if cum[-1] > 100:
            print("🚨 ALERTA: crescimento explosivo (provável bug)")

        # =========================
        # SALVAR
        # =========================
        pd.DataFrame(portfolio_returns).to_csv(f"output/{m}_returns.csv")
        pd.DataFrame(cum).to_csv(f"output/{m}_cum.csv")

        results[m] = {
            "returns": portfolio_returns,
            "mse": mse(reals, preds),
            "mae": mae(reals, preds),
            "direction": directional_accuracy(reals, preds),
            "errors": (reals - preds).flatten()
        }

    # =========================
    # BENCHMARKS
    # =========================
    print("\n######## BENCHMARKS ########")

    eq = equal_weight_portfolio(returns)
    print("\n--- DEBUG equal_weight ---")
    print("Max:", eq.max())
    print("Min:", eq.min())

    ibov = ibov_returns(START_DATE, END_DATE)
    print("\n--- DEBUG IBOV ---")
    print("Max:", ibov.max())
    print("Min:", ibov.min())

    results["equal_weight"] = {"returns": eq.values}
    results["ibov"] = {"returns": ibov.values}

    # =========================
    # REPORT
    # =========================
    report = generate_report(results)

    print("\n=== RESULTADOS ===")
    print(report)

    report.to_csv("output/results_summary.csv", index=False)

    # =========================
    # GRÁFICO
    # =========================
    plt.figure(figsize=(10,6))

    for name, data in results.items():
        cum = np.cumprod(1 + data["returns"])
        plt.plot(cum, label=name)

    plt.title("Comparação de Estratégias")
    plt.legend()
    plt.grid(True)

    plt.savefig("output/comparison_plot.png")
    plt.show()

    # =========================
    # DIEBOLD-MARIANO
    # =========================
    print("\n=== TESTE DIEBOLD-MARIANO ===")
    base = results["elastic"]["errors"]

    for m in ["lasso", "ridge"]:
        dm, p = diebold_mariano(base, results[m]["errors"])
        print(f"Elastic vs {m}: DM={dm:.4f}, p-value={p:.4f}")


if __name__ == "__main__":
    main()
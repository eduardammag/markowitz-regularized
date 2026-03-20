
import numpy as np
import matplotlib.pyplot as plt

# FUNÇÕES DE ANÁLISE

def compute_drawdown(returns):
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    return (cum - peak) / peak


def plot_top_cumulative(results, report, top=5):

    top_models = report["Model"].head(top)

    plt.figure(figsize=(10,6))

    for name in top_models:
        cum = np.cumprod(1 + results[name]["returns"])
        plt.plot(cum, label=name)

    plt.title("Top Estratégias - Retorno Cumulativo")
    plt.legend()
    plt.grid()
    plt.show()


def plot_risk_return(results):

    vols = []
    rets = []
    names = []

    for name, data in results.items():

        r = np.array(data["returns"])

        vols.append(np.std(r))
        rets.append(np.mean(r))
        names.append(name)

    plt.figure(figsize=(8,6))
    plt.scatter(vols, rets)

    for i, name in enumerate(names):
        plt.annotate(name, (vols[i], rets[i]))

    plt.xlabel("Volatilidade")
    plt.ylabel("Retorno médio")
    plt.title("Risk vs Return")
    plt.grid()
    plt.show()


def plot_drawdowns(results, report, top=5):

    top_models = report["Model"].head(top)

    plt.figure(figsize=(10,6))

    for name in top_models:

        dd = compute_drawdown(results[name]["returns"])
        plt.plot(dd, label=name)

    plt.title("Drawdowns")
    plt.legend()
    plt.grid()
    plt.show()


def plot_return_boxplot(results, report, top=5):

    top_models = report["Model"].head(top)

    data = [results[m]["returns"] for m in top_models]

    plt.figure(figsize=(8,6))
    plt.boxplot(data, labels=top_models)
    plt.title("Distribuição de Retornos")
    plt.grid()
    plt.show()


    plt.figure(figsize=(12,7))

    for name, data in results.items():

        r = np.array(data["returns"])
        cum = np.cumprod(1 + r)

        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak

        plt.plot(drawdown, label=name)

    plt.title("Drawdown Comparison")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")

    plt.legend()
    plt.grid(True)

    plt.savefig("output/drawdowns.png")
    plt.close()
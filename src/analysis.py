import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_model_types, filter_models, save_plot, compute_drawdown

# =========================
# 1. CUMULATIVE RETURNS
# =========================
def plot_top_cumulative(results, report, top=5):

    model_types = get_model_types(results)

    for model_type in model_types:

        filtered = filter_models(results, model_type)
        top_models = [m for m in report["Model"] if m in filtered][:top]

        if not top_models:
            continue

        fig = plt.figure(figsize=(10, 6))

        for name in top_models:
            r = np.array(results[name]["returns"])
            cum = np.cumprod(1 + r)
            plt.plot(cum, label=name)

        plt.title(f"{model_type.upper()} - Retorno Cumulativo")
        plt.legend()
        plt.grid()

        save_plot(fig, f"{model_type}.png", "cumulative")

    # 🔥 comparação final (melhores modelos + benchmarks)
    fig = plt.figure(figsize=(12, 7))

    best_models = []
    for model_type in model_types:
        best = next((m for m in report["Model"] if m.startswith(model_type)), None)
        if best:
            best_models.append(best)

    selected = best_models + ["equal_weight", "ibov"]

    for name in selected:
        if name in results:
            r = np.array(results[name]["returns"])
            cum = np.cumprod(1 + r)
            plt.plot(cum, label=name)

    plt.title("Melhores Modelos vs Benchmarks")
    plt.legend()
    plt.grid()

    save_plot(fig, "best_comparison.png", "comparison")


# =========================
# 2. RISK vs RETURN
# =========================
def plot_risk_return(results):

    model_types = get_model_types(results)

    for model_type in model_types:

        filtered = filter_models(results, model_type)

        vols, rets, names = [], [], []

        for name, data in filtered.items():
            r = np.array(data["returns"])
            vols.append(np.std(r))
            rets.append(np.mean(r))
            names.append(name)

        if not names:
            continue

        fig = plt.figure(figsize=(8, 6))
        plt.scatter(vols, rets)

        for i, name in enumerate(names):
            plt.annotate(name, (vols[i], rets[i]))

        plt.xlabel("Volatilidade")
        plt.ylabel("Retorno médio")
        plt.title(f"{model_type.upper()} - Risk vs Return")
        plt.grid()

        save_plot(fig, f"{model_type}.png", "risk_return")


# =========================
# 3. DRAWDOWN
# =========================
def plot_drawdowns(results, report, top=5):

    model_types = get_model_types(results)

    for model_type in model_types:

        filtered = filter_models(results, model_type)
        top_models = [m for m in report["Model"] if m in filtered][:top]

        if not top_models:
            continue

        fig = plt.figure(figsize=(10, 6))

        for name in top_models:
            dd = compute_drawdown(results[name]["returns"])
            plt.plot(dd, label=name)

        plt.title(f"{model_type.upper()} - Drawdown")
        plt.legend()
        plt.grid()

        save_plot(fig, f"{model_type}.png", "drawdown")


# =========================
# 4. BOXPLOT
# =========================
def plot_return_boxplot(results, report, top=5):

    model_types = get_model_types(results)

    for model_type in model_types:

        filtered = filter_models(results, model_type)
        top_models = [m for m in report["Model"] if m in filtered][:top]

        if not top_models:
            continue

        data = [results[m]["returns"] for m in top_models]

        fig = plt.figure(figsize=(8, 6))
        plt.boxplot(data, labels=top_models)

        plt.title(f"{model_type.upper()} - Distribuição de Retornos")
        plt.grid()

        save_plot(fig, f"{model_type}.png", "boxplot")
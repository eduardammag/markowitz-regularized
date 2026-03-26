import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_model_types, filter_models, save_plot, compute_drawdown

sns.set_style("whitegrid")
palette = sns.color_palette("tab10")


# Função auxiliar para alinhar datas e retornos
def _align_data(data):
    r = np.array(data["returns"])
    dates = np.array(data["dates"])

    try:
        dates = pd.to_datetime(dates)
    except:
        pass

    min_len = min(len(r), len(dates))
    return r[-min_len:], dates[-min_len:]


# 1. CUMULATIVE RETURNS
def plot_top_cumulative(results, report, top=5):
    print("[DEBUG] Gerando gráficos de retorno cumulativo...")

    model_types = get_model_types(results)

    for model_type in model_types:
        print(f"[DEBUG] Processando tipo: {model_type}")

        filtered = filter_models(results, model_type)
        top_models = [name for name in filtered.keys() if name in results][:top]

        if not top_models:
            print(f"[DEBUG] Nenhum modelo encontrado para {model_type}")
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, name in enumerate(top_models):
            r, dates = _align_data(results[name])
            cum = np.cumprod(1 + r)
            ax.plot(dates, cum, label=name, color=palette[idx % len(palette)], linewidth=2)

        ax.set_title(f"{model_type.upper()} - Retorno Cumulativo", fontsize=16)
        ax.set_xlabel("Data", fontsize=12)
        ax.set_ylabel("Retorno Acumulado", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_plot(fig, f"{model_type}.png", "cumulative")

    # Comparação final: melhores modelos vs benchmarks
    print("[DEBUG] Gerando comparação final entre melhores modelos...")
    fig, ax = plt.subplots(figsize=(14, 7))

    best_models = []
    for model_type in model_types:
        filtered = filter_models(results, model_type)
        if filtered:
            # Melhor modelo pelo retorno acumulado
            best = max(filtered.items(), key=lambda x: np.prod(1 + np.array(x[1]["returns"])))[0]
            best_models.append(best)

    benchmarks = ["equal_weight", "ibov"]
    selected = best_models + [b for b in benchmarks if b in results]

    for idx, name in enumerate(selected):
        r, dates = _align_data(results[name])
        cum = np.cumprod(1 + r)
        ax.plot(dates, cum, label=name, color=palette[idx % len(palette)], linewidth=2)

    ax.set_title("Melhores Modelos vs Benchmarks", fontsize=16)
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Retorno Acumulado", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_plot(fig, "best_comparison.png", "comparison")


# 2. RISK vs RETURN
def plot_risk_return(results):
    print("[DEBUG] Gerando gráficos Risk vs Return...")

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
            print(f"[DEBUG] Nenhum dado para {model_type}")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(vols, rets, s=100, c=range(len(names)), cmap='tab10')

        for i, name in enumerate(names):
            ax.annotate(name, (vols[i], rets[i]), textcoords="offset points", xytext=(5, 5))

        ax.set_title(f"{model_type.upper()} - Risco vs Retorno", fontsize=16)
        ax.set_xlabel("Volatilidade", fontsize=12)
        ax.set_ylabel("Retorno Médio", fontsize=12)
        ax.grid(True, alpha=0.5)
        plt.tight_layout()

        save_plot(fig, f"{model_type}.png", "risk_return")


# 3. DRAWDOWN
def plot_drawdowns(results, report, top=5):
    print("[DEBUG] Gerando gráficos de drawdown...")

    model_types = get_model_types(results)

    for model_type in model_types:
        filtered = filter_models(results, model_type)
        top_models = [name for name in filtered.keys() if name in results][:top]

        if not top_models:
            print(f"[DEBUG] Nenhum modelo para drawdown em {model_type}")
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, name in enumerate(top_models):
            r, dates = _align_data(results[name])
            dd = compute_drawdown(r)
            ax.plot(dates, dd, label=name, color=palette[idx % len(palette)], linewidth=2)

        ax.set_title(f"{model_type.upper()} - Drawdown", fontsize=16)
        ax.set_xlabel("Data", fontsize=12)
        ax.set_ylabel("Drawdown", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_plot(fig, f"{model_type}.png", "drawdown")


# 4. BOXPLOT
def plot_return_boxplot(results, report, top=5):
    print("[DEBUG] Gerando boxplots de retornos...")

    model_types = get_model_types(results)

    for model_type in model_types:
        filtered = filter_models(results, model_type)
        top_models = [name for name in filtered.keys() if name in results][:top]

        if not top_models:
            print(f"[DEBUG] Nenhum modelo para boxplot em {model_type}")
            continue

        data = [results[m]["returns"] for m in top_models]

        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(data, labels=top_models, patch_artist=True)

        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_title(f"{model_type.upper()} - Distribuição de Retornos", fontsize=16)
        ax.set_xlabel("Modelos", fontsize=12)
        ax.set_ylabel("Retornos", fontsize=12)
        ax.grid(True, alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_plot(fig, f"{model_type}.png", "boxplot")
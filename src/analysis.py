import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import get_model_types, filter_models, save_plot, compute_drawdown


# Função auxiliar (padroniza alinhamento)
def _align_data(data):
    r = np.array(data["returns"])
    dates = np.array(data["dates"])

    # Converte datas se necessário
    try:
        dates = pd.to_datetime(dates)
    except:
        pass

    # Garante mesmo tamanho
    min_len = min(len(r), len(dates))
    return r[-min_len:], dates[-min_len:]


# 1. CUMULATIVE RETURNS
def plot_top_cumulative(results, report, top=5):

    print("[DEBUG] Gerando gráficos de retorno cumulativo...")

    model_types = get_model_types(results)

    for model_type in model_types:

        print(f"[DEBUG] Processando tipo: {model_type}")

        filtered = filter_models(results, model_type)
        top_models = [m for m in report["Model"] if m in filtered][:top]

        if not top_models:
            print(f"[DEBUG] Nenhum modelo encontrado para {model_type}")
            continue

        fig = plt.figure(figsize=(10, 6))

        for name in top_models:
            r, dates = _align_data(results[name])
            cum = np.cumprod(1 + r)

            plt.plot(dates, cum, label=name)

        plt.title(f"{model_type.upper()} - Retorno Cumulativo")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_plot(fig, f"{model_type}.png", "cumulative")

    # comparação final
    print("[DEBUG] Gerando comparação final entre melhores modelos...")

    fig = plt.figure(figsize=(12, 7))

    best_models = []

    for model_type in model_types:
        filtered = filter_models(results, model_type)
        best = next((m for m in report["Model"] if m in filtered), None)

        if best:
            best_models.append(best)

    benchmarks = ["equal_weight", "ibov"]
    selected = best_models + benchmarks

    for name in selected:
        if name in results:
            r, dates = _align_data(results[name])
            cum = np.cumprod(1 + r)

            plt.plot(dates, cum, label=name)
        else:
            print(f"{name} não encontrado em results")

    plt.title("Melhores Modelos vs Benchmarks")
    plt.legend()
    plt.grid()
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

        fig = plt.figure(figsize=(8, 6))

        plt.scatter(vols, rets)

        for i, name in enumerate(names):
            plt.annotate(name, (vols[i], rets[i]))

        plt.xlabel("Volatilidade")
        plt.ylabel("Retorno médio")
        plt.title(f"{model_type.upper()} - Risk vs Return")
        plt.grid()
        plt.tight_layout()

        save_plot(fig, f"{model_type}.png", "risk_return")


# 3. DRAWDOWN
def plot_drawdowns(results, report, top=5):

    print("[DEBUG] Gerando gráficos de drawdown...")

    model_types = get_model_types(results)

    for model_type in model_types:

        filtered = filter_models(results, model_type)
        top_models = [m for m in report["Model"] if m in filtered][:top]

        if not top_models:
            print(f"[DEBUG] Nenhum modelo para drawdown em {model_type}")
            continue

        fig = plt.figure(figsize=(10, 6))

        for name in top_models:
            r, dates = _align_data(results[name])
            dd = compute_drawdown(r)

            plt.plot(dates, dd, label=name)

        plt.title(f"{model_type.upper()} - Drawdown")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_plot(fig, f"{model_type}.png", "drawdown")


# 4. BOXPLOT
def plot_return_boxplot(results, report, top=5):

    print("[DEBUG] Gerando boxplots de retornos...")

    model_types = get_model_types(results)

    for model_type in model_types:

        filtered = filter_models(results, model_type)
        top_models = [m for m in report["Model"] if m in filtered][:top]

        if not top_models:
            print(f"[DEBUG] Nenhum modelo para boxplot em {model_type}")
            continue

        data = [results[m]["returns"] for m in top_models]

        fig = plt.figure(figsize=(8, 6))

        plt.boxplot(data, labels=top_models)

        plt.title(f"{model_type.upper()} - Distribuição de Retornos")
        plt.grid()
        plt.tight_layout()

        save_plot(fig, f"{model_type}.png", "boxplot")
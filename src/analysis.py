import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_model_types, filter_models, save_plot, compute_drawdown


# 1. CUMULATIVE RETURNS
def plot_top_cumulative(results, report, top=5):

    print("[DEBUG] Gerando gráficos de retorno cumulativo...")

    # Identifica os tipos de modelos (ex: ridge, lasso, etc.)
    model_types = get_model_types(results)

    for model_type in model_types:

        print(f"[DEBUG] Processando tipo: {model_type}")

        # Filtra apenas modelos desse tipo
        filtered = filter_models(results, model_type)

        # Seleciona os top N modelos com base no ranking do report
        top_models = [m for m in report["Model"] if m in filtered][:top]

        if not top_models:
            print(f"[DEBUG] Nenhum modelo encontrado para {model_type}")
            continue

        # Cria figura
        fig = plt.figure(figsize=(10, 6))

        # Plota retorno acumulado de cada modelo
        for name in top_models:
            r = np.array(results[name]["returns"])
            cum = np.cumprod(1 + r)
            plt.plot(cum, label=name)

        plt.title(f"{model_type.upper()} - Retorno Cumulativo")
        plt.legend()
        plt.grid()

        # Salva o gráfico
        save_plot(fig, f"{model_type}.png", "cumulative")

    #  comparação final (melhor de cada família + benchmarks)
    print("[DEBUG] Gerando comparação final entre melhores modelos...")

    fig = plt.figure(figsize=(12, 7))

    best_models = []

    for model_type in model_types:
        filtered = filter_models(results, model_type)

        # Pega o melhor modelo dessa família baseado no ranking
        best = next((m for m in report["Model"] if m in filtered), None)

        if best:
            best_models.append(best)
        else:
            print(f"Nenhum modelo encontrado para {model_type}")

    # Define benchmarks
    benchmarks = ["equal_weight", "ibov"]
    selected = best_models + benchmarks

    # Plota todos os selecionados
    for name in selected:
        if name in results:
            r = np.array(results[name]["returns"])
            cum = np.cumprod(1 + r)
            plt.plot(cum, label=name)
        else:
            print(f" {name} não encontrado em results")

    plt.title("Melhor Ridge vs Lasso vs ElasticNet vs Benchmarks")
    plt.legend()
    plt.grid()

    save_plot(fig, "best_comparison.png", "comparison")


# 2. RISK vs RETURN
def plot_risk_return(results):

    print("[DEBUG] Gerando gráficos Risk vs Return...")

    model_types = get_model_types(results)

    for model_type in model_types:

        filtered = filter_models(results, model_type)

        vols, rets, names = [], [], []

        # Calcula volatilidade e retorno médio
        for name, data in filtered.items():
            r = np.array(data["returns"])
            vols.append(np.std(r))
            rets.append(np.mean(r))
            names.append(name)

        if not names:
            print(f"[DEBUG] Nenhum dado para {model_type}")
            continue

        fig = plt.figure(figsize=(8, 6))

        # Scatter plot risco vs retorno
        plt.scatter(vols, rets)

        # Adiciona labels nos pontos
        for i, name in enumerate(names):
            plt.annotate(name, (vols[i], rets[i]))

        plt.xlabel("Volatilidade")
        plt.ylabel("Retorno médio")
        plt.title(f"{model_type.upper()} - Risk vs Return")
        plt.grid()

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

        # Calcula e plota drawdown
        for name in top_models:
            dd = compute_drawdown(results[name]["returns"])
            plt.plot(dd, label=name)

        plt.title(f"{model_type.upper()} - Drawdown")
        plt.legend()
        plt.grid()

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

        # Coleta os retornos dos modelos
        data = [results[m]["returns"] for m in top_models]

        fig = plt.figure(figsize=(8, 6))

        # Cria boxplot
        plt.boxplot(data, labels=top_models)

        plt.title(f"{model_type.upper()} - Distribuição de Retornos")
        plt.grid()

        save_plot(fig, f"{model_type}.png", "boxplot")
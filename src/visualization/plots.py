import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.performance import compute_drawdown
from src.visualization.helpers import save_plot

sns.set_theme(style="whitegrid", context="paper")

MODEL_ORDER = [
    "historical_mean",
    "lasso",
    "ridge",
    "elastic",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "equal_weight",
    "ibov",
]

MODEL_LABELS = {
    "historical_mean": "Media historica",
    "lasso": "Lasso",
    "ridge": "Ridge",
    "elastic": "Elastic Net",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "equal_weight": "Equal Weight",
    "ibov": "IBOV",
}


def _base_model_name(name):
    return name.split("_g")[0]


def _label(name):
    return MODEL_LABELS.get(_base_model_name(name), name)


def _ordered_result_names(results, report=None, include_benchmarks=True):
    available = list(results.keys())
    ordered = []

    for model in MODEL_ORDER:
        matches = [name for name in available if _base_model_name(name) == model]
        if matches:
            ordered.extend(matches)

    remaining = [name for name in available if name not in ordered]
    ordered.extend(remaining)

    if not include_benchmarks:
        ordered = [name for name in ordered if name not in ("equal_weight", "ibov")]

    if report is not None and "Model" in report:
        report_names = set(report["Model"])
        ordered = [name for name in ordered if name in report_names]

    return ordered


def _align_data(data):
    returns = np.asarray(data["returns"], dtype=float)
    dates = pd.to_datetime(np.asarray(data["dates"]))

    min_len = min(len(returns), len(dates))
    return returns[-min_len:], dates[-min_len:]


def _strategy_palette(names):
    colors = sns.color_palette("tab10", n_colors=max(len(names), 3))
    return {name: colors[idx % len(colors)] for idx, name in enumerate(names)}


def plot_cumulative_returns(results, report):
    names = _ordered_result_names(results, report)
    colors = _strategy_palette(names)

    fig, ax = plt.subplots(figsize=(12, 6))

    for name in names:
        returns, dates = _align_data(results[name])
        cumulative = np.cumprod(1 + returns) - 1
        linewidth = 2.4 if name not in ("equal_weight", "ibov") else 2.0
        linestyle = "-" if name not in ("equal_weight", "ibov") else "--"
        ax.plot(
            dates,
            cumulative,
            label=_label(name),
            color=colors[name],
            linewidth=linewidth,
            linestyle=linestyle,
        )

    ax.set_title("Retorno acumulado das estrategias")
    ax.set_xlabel("Data")
    ax.set_ylabel("Retorno acumulado")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.autofmt_xdate()
    fig.tight_layout()

    save_plot(fig, "cumulative_returns.png", "tcc")


def plot_drawdowns(results, report):
    names = _ordered_result_names(results, report)
    colors = _strategy_palette(names)

    fig, ax = plt.subplots(figsize=(12, 6))

    for name in names:
        returns, dates = _align_data(results[name])
        drawdown = compute_drawdown(returns)
        linewidth = 2.4 if name not in ("equal_weight", "ibov") else 2.0
        linestyle = "-" if name not in ("equal_weight", "ibov") else "--"
        ax.plot(
            dates,
            drawdown,
            label=_label(name),
            color=colors[name],
            linewidth=linewidth,
            linestyle=linestyle,
        )

    ax.set_title("Drawdown das estrategias")
    ax.set_xlabel("Data")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.autofmt_xdate()
    fig.tight_layout()

    save_plot(fig, "drawdowns.png", "tcc")


def plot_risk_return(results, report):
    names = _ordered_result_names(results, report)

    rows = []
    for name in names:
        returns = np.asarray(results[name]["returns"], dtype=float)
        rows.append(
            {
                "Model": name,
                "Label": _label(name),
                "Mean Return": np.mean(returns),
                "Volatility": np.std(returns),
                "Benchmark": name in ("equal_weight", "ibov"),
            }
        )

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Volatility",
        y="Mean Return",
        hue="Benchmark",
        style="Benchmark",
        s=120,
        ax=ax,
        legend=False,
    )

    for _, row in df.iterrows():
        ax.annotate(row["Label"], (row["Volatility"], row["Mean Return"]), xytext=(6, 4), textcoords="offset points")

    ax.set_title("Risco vs retorno medio por periodo")
    ax.set_xlabel("Volatilidade por periodo")
    ax.set_ylabel("Retorno medio por periodo")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    save_plot(fig, "risk_return.png", "tcc")


def plot_performance_bars(results, report):
    df = report.copy()
    df["Label"] = df["Model"].map(_label)
    df["ModelOrder"] = df["Model"].apply(lambda name: MODEL_ORDER.index(_base_model_name(name)) if _base_model_name(name) in MODEL_ORDER else 999)
    df = df.sort_values("ModelOrder")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.barplot(data=df, y="Label", x="Sharpe", ax=axes[0], color="#4c72b0")
    axes[0].set_title("Sharpe")
    axes[0].set_xlabel("Sharpe")
    axes[0].set_ylabel("")

    sns.barplot(data=df, y="Label", x="Return", ax=axes[1], color="#55a868")
    axes[1].set_title("Retorno acumulado")
    axes[1].set_xlabel("Retorno")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")

    sns.barplot(data=df, y="Label", x="Drawdown", ax=axes[2], color="#c44e52")
    axes[2].set_title("Max drawdown")
    axes[2].set_xlabel("Drawdown")
    axes[2].set_ylabel("")
    axes[2].xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")

    for ax in axes:
        ax.grid(True, axis="x", alpha=0.35)

    fig.suptitle("Resumo de performance das estrategias", y=1.03)
    fig.tight_layout()

    save_plot(fig, "performance_bars.png", "tcc")


def plot_average_weights(results, report):
    names = _ordered_result_names(results, report, include_benchmarks=False)
    rows = []

    for name in names:
        if "weights" not in results[name]:
            continue

        weights = np.asarray(results[name]["weights"], dtype=float)
        if weights.size == 0:
            continue

        asset_labels = [
            asset.replace(".SA", "")
            for asset in results[name].get("assets", [f"Ativo {idx + 1}" for idx in range(weights.shape[1])])
        ]
        avg_weights = weights.mean(axis=0)
        rows.append(pd.Series(avg_weights, index=asset_labels, name=_label(name)))

    if not rows:
        return

    weights_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        weights_df,
        annot=True,
        fmt=".1%",
        cmap="YlGnBu",
        vmin=0,
        vmax=0.30,
        linewidths=0.5,
        cbar_kws={"label": "Peso medio"},
        ax=ax,
    )
    ax.set_title("Pesos medios por ativo")
    ax.set_xlabel("Ativo")
    ax.set_ylabel("Estrategia")
    fig.tight_layout()

    save_plot(fig, "average_weights.png", "tcc")

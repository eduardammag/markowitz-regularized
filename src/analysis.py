import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 📊 METRICAS FINANCEIRAS
# =========================
def sharpe_ratio(returns):
    return np.mean(returns) / np.std(returns) * np.sqrt(252)


def annual_return(returns):
    return np.mean(returns) * 252


def annual_volatility(returns):
    return np.std(returns) * np.sqrt(252)


def max_drawdown(cumulative):
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


# =========================
# 📊 GERAR RELATÓRIO
# =========================
def generate_performance_table(results):

    rows = []

    for name, data in results.items():

        r = np.array(data["returns"])
        cum = np.cumprod(1 + r)

        rows.append({
            "Model": name,
            "Return": annual_return(r),
            "Volatility": annual_volatility(r),
            "Sharpe": sharpe_ratio(r),
            "MaxDrawdown": max_drawdown(cum)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Sharpe", ascending=False)

    return df


# =========================
# 📈 GRAFICO CAPITAL
# =========================
def plot_cumulative_returns(results):

    plt.figure(figsize=(12,7))

    for name, data in results.items():

        r = np.array(data["returns"])
        cum = np.cumprod(1 + r)

        plt.plot(cum, label=name)

    plt.title("Cumulative Portfolio Returns")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")

    plt.legend()
    plt.grid(True)

    plt.savefig("output/cumulative_returns.png")
    plt.close()


# =========================
# 📉 GRAFICO DRAWDOWN
# =========================
def plot_drawdowns(results):

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
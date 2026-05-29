import numpy as np
import yfinance as yf

from src.data.yahoo import configure_yfinance


def _compound_by_backtest_windows(returns, train_window, test_window):
    period_returns = []
    dates = []

    for i in range(train_window, len(returns) - test_window, test_window):
        test = returns.iloc[i:i + test_window]
        period_returns.append(np.prod(1 + test.values) - 1)
        dates.append(test.index[-1])

    return np.array(period_returns), np.array(dates)


# Portfólio com todos os pesos iguais (1/n)
def equal_weight_portfolio(returns, train_window, test_window):
    weights = np.ones(returns.shape[1]) / returns.shape[1]

    portfolio = returns @ weights  # vira Series
    period_returns, dates = _compound_by_backtest_windows(
        portfolio,
        train_window,
        test_window
    )

    return {
        "returns": period_returns,
        "dates": dates
    }

# Função para obter retornos do IBOVESPA
def ibov_returns(start, end, train_window, test_window):
    configure_yfinance()
    data = yf.download("^BVSP", start=start, end=end, auto_adjust=True, progress=False)

    if data is None or data.empty or "Close" not in data:
        print("[WARNING] Nao foi possivel baixar o IBOV. Benchmark sera ignorado.")
        return {
            "returns": np.array([]),
            "dates": np.array([])
        }

    ibov = data["Close"]
    if hasattr(ibov, "columns"):
        ibov = ibov.iloc[:, 0]

    returns = ibov.pct_change().dropna()

    if returns.empty:
        print("[WARNING] IBOV sem retornos validos. Benchmark sera ignorado.")
        return {
            "returns": np.array([]),
            "dates": np.array([])
        }

    period_returns, dates = _compound_by_backtest_windows(
        returns,
        train_window,
        test_window
    )

    return {
        "returns": period_returns,
        "dates": dates
    }


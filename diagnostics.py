"""
Diagnosticos do backtest.

Este script verifica pontos que podem inflar ou distorcer os resultados:
- Sharpe anualizado como diario vs frequencia real do backtest;
- alinhamento entre janela de treino e janela de teste;
- sensibilidade a custos de transacao;
- concentracao dos pesos em poucos ativos ou no ativo vencedor.

Uso:
    python diagnostics.py
"""

import os
import warnings

import numpy as np
import pandas as pd

import config
from src.data.loader import load_data
from src.ml_models import predict_returns
from src.portfolio.covariance import estimate_covariance
from src.portfolio.optimizer import optimize_portfolio

warnings.filterwarnings("ignore")


RISK_FREE_RATE = 0.02
TRANSACTION_COSTS = [0.0, 0.001, 0.005, 0.01]


def annualized_sharpe(period_returns, periods_per_year, rf_annual=RISK_FREE_RATE):
    period_returns = np.asarray(period_returns)
    if period_returns.size == 0 or np.std(period_returns) == 0:
        return np.nan

    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess = period_returns - rf_period
    return np.sqrt(periods_per_year) * np.mean(excess) / np.std(excess)


def current_daily_sharpe_formula(period_returns, rf_annual=RISK_FREE_RATE):
    period_returns = np.asarray(period_returns)
    if period_returns.size == 0 or np.std(period_returns) == 0:
        return np.nan

    excess = period_returns - rf_annual / 252
    return np.sqrt(252) * np.mean(excess) / np.std(excess)


def max_drawdown(period_returns):
    period_returns = np.asarray(period_returns)
    if period_returns.size == 0:
        return np.nan

    cumulative = np.cumprod(1 + period_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def run_diagnostic_backtest(returns, model_name, gamma, lambda_reg):
    gross_returns = []
    turnovers = []
    weights_history = []
    dates = []
    windows = []
    prev_weights = None

    for i in range(config.TRAIN_WINDOW, len(returns) - config.TEST_WINDOW, config.TEST_WINDOW):
        train = returns.iloc[i - config.TRAIN_WINDOW:i]
        test = returns.iloc[i:i + config.TEST_WINDOW]

        mu_pred = predict_returns(train, model_type=model_name)
        cov = estimate_covariance(train)
        weights = optimize_portfolio(mu_pred, cov, lambda_reg=lambda_reg, gamma=gamma)

        daily_portfolio_returns = test.values @ weights
        gross_return = np.prod(1 + daily_portfolio_returns) - 1

        turnover = 0.0
        if prev_weights is not None:
            turnover = np.sum(np.abs(weights - prev_weights))

        gross_returns.append(gross_return)
        turnovers.append(turnover)
        weights_history.append(weights)
        dates.append(test.index[-1])
        windows.append(
            {
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "train_size": len(train),
                "test_size": len(test),
            }
        )

        prev_weights = weights

    return {
        "gross_returns": np.array(gross_returns),
        "turnovers": np.array(turnovers),
        "weights": np.array(weights_history),
        "dates": np.array(dates),
        "windows": windows,
    }


def summarize_alignment(windows):
    if not windows:
        return {
            "n_periods": 0,
            "overlap_count": np.nan,
            "train_size_min": np.nan,
            "train_size_max": np.nan,
            "test_size_min": np.nan,
            "test_size_max": np.nan,
        }

    overlap_count = sum(w["train_end"] >= w["test_start"] for w in windows)
    return {
        "n_periods": len(windows),
        "overlap_count": overlap_count,
        "train_size_min": min(w["train_size"] for w in windows),
        "train_size_max": max(w["train_size"] for w in windows),
        "test_size_min": min(w["test_size"] for w in windows),
        "test_size_max": max(w["test_size"] for w in windows),
    }


def summarize_concentration(weights, asset_names, asset_total_returns):
    if weights.size == 0:
        return {
            "avg_max_weight": np.nan,
            "max_weight_seen": np.nan,
            "avg_hhi": np.nan,
            "top_asset": None,
            "winner_asset": None,
            "winner_avg_weight": np.nan,
        }

    max_weights = weights.max(axis=1)
    hhi = np.sum(weights ** 2, axis=1)

    avg_weights = weights.mean(axis=0)
    top_asset = asset_names[int(np.argmax(avg_weights))]
    winner_asset = asset_names[int(np.argmax(asset_total_returns))]
    winner_idx = asset_names.index(winner_asset)

    return {
        "avg_max_weight": float(np.mean(max_weights)),
        "max_weight_seen": float(np.max(max_weights)),
        "avg_hhi": float(np.mean(hhi)),
        "top_asset": top_asset,
        "winner_asset": winner_asset,
        "winner_avg_weight": float(np.mean(weights[:, winner_idx])),
    }


def main():
    returns = load_data(config.TICKERS, config.START_DATE, config.END_DATE)
    asset_names = list(returns.columns)
    asset_total_returns = (1 + returns).prod().values - 1

    periods_per_year = 252 / config.TEST_WINDOW
    gamma = config.gammas[0]
    lambda_reg = config.lambdas[0]

    rows = []
    print("\n===== DIAGNOSTICOS DO BACKTEST =====\n")
    print(f"Periodo: {config.START_DATE} a {config.END_DATE}")
    print(f"TRAIN_WINDOW: {config.TRAIN_WINDOW} dias")
    print(f"TEST_WINDOW: {config.TEST_WINDOW} dias")
    print(f"Frequencia inferida: {periods_per_year:.2f} periodos por ano")
    print(f"Gamma usado: {gamma}")
    print(f"Lambda usado: {lambda_reg}\n")

    for model_name in config.models:
        print(f"[INFO] Diagnosticando {model_name}...")
        result = run_diagnostic_backtest(returns, model_name, gamma, lambda_reg)

        gross = result["gross_returns"]
        turnovers = result["turnovers"]
        weights = result["weights"]

        row = {
            "Model": model_name,
            "Sharpe_sqrt252_atual": current_daily_sharpe_formula(gross),
            "Sharpe_freq_correta": annualized_sharpe(gross, periods_per_year),
            "Return_gross": np.prod(1 + gross) - 1 if gross.size else np.nan,
            "Drawdown_gross": max_drawdown(gross),
            "Avg_turnover": np.mean(turnovers[1:]) if turnovers.size > 1 else 0.0,
        }

        for cost in TRANSACTION_COSTS:
            net = gross - cost * turnovers
            row[f"Sharpe_cost_{cost:.3f}"] = annualized_sharpe(net, periods_per_year)
            row[f"Return_cost_{cost:.3f}"] = np.prod(1 + net) - 1 if net.size else np.nan

        row.update(summarize_alignment(result["windows"]))
        row.update(summarize_concentration(weights, asset_names, asset_total_returns))
        rows.append(row)

    summary = pd.DataFrame(rows)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, "diagnostics_summary.csv")
    summary.to_csv(output_path, index=False)

    columns_to_show = [
        "Model",
        "Sharpe_sqrt252_atual",
        "Sharpe_freq_correta",
        "Avg_turnover",
        "avg_max_weight",
        "max_weight_seen",
        "avg_hhi",
        "top_asset",
        "winner_asset",
        "winner_avg_weight",
        "overlap_count",
    ]

    print("\n===== RESUMO =====\n")
    print(summary[columns_to_show].sort_values("Sharpe_freq_correta", ascending=False))

    print("\n===== LEITURA RAPIDA =====\n")
    print("- Se Sharpe_sqrt252_atual for muito maior que Sharpe_freq_correta, o Sharpe estava anualizado como diario.")
    print("- overlap_count deve ser 0. Se for maior que 0, ha vazamento entre treino e teste.")
    print(f"- max_weight_seen perto de {config.MAX_WEIGHT:.0%} indica que a restricao maxima por ativo esta frequentemente ativa.")
    print("- winner_avg_weight alto indica concentracao no ativo vencedor do periodo completo.")
    print(f"\nArquivo salvo em: {output_path}")


if __name__ == "__main__":
    main()

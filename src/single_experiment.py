from config import *

from src.ml_models import predict_returns
from src.covariance import estimate_covariance
from src.optimizer import optimize_portfolio
from src.backtest import run_backtest
from src.metrics import mse, mae, directional_accuracy, sortino_ratio, calmar_ratio, turnover

import warnings
warnings.filterwarnings("ignore")


# EXPERIMENTO INDIVIDUAL
def run_single_experiment(args):

    print("[DEBUG] Iniciando experimento individual...")

    m, gamma, lambda_reg, returns = args

    name = f"{m}_g{gamma}_l{lambda_reg}"
    print(f"[DEBUG] Rodando modelo: {name}")

    def model_wrapper(data):
        return predict_returns(data, model_type=m)

    print("[DEBUG] Iniciando backtest...")
    portfolio_returns, preds, reals, weights_history, dates = run_backtest(
        returns,
        model_wrapper,
        estimate_covariance,
        lambda mu, cov: optimize_portfolio(mu, cov, lambda_reg, gamma),
        config=__import__("config")
    )
    print("[DEBUG] Backtest concluído.")

    # MÉTRICAS
    print("[DEBUG] Calculando métricas...")

    result = {
        "returns": portfolio_returns,

        # erro
        "mse": mse(reals, preds),
        "mae": mae(reals, preds),
        "direction": directional_accuracy(reals, preds),

        # novas métricas (AGORA CORRETAS)
        "sortino": sortino_ratio(portfolio_returns),
        "calmar": calmar_ratio(portfolio_returns),
        "turnover": turnover(weights_history),
        "dates": dates,
        "errors": (reals - preds).flatten()
    }

    print(f"[DEBUG] Experimento finalizado: {name}")

    return name, result
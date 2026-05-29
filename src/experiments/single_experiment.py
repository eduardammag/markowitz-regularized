import config as project_config
from src.ml_models import predict_returns
from src.backtesting.engine import run_backtest
from src.evaluation.prediction_metrics import (
    calmar_ratio,
    directional_accuracy,
    mae,
    mse,
    sortino_ratio,
    turnover,
)
from src.portfolio.covariance import estimate_covariance
from src.portfolio.optimizer import optimize_portfolio

import warnings
warnings.filterwarnings("ignore")


# EXPERIMENTO INDIVIDUAL
def run_single_experiment(args):

    m, gamma, lambda_reg, returns = args

    name = f"{m}_g{gamma}_l{lambda_reg}"
    print(f"[INFO] Rodando experimento: {name}")

    def model_wrapper(data):
        return predict_returns(data, model_type=m)

    portfolio_returns, preds, reals, weights_history, dates = run_backtest(
        returns,
        model_wrapper,
        estimate_covariance,
        lambda mu, cov: optimize_portfolio(mu, cov, lambda_reg, gamma),
        config=project_config
    )
    # MÉTRICAS

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
        "weights": weights_history,
        "assets": list(returns.columns),
        "errors": (reals - preds).flatten()
    }

    return name, result

from config import *
from src.ml_models import predict_returns
from src.covariance import estimate_covariance
from src.optimizer import optimize_portfolio
from src.backtest import run_backtest
from src.metrics import mse, mae, directional_accuracy

import warnings
warnings.filterwarnings("ignore")


# EXPERIMENTO INDIVIDUAL
def run_single_experiment(args):

    m, gamma, lambda_reg, returns = args
    name = f"{m}_g{gamma}_l{lambda_reg}"

    def model_wrapper(data):
        return predict_returns(data, model_type=m)

    portfolio_returns, preds, reals = run_backtest(
        returns,
        model_wrapper,
        estimate_covariance,
        lambda mu, cov: optimize_portfolio(mu, cov, lambda_reg, gamma),
        config=__import__("config")
    )

    return name, {
        "returns": portfolio_returns,
        "mse": mse(reals, preds),
        "mae": mae(reals, preds),
        "direction": directional_accuracy(reals, preds),
        "errors": (reals - preds).flatten()
    }
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

    if len(args) == 4:
        m, gamma, returns, prediction_cache = args
    else:
        m, gamma, returns = args
        prediction_cache = None

    name = f"{m}_g{gamma}"
    print(f"[INFO] Rodando experimento: {name}")

    def model_wrapper(data):
        return predict_returns(
            data,
            model_type=m,
            horizon=project_config.TEST_WINDOW,
        )

    portfolio_returns, preds, reals, weights_history, dates = run_backtest(
        returns,
        model_wrapper,
        estimate_covariance,
        lambda mu, cov: optimize_portfolio(mu, cov, gamma),
        config=project_config,
        prediction_cache=prediction_cache,
        prediction_cache_key=m,
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

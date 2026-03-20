import numpy as np
from tqdm import tqdm

def run_backtest(returns, model_fn, cov_fn, opt_fn, config):

    train_window = config.TRAIN_WINDOW
    test_window = config.TEST_WINDOW

    portfolio_returns = []

    preds_all = []
    reals_all = []

    for i in tqdm(range(train_window, len(returns) - test_window)):

        train = returns.iloc[i-train_window:i]
        test = returns.iloc[i:i+test_window]

        # previsão de retornos
        mu_pred = model_fn(train)

        # retorno real seguinte (média do período de teste)
        real_ret = test.mean().values

        preds_all.append(mu_pred)
        reals_all.append(real_ret)

        # covariância
        cov = cov_fn(train)

        # otimização
        weights = opt_fn(mu_pred, cov)

        # retorno do portfólio
        portfolio_ret = test @ weights
        portfolio_returns.extend(portfolio_ret)

    return (
        np.array(portfolio_returns),
        np.array(preds_all),
        np.array(reals_all)
    )
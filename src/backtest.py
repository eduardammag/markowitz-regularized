import numpy as np
from tqdm import tqdm

def run_backtest(returns, model_fn, cov_fn, opt_fn, config):

    train_window = config.TRAIN_WINDOW

    portfolio_returns = []

    preds_all = []
    reals_all = []

    capital = 1.0
    prev_weights = None

    cost = 0.001  # custo de transação (0.1%)

    for i in tqdm(range(train_window, len(returns) - 1)):

        train = returns.iloc[i-train_window:i]
        test = returns.iloc[i]  # 🔥 apenas 1 passo à frente

        # =========================
        # PREVISÃO
        # =========================
        mu_pred = model_fn(train)
        real_ret = test.values

        preds_all.append(mu_pred)
        reals_all.append(real_ret)

        # =========================
        # COVARIÂNCIA
        # =========================
        cov = cov_fn(train)

        # =========================
        # OTIMIZAÇÃO
        # =========================
        weights = opt_fn(mu_pred, cov)

        # =========================
        # RETORNO DO PORTFÓLIO
        # =========================
        port_ret = np.dot(weights, real_ret)

        # =========================
        # CUSTO DE TRANSAÇÃO
        # =========================
        if prev_weights is not None:
            turnover = np.sum(np.abs(weights - prev_weights))
            port_ret -= cost * turnover

        prev_weights = weights

        # =========================
        # ARMAZENAR
        # =========================
        portfolio_returns.append(port_ret)

        # =========================
        # DEBUG (opcional)
        # =========================
        if i % 200 == 0:
            print(f"Step {i} | Ret: {port_ret:.5f}")

    return (
        np.array(portfolio_returns),
        np.array(preds_all),
        np.array(reals_all)
    )
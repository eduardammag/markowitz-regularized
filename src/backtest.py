import numpy as np
from tqdm import tqdm


def run_backtest(returns, model_fn, cov_fn, opt_fn, config):

    print("[DEBUG] Iniciando backtest...")

    # Obtém parâmetros de janela a partir do config
    train_window = config.TRAIN_WINDOW
    test_window = config.TEST_WINDOW

    print(f"[DEBUG] TRAIN_WINDOW: {train_window}, TEST_WINDOW: {test_window}")

    # Listas para armazenar resultados
    portfolio_returns = []  # retornos do portfólio
    preds_all = []          # previsões do modelo
    reals_all = []          # valores reais
    weights_history = []

    # Variável para armazenar pesos do período anterior (para custo de transação)
    prev_weights = None

    # Custo de transação (0.1%)
    cost = 0.001  

    # Loop principal do backtest (rebalanceamento periódico)
    print("[DEBUG] Iniciando loop do backtest...")
    for i in tqdm(range(train_window, len(returns) - test_window, test_window)):

        # SPLIT TREINO / TESTE
        # Janela de treino (histórico passado)
        train = returns.iloc[i - train_window:i]

        # Janela de teste (período futuro)
        test = returns.iloc[i:i + test_window]

        # PREVISÃO
        # Gera previsão de retorno esperado
        mu_pred = model_fn(train)
        preds_all.append(mu_pred)

        # COVARIÂNCIA
        # Estima matriz de covariância dos ativos
        cov = cov_fn(train)

        # OTIMIZAÇÃO
        # Calcula os pesos ótimos do portfólio
        weights = opt_fn(mu_pred, cov)
        weights_history.append(weights)
        # RETORNO REAL (MULTI-PERÍODO)

        # Calcula retornos diários do portfólio na janela de teste
        port_rets = test.values @ weights

        # Calcula retorno acumulado composto no período
        cumulative_ret = np.prod(1 + port_rets) - 1

        # Armazena retornos reais (usando média como proxy)
        reals_all.append(test.mean().values)

        # CUSTO DE TRANSAÇÃO
        if prev_weights is not None:
            # Calcula turnover (mudança nos pesos)
            turnover = np.sum(np.abs(weights - prev_weights))

            # Aplica custo de transação
            cumulative_ret -= cost * turnover

        # Atualiza pesos anteriores
        prev_weights = weights

        # ARMAZENAMENTO
        portfolio_returns.append(cumulative_ret)

        # DEBUG (não muito verboso, só progresso essencial)
        print(f"[DEBUG] Step {i} | Ret (janela): {cumulative_ret:.4f}")

    print("[DEBUG] Backtest finalizado.")
    
    # Retorna resultados como arrays numpy
    return (
        np.array(portfolio_returns),
        np.array(preds_all),
        np.array(reals_all),
        np.array(weights_history)
    )
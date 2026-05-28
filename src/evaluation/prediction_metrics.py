import numpy as np

# MEAN SQUARED ERROR (MSE)
def mse(y_true, y_pred):
    print("[DEBUG] Calculando MSE...")

    # Calcula o erro quadrático médio
    value = np.mean((y_true - y_pred) ** 2)

    print(f"[DEBUG] MSE: {value:.6f}")
    return value


# MEAN ABSOLUTE ERROR (MAE)
def mae(y_true, y_pred):
    print("[DEBUG] Calculando MAE...")

    # Calcula o erro absoluto médio
    value = np.mean(np.abs(y_true - y_pred))

    print(f"[DEBUG] MAE: {value:.6f}")
    return value


# DIRECTIONAL ACCURACY
def directional_accuracy(y_true, y_pred):
    print("[DEBUG] Calculando Directional Accuracy...")

    # Compara o sinal (positivo/negativo) entre real e previsto
    value = np.mean(np.sign(y_true) == np.sign(y_pred))

    print(f"[DEBUG] Directional Accuracy: {value:.4f}")
    return value


# SORTINO RATIO
def sortino_ratio(returns, rf=0.02):
    print("[DEBUG] Calculando Sortino Ratio...")

    # Converte taxa livre de risco anual para diária
    rf_daily = rf / 252

    # Retornos em excesso
    excess = returns - rf_daily

    # Apenas retornos negativos (downside risk)
    downside = excess[excess < 0]

    # Evita divisão por zero
    if len(downside) == 0:
        print("[WARNING] Sem downside, retornando 0")
        return 0.0

    downside_std = np.std(downside)

    # Sortino anualizado
    value = np.sqrt(252) * np.mean(excess) / downside_std

    print(f"[DEBUG] Sortino: {value:.4f}")
    return value


# CALMAR RATIO
def calmar_ratio(returns):
    print("[DEBUG] Calculando Calmar Ratio...")

    # Retorno acumulado
    total_return = np.prod(1 + returns) - 1

    # Calcula drawdown
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak

    max_dd = np.min(drawdown)

    # Evita divisão por zero
    if max_dd == 0:
        print("[WARNING] Drawdown zero, retornando 0")
        return 0.0

    # Calmar
    value = total_return / abs(max_dd)

    print(f"[DEBUG] Calmar: {value:.4f}")
    return value


# TURNOVER
def turnover(weights_list):
    print("[DEBUG] Calculando Turnover...")

    if len(weights_list) < 2:
        print("[WARNING] Poucos pesos, turnover = 0")
        return 0.0

    total_turnover = 0.0

    for i in range(1, len(weights_list)):
        # GARANTIR FORMATO CORRETO AQUI
        w_t = np.array(weights_list[i]).flatten()
        w_prev = np.array(weights_list[i - 1]).flatten()

        if w_t.shape != w_prev.shape:
            raise ValueError(f"Shapes incompatíveis: {w_t.shape} vs {w_prev.shape}")

        step_turnover = np.sum(np.abs(w_t - w_prev))
        total_turnover += step_turnover

    avg_turnover = total_turnover / (len(weights_list) - 1)

    print(f"[DEBUG] Turnover médio: {avg_turnover:.4f}")
    return avg_turnover
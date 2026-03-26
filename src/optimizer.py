import cvxpy as cp 
import numpy as np


def optimize_portfolio(mu, cov, lambda_reg=0.1, gamma=10):

    print("[DEBUG] Iniciando otimização de portfólio...")

    # Número de ativos
    n = len(mu)
    print(f"[DEBUG] Número de ativos: {n}")

    # ESTABILIDADE NUMÉRICA
    # Adiciona pequeno valor na diagonal para evitar problemas de inversão/matriz singular
    cov = cov + 1e-6 * np.eye(n)

    # VARIÁVEL DE DECISÃO
    # Vetor de pesos do portfólio
    w = cp.Variable(n)

    # COMPONENTES DA FUNÇÃO OBJETIVO

    # Retorno esperado do portfólio
    portfolio_return = mu @ w

    # Risco (variância do portfólio)
    portfolio_risk = cp.quad_form(w, cov)

    # Regularização L2 (penaliza grandes pesos - suaviza solução)
    reg_l2 = cp.norm(w, 2)

    # Regularização L1 (induz sparsidade - menos ativos)
    reg_l1 = cp.norm(w, 1)

    # FUNÇÃO OBJETIVO
    # Maximiza retorno ajustado por risco e penalizações
    objective = cp.Maximize(
        portfolio_return 
        - gamma * portfolio_risk
        - lambda_reg * reg_l1
        - lambda_reg * reg_l2
    )

    # RESTRIÇÕES
    constraints = [
        # Soma dos pesos = 1 (portfólio totalmente investido)
        cp.sum(w) == 1,

        # Sem short (apenas posições compradas)
        w >= 0,

        # Limite máximo por ativo (30%)
        w <= 0.3
    ]

    # RESOLUÇÃO DO PROBLEMA
    prob = cp.Problem(objective, constraints)

    print("[DEBUG] Resolvendo problema de otimização...")
    prob.solve(solver=cp.ECOS)
    if w.value is None:
        print("[WARNING] Otimização falhou, usando equal weight")
        return np.ones(n) / n

    print(f"[DEBUG] Status da otimização: {prob.status}")

    # Retorna os pesos ótimos encontrados
    return w.value
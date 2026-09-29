import cvxpy as cp 
import numpy as np

from config import MAX_WEIGHT


def optimize_portfolio(mu, cov, gamma=10):

    # Número de ativos
    n = len(mu)
    
    # VARIÁVEL DE DECISÃO
    # Vetor de pesos do portfólio
    w = cp.Variable(n)

    # COMPONENTES DA FUNÇÃO OBJETIVO

    # Retorno esperado do portfólio
    portfolio_return = mu @ w

    # Risco (variância do portfólio)
    portfolio_risk = cp.quad_form(w, cov)

    # Markowitz clássico: retorno esperado menos aversão ao risco vezes variância.
    objective = cp.Maximize(portfolio_return - gamma * portfolio_risk)

    # RESTRIÇÕES
    constraints = [
        # Soma dos pesos = 1 (portfólio totalmente investido)
        cp.sum(w) == 1,

        # Sem short (apenas posições compradas)
        w >= 0,

        # Limite máximo por ativo definido em config.MAX_WEIGHT.
        w <= MAX_WEIGHT
    ]

    # RESOLUÇÃO DO PROBLEMA
    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        print("[WARNING] Otimização falhou, usando equal weight")
        return np.ones(n) / n

    # Retorna os pesos ótimos encontrados
    return w.value

import cvxpy as cp
import numpy as np

def optimize_portfolio(mu, cov, lambda_reg=0.1, gamma=10):

    n = len(mu)

    # estabilidade numérica
    cov = cov + 1e-6 * np.eye(n)

    w = cp.Variable(n)

    portfolio_return = mu @ w
    portfolio_risk = cp.quad_form(w, cov)

    reg_l2 = cp.norm(w, 2)
    reg_l1 = cp.norm(w, 1)

    objective = cp.Maximize(
        portfolio_return 
        - gamma * portfolio_risk
        - lambda_reg * reg_l1
        - lambda_reg * reg_l2
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 0.3
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value
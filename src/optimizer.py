import cvxpy as cp
import numpy as np

def optimize_portfolio(mu, cov, lambda_reg=0.1):
    
    n = len(mu)
    w = cp.Variable(n)

    portfolio_return = mu @ w
    portfolio_risk = cp.quad_form(w, cov)

    # L2 (Ridge) regularization
    reg_l2 = cp.norm(w, 2)

    # L1 (Lasso) regularization
    reg_l1 = cp.norm(w, 1)

    objective = cp.Maximize(
        portfolio_return - 0.5 * portfolio_risk
        - lambda_reg * reg_l1
        - lambda_reg * reg_l2
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value
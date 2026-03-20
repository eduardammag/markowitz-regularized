import numpy as np

def sharpe_ratio(returns, rf=0.02):
    excess = returns - rf/252
    return np.mean(excess) / np.std(excess)

def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return dd.min()
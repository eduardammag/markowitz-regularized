import numpy as np
from scipy import stats

def diebold_mariano(e1, e2, h=1):

    d = (e1 ** 2) - (e2 ** 2)
    mean_d = np.mean(d)

    gamma = [np.cov(d[:-lag], d[lag:])[0,1] for lag in range(1, h)]
    var_d = np.var(d) + 2 * np.sum(gamma)

    DM = mean_d / np.sqrt(var_d / len(d))

    p_value = 2 * (1 - stats.norm.cdf(abs(DM)))

    return DM, p_value
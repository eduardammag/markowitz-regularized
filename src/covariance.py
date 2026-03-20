import numpy as np
from sklearn.covariance import LedoitWolf

def estimate_covariance(returns):
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_
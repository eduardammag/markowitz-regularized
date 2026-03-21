from sklearn.covariance import LedoitWolf

def estimate_covariance(returns):
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_

    # usar isso EWMA (mais peso para dados recentes)??
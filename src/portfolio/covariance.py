from sklearn.covariance import LedoitWolf


def estimate_covariance(returns):
    # Inicializa o modelo Ledoit-Wolf
    # Esse método aplica shrinkage para melhorar a estabilidade da matriz de covariância
    lw = LedoitWolf()

    # Ajusta o modelo aos dados de retornos
    lw.fit(returns)

    # Retorna a matriz de covariância estimada
    cov = lw.covariance_

    return cov

    # usar isso EWMA (mais peso para dados recentes)??

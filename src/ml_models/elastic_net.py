"""
Regressao regularizada Elastic Net.

Combina penalizacao L1 e L2. Na pratica, fica entre Lasso e Ridge: pode reduzir
coeficientes e tambem lidar melhor com features correlacionadas.
"""

from sklearn.linear_model import ElasticNet

from src.ml_models.feature_engineering import make_scaled_supervised_dataset


def predict(returns):
    """
    Treina Elastic Net em dados historicos e retorna a previsao do proximo periodo.
    """

    print("[DEBUG] Modelo elastic: treinando ElasticNet")

    X_train, y_train, X_test = make_scaled_supervised_dataset(returns)

    model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred.flatten()

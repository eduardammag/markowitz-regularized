"""
Regressao regularizada Ridge.

O Ridge usa penalizacao L2, reduzindo coeficientes extremos sem necessariamente
zera-los. Costuma ser estavel quando as features sao correlacionadas.
"""

from sklearn.linear_model import Ridge

from src.ml_models.feature_engineering import make_scaled_supervised_dataset


def predict(returns):
    """
    Treina Ridge em dados historicos e retorna a previsao do proximo periodo.
    """

    print("[DEBUG] Modelo ridge: treinando Ridge")

    X_train, y_train, X_test = make_scaled_supervised_dataset(returns)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred.flatten()

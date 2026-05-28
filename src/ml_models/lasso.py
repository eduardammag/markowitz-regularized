"""
Regressao regularizada Lasso.

O Lasso usa penalizacao L1, que pode zerar coeficientes e selecionar features
implicitamente. Isso e util quando ha muitas variaveis defasadas e indicadores.
"""

from sklearn.linear_model import Lasso

from src.ml_models.feature_engineering import make_scaled_supervised_dataset


def predict(returns):
    """
    Treina Lasso em dados historicos e retorna a previsao do proximo periodo.
    """

    print("[DEBUG] Modelo lasso: treinando Lasso")

    X_train, y_train, X_test = make_scaled_supervised_dataset(returns)

    # Alpha pequeno mantem regularizacao suave, seguindo a configuracao original.
    model = Lasso(alpha=0.001, max_iter=10000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred.flatten()

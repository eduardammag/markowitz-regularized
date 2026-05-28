"""
Modelo nao linear: Random Forest Regressor.

Random Forest combina varias arvores de decisao e pode capturar relacoes nao
lineares entre lags, medias moveis, volatilidade e retornos futuros.
"""

from sklearn.ensemble import RandomForestRegressor

from src.ml_models.feature_engineering import make_supervised_dataset


def predict(returns):
    """
    Treina Random Forest multioutput e retorna a previsao do proximo periodo.
    """

    print("[DEBUG] Modelo random_forest: treinando RandomForestRegressor")

    # Arvores nao exigem padronizacao, entao usamos as features originais.
    X_train, y_train, X_test = make_supervised_dataset(returns)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        # O main.py ja roda experimentos em paralelo; manter n_jobs=1 evita
        # paralelismo aninhado e problemas de permissao no Windows.
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred.flatten()

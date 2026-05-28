"""
Modelo nao linear: Gradient Boosting.

O scikit-learn nao oferece GradientBoostingRegressor multioutput diretamente.
Por isso usamos MultiOutputRegressor, treinando um modelo separado para cada
ativo e mantendo a mesma interface de previsao vetorial.
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.ml_models.feature_engineering import make_supervised_dataset


def predict(returns):
    """
    Treina Gradient Boosting para cada ativo e preve o proximo periodo.
    """

    print("[DEBUG] Modelo gradient_boosting: treinando GradientBoostingRegressor")

    # Arvores de boosting tambem funcionam bem sem padronizacao das features.
    X_train, y_train, X_test = make_supervised_dataset(returns)

    base_model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred.flatten()

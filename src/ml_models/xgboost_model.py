"""
Modelo nao linear: XGBoost.

XGBoost e uma biblioteca externa especializada em gradient boosting. Como ele
nao faz parte do scikit-learn, o projeto precisa ter xgboost instalado.
"""

from sklearn.multioutput import MultiOutputRegressor

from src.ml_models.feature_engineering import make_supervised_dataset


def predict(returns):
    """
    Treina XGBoost para cada ativo e retorna a previsao do proximo periodo.
    """

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "O modelo xgboost foi selecionado, mas a biblioteca xgboost nao esta "
            "instalada. Rode: pip install -r requirements.txt"
        ) from exc

    X_train, y_train, X_test = make_supervised_dataset(returns)

    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred.flatten()

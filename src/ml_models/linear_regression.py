"""
Baseline supervisionado: regressao linear ordinaria.

Serve como comparacao simples para verificar se regularizacao e modelos nao
lineares realmente adicionam valor preditivo.
"""

from sklearn.linear_model import LinearRegression

from src.ml_models.feature_engineering import make_scaled_supervised_dataset


def predict(returns):
    """
    Treina uma regressao linear multioutput e preve retornos dos ativos.
    """

    print("[DEBUG] Modelo linear: treinando LinearRegression")

    X_train, y_train, X_test = make_scaled_supervised_dataset(returns)

    # LinearRegression aceita y com varias colunas, uma para cada ativo.
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred.flatten()

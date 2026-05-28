"""
Pacote de modelos de previsao de retorno.

Este arquivo funciona como a "porta de entrada" do pacote: o restante do
projeto continua chamando predict_returns(...), mas a implementacao real de
cada modelo fica separada em seu proprio arquivo.
"""

from src.ml_models.elastic_net import predict as predict_elastic_net
from src.ml_models.gradient_boosting import predict as predict_gradient_boosting
from src.ml_models.historical_mean import predict as predict_historical_mean
from src.ml_models.lasso import predict as predict_lasso
from src.ml_models.linear_regression import predict as predict_linear_regression
from src.ml_models.random_forest import predict as predict_random_forest
from src.ml_models.ridge import predict as predict_ridge
from src.ml_models.xgboost_model import predict as predict_xgboost


# Registro central dos modelos disponiveis.
# A chave e o nome usado em config.py; o valor e a funcao de previsao.
MODEL_REGISTRY = {
    "historical_mean": predict_historical_mean,
    "linear": predict_linear_regression,
    "lasso": predict_lasso,
    "ridge": predict_ridge,
    "elastic": predict_elastic_net,
    "random_forest": predict_random_forest,
    "gradient_boosting": predict_gradient_boosting,
    "xgboost": predict_xgboost,
}


def predict_returns(returns, model_type="lasso"):
    """
    Preve o vetor de retornos esperados para o proximo periodo.

    Parameters
    ----------
    returns : pandas.DataFrame
        Historico de retornos dos ativos dentro da janela de treino.
    model_type : str
        Nome do modelo escolhido em config.py.

    Returns
    -------
    numpy.ndarray
        Vetor 1D com uma previsao de retorno para cada ativo.
    """

    print(f"[DEBUG] Iniciando predicao com modelo: {model_type}")

    if model_type not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Modelo invalido: {model_type}. Disponiveis: {available}")

    return MODEL_REGISTRY[model_type](returns)

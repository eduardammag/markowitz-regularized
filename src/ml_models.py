from sklearn.linear_model import Lasso, Ridge, ElasticNet
import numpy as np

def train_model(X, y, model_type="lasso"):
    
    if model_type == "lasso":
        model = Lasso(alpha=0.001)
    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "elastic":
        model = ElasticNet(alpha=0.001, l1_ratio=0.5)
    else:
        raise ValueError("Modelo inválido")

    model.fit(X, y)
    return model


def predict_returns(returns, model_type="lasso"):
    
    X = returns.shift(1).dropna()
    y = returns.loc[X.index]

    model = train_model(X, y, model_type)

    preds = model.predict(X)

    return preds[-1]  # previsão mais recente
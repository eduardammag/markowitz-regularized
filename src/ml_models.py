from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler


def train_model(X, y, model_type="lasso"):
    
    if model_type == "lasso":
        model = Lasso(alpha=0.001, max_iter=10000)
    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "elastic":
        model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
    else:
        raise ValueError("Modelo inválido")

    model.fit(X, y)
    return model


def predict_returns(returns, model_type="lasso"):
    
    # =========================
    # FEATURES (lag)
    # =========================
    X = returns.shift(1).dropna()
    y = returns.loc[X.index]

    # =========================
    # SPLIT: usa só passado
    # =========================
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]

    X_test = X.iloc[-1:].values  # último ponto

    # =========================
    # NORMALIZAÇÃO (CRUCIAL)
    # =========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # TREINO
    # =========================
    model = train_model(X_train, y_train, model_type)

    # =========================
    # PREDIÇÃO OUT-OF-SAMPLE
    # =========================
    pred = model.predict(X_test)

    return pred.flatten()
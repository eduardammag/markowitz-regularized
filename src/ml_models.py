# from sklearn.linear_model import Lasso, Ridge, ElasticNet
# from sklearn.preprocessing import StandardScaler



# def predict_returns(returns, model_type="lasso"):
    
#     # FEATURES (lag)
#     X = returns.shift(1).dropna()
#     y = returns.loc[X.index]

#     # SPLIT: usa só passado
#     X_train = X.iloc[:-1]
#     y_train = y.iloc[:-1]

#     X_test = X.iloc[-1:].values  # último ponto

#     # NORMALIZAÇÃO (CRUCIAL)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # TREINO
#     model = train_model(X_train, y_train, model_type)

#     # PREDIÇÃO OUT-OF-SAMPLE
#     pred = model.predict(X_test)

#     return pred.flatten()


import pandas as pd
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


def build_features(returns):
    """
    Cria features mais ricas:
    - múltiplos lags
    - média móvel
    - volatilidade
    """

    lags = [1, 2, 3, 5]

    lagged = [returns.shift(lag) for lag in lags]
    lagged_df = pd.concat(lagged, axis=1)

    # Renomeia colunas (importante pra evitar duplicação de nomes)
    lagged_df.columns = [
        f"{col}_lag{lag}"
        for lag in lags
        for col in returns.columns
    ]

    # Estatísticas rolling
    rolling_mean = returns.rolling(20).mean()
    rolling_std = returns.rolling(20).std()

    rolling_mean.columns = [f"{col}_ma20" for col in returns.columns]
    rolling_std.columns = [f"{col}_vol20" for col in returns.columns]

    # Junta tudo
    X = pd.concat([lagged_df, rolling_mean, rolling_std], axis=1)

    return X


def predict_returns(returns, model_type="lasso"):
    
    # =========================
    # 1. FEATURES
    # =========================
    X = build_features(returns)

    # Target (retorno atual)
    y = returns.copy()

    # Remove NaNs (por causa dos lags e rolling)
    data = pd.concat([X, y], axis=1).dropna()

    X = data[X.columns]
    y = data[y.columns]

    # =========================
    # 2. SPLIT TEMPORAL
    # =========================
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]

    X_test = X.iloc[-1:]  # último ponto (sem .values ainda)

    # =========================
    # 3. NORMALIZAÇÃO
    # =========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # 4. TREINO
    # =========================
    model = train_model(X_train, y_train, model_type)

    # =========================
    # 5. PREDIÇÃO OUT-OF-SAMPLE
    # =========================
    pred = model.predict(X_test)

    return pred.flatten()
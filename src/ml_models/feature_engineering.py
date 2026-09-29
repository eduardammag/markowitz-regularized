"""
Funcoes comuns de engenharia de features para os modelos supervisionados.

Os modelos deste projeto recebem retornos historicos e tentam prever a media
diaria dos retornos no horizonte futuro de avaliacao. Para isso, transformamos
a serie temporal em uma tabela supervisionada com lags, media movel e volatilidade.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(returns):
    """
    Cria variaveis explicativas a partir dos retornos historicos.

    Features usadas:
    - retornos defasados em 1, 2, 3 e 5 dias;
    - media movel de 20 dias;
    - volatilidade movel de 20 dias.
    """

    # Lags capturam memoria curta dos retornos.
    lags = [1, 2, 3, 5]
    lagged = [returns.shift(lag) for lag in lags]
    lagged_df = pd.concat(lagged, axis=1)

    # Renomeia colunas para deixar claro qual ativo e qual defasagem gerou a feature.
    lagged_df.columns = [
        f"{col}_lag{lag}"
        for lag in lags
        for col in returns.columns
    ]

    # Media e volatilidade movel resumem comportamento recente do ativo.
    rolling_mean = returns.rolling(20).mean()
    rolling_std = returns.rolling(20).std()

    rolling_mean.columns = [f"{col}_ma20" for col in returns.columns]
    rolling_std.columns = [f"{col}_vol20" for col in returns.columns]

    X = pd.concat([lagged_df, rolling_mean, rolling_std], axis=1)

    return X


def make_supervised_dataset(returns, horizon=1):
    """
    Monta X_train, y_train e X_test respeitando ordem temporal.

    Cada linha de features no instante t prevê a média dos retornos entre
    t+1 e t+horizon. A última linha disponível vira X_test.
    """
    if horizon < 1:
        raise ValueError("horizon deve ser pelo menos 1")

    features = build_features(returns)
    target = returns.shift(-1).rolling(horizon).mean().shift(-(horizon - 1))
    target.columns = [f"target_{column}" for column in returns.columns]

    data = pd.concat([features, target], axis=1).dropna()
    target_columns = list(target.columns)

    X_train = data[features.columns]
    y_train = data[target_columns].copy()
    y_train.columns = returns.columns
    X_test = features.iloc[[-1]]

    return X_train, y_train, X_test


def make_scaled_supervised_dataset(returns, horizon=1):
    """
    Cria dataset supervisionado e aplica padronizacao nas features.

    A escala e ajustada apenas no treino para evitar vazamento de informacao.
    """

    X_train, y_train, X_test = make_supervised_dataset(returns, horizon)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled

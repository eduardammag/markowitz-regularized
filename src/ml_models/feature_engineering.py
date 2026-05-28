"""
Funcoes comuns de engenharia de features para os modelos supervisionados.

Os modelos deste projeto recebem retornos historicos e tentam prever o retorno
do proximo periodo. Para isso, transformamos a serie temporal em uma tabela
supervisionada com lags, media movel e volatilidade.
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

    print("[DEBUG] Construindo features...")

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

    print(f"[DEBUG] Features criadas: {X.shape}")
    return X


def make_supervised_dataset(returns):
    """
    Monta X_train, y_train e X_test respeitando ordem temporal.

    O ultimo ponto disponivel vira X_test, pois queremos prever o proximo vetor
    de retornos esperado a partir da janela de treino do backtest.
    """

    # Shift evita usar informacao do mesmo dia para prever o proprio dia.
    X = build_features(returns).shift(1)
    y = returns.copy()

    # Remove linhas incompletas criadas por lags e janelas moveis.
    data = pd.concat([X, y], axis=1).dropna()

    X = data[X.columns]
    y = data[y.columns]

    # Treina em todo o historico menos a ultima observacao.
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]

    # A ultima linha e usada como ponto fora da amostra.
    X_test = X.iloc[-1:]

    print(f"[DEBUG] Dataset supervisionado: X_train={X_train.shape}, X_test={X_test.shape}")

    return X_train, y_train, X_test


def make_scaled_supervised_dataset(returns):
    """
    Cria dataset supervisionado e aplica padronizacao nas features.

    A escala e ajustada apenas no treino para evitar vazamento de informacao.
    """

    X_train, y_train, X_test = make_supervised_dataset(returns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled

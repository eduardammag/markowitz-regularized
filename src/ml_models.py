import pandas as pd
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
# proximo passo é adicionar o modelo XGBoost

def train_model(X, y, model_type="lasso"):
    print(f"[DEBUG] Treinando modelo: {model_type}")

    # Seleciona o modelo com base no tipo
    if model_type == "lasso":
        model = Lasso(alpha=0.001, max_iter=10000)
    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "elastic":
        model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
    else:
        raise ValueError("Modelo inválido")

    # Treina o modelo
    model.fit(X, y)

    print("[DEBUG] Modelo treinado com sucesso.")
    return model


def build_features(returns):
    """
    Cria features mais ricas:
    - múltiplos lags
    - média móvel
    - volatilidade
    """

    print("[DEBUG] Construindo features...")

    # Define defasagens (lags)
    lags = [1, 2, 3, 5]

    # Cria versões defasadas dos retornos
    lagged = [returns.shift(lag) for lag in lags]

    # Concatena todas as features de lag
    lagged_df = pd.concat(lagged, axis=1)

    # Renomeia colunas para evitar duplicação
    lagged_df.columns = [
        f"{col}_lag{lag}"
        for lag in lags
        for col in returns.columns
    ]

    # FEATURES ROLLING

    # Média móvel de 20 períodos
    rolling_mean = returns.rolling(20).mean()

    # Volatilidade (desvio padrão) de 20 períodos
    rolling_std = returns.rolling(20).std()

    # Renomeia colunas
    rolling_mean.columns = [f"{col}_ma20" for col in returns.columns]
    rolling_std.columns = [f"{col}_vol20" for col in returns.columns]

    # CONCATENAÇÃO FINAL
    X = pd.concat([lagged_df, rolling_mean, rolling_std], axis=1)

    print(f"[DEBUG] Features criadas: {X.shape}")

    return X


def predict_returns(returns, model_type="lasso"):
    
    print("[DEBUG] Iniciando predição de retornos...")

    # 1. FEATURES
    X = build_features(returns).shift(1)
    # Target (retornos atuais)
    y = returns.copy()

    # Remove NaNs (causados por lag e rolling)
    data = pd.concat([X, y], axis=1).dropna()

    X = data[X.columns]
    y = data[y.columns]

    print(f"[DEBUG] Dados após limpeza: X={X.shape}, y={y.shape}")

    # 2. SPLIT TEMPORAL
    # Usa todos os dados menos o último para treino
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]

    # Último ponto para previsão (out-of-sample)
    X_test = X.iloc[-1:]

    print(f"[DEBUG] Split: treino={X_train.shape}, teste={X_test.shape}")

    # 3. NORMALIZAÇÃO
    scaler = StandardScaler()

    # Ajusta scaler no treino e transforma
    X_train = scaler.fit_transform(X_train)

    # Aplica transformação no teste
    X_test = scaler.transform(X_test)

    # 4. TREINO
    model = train_model(X_train, y_train, model_type)

    # 5. PREDIÇÃO OUT-OF-SAMPLE
    pred = model.predict(X_test)

    print("[DEBUG] Predição realizada.")

    # Retorna vetor achatado (1D)
    return pred.flatten()
import os

# Define o diretorio base do projeto (onde este arquivo esta localizado).
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Define o diretorio de saida dentro do diretorio base.
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Cria o diretorio de saida caso ele nao exista.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Universo de acoes brasileiras no Yahoo Finance.
# Criterio: empresas brasileiras grandes, consolidadas, liquidas e com
# historico completo no Yahoo Finance desde o inicio da amostra.
TICKERS = [
    "PETR4.SA",   # Petrobras
    "VALE3.SA",   # Vale
    "ITUB4.SA",   # Itau Unibanco
    "CMIG4.SA",   # Cemig
    "ABEV3.SA",   # Ambev
    "WEGE3.SA",   # WEG
    "BBAS3.SA",   # Banco do Brasil
    "BBDC4.SA",
    "SANB11.SA",  # Santander Brasil
    "B3SA3.SA",   # B3
    "SUZB3.SA",   # Suzano
    "VIVT3.SA",   # Telefonica Brasil
    "EQTL3.SA",   # Equatorial
    "RENT3.SA",   # Localiza
    "RADL3.SA",   # Raia Drogasil
    "TOTS3.SA",   # Totvs
    "GGBR4.SA",   # Gerdau
    "EGIE3.SA",   # Engie Brasil
    "LREN3.SA",   # Lojas Renner
    "CSNA3.SA",   # CSN
    ]

# Data inicial da analise.
START_DATE = "2010-01-31"

# Data final da analise.
END_DATE = "2026-01-31"

# Janela de treino (aproximadamente 1 ano de pregao).
TRAIN_WINDOW = 252

# Janela de teste (aproximadamente 1 mes).
TEST_WINDOW = 21

# Limite maximo de peso por ativo na carteira.
MAX_WEIGHT = 0.15

# Modelos principais do experimento.
#
# A versao principal do TCC fica propositalmente enxuta:
# - historical_mean: baseline tradicional de retorno esperado
# - lasso/ridge/elastic: regressoes regularizadas
# - random_forest/gradient_boosting/xgboost: modelos nao lineares
models = [
    "historical_mean",
    "lasso",
    "ridge",
    "elastic",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

# Gamma controla a aversao a risco no Markowitz.
# O experimento principal usa gamma=5 como configuracao central.
# Para analise de robustez, pode-se testar: [1, 5, 10].
gammas = [5]

# Lambda controla a regularizacao dos pesos da carteira.
# O experimento principal usa lambda=0.1 como configuracao central.
# Para analise de robustez, pode-se testar: [0.01, 0.1, 1].
lambdas = [0.1]

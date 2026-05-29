import os 

# Define o diretório base do projeto (onde este arquivo está localizado)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Define o diretório de saída (output) dentro do diretório base
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Cria o diretório de saída caso ele não exista (evita erro se já existir)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Universo de acoes brasileiras no Yahoo Finance.
# Criterio: empresas brasileiras grandes, consolidadas, liquidas e com
# historico completo no Yahoo Finance desde o inicio da amostra.
TICKERS = [
    "PETR4.SA",   # Petrobras
    "VALE3.SA",   # Vale
    "ITUB4.SA",   # Itau Unibanco
    "BPAC11.SA",  # BTG Pactual
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
    "PRIO3.SA",   # PRIO
    "GGBR4.SA",   # Gerdau
    "EGIE3.SA",   # Engie Brasil
    "KLBN11.SA",  # Klabin
    "BBSE3.SA",   # BB Seguridade
]

# Data inicial da análise
START_DATE = "2020-01-31"

# Data final da análise
END_DATE = "2026-01-31"

# Janela de treino (aproximadamente 1 anos de pregão)
TRAIN_WINDOW = 252

# Janela de teste (aproximadamente 1 mês)
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
# Usamos poucos valores para comparar baixa, media e alta penalizacao de risco.
#gammas = [1, 5, 10]
gammas = [5]

# Lambda controla a regularizacao dos pesos da carteira.
# Mantemos poucos valores para evitar uma busca de hiperparametros excessiva.
#lambdas = [0.01, 0.1, 1]

lambdas = [0.1]

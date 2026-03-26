import os 

# Define o diretório base do projeto (onde este arquivo está localizado)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Define o diretório de saída (output) dentro do diretório base
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Cria o diretório de saída caso ele não exista (evita erro se já existir)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lista de tickers (ações brasileiras no Yahoo Finance)
TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"]

# Data inicial da análise
START_DATE = "2020-01-31"

# Data final da análise
END_DATE = "2025-01-31"

# Janela de treino (aproximadamente 1 anos de pregão)
TRAIN_WINDOW = 252

# Janela de teste (aproximadamente 1 mês)
TEST_WINDOW = 21

# Taxa livre de risco (ex: 2% ao ano)
RISK_FREE_RATE = 0.02

# Parâmetro de regularização para otimização de portfólio (Markowitz)
LAMBDA_REG = 0.1  # regularização Markowitz

import os 

print("[DEBUG] Configurando diretórios...")

# Define o diretório base do projeto (onde este arquivo está localizado)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
print(f"[DEBUG] BASE_DIR definido como: {BASE_DIR}")

# Define o diretório de saída (output) dentro do diretório base
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Cria o diretório de saída caso ele não exista (evita erro se já existir)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[DEBUG] OUTPUT_DIR pronto em: {OUTPUT_DIR}")


# CONFIGURAÇÃO DE ATIVOS

# Lista de tickers (ações brasileiras no Yahoo Finance)
TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA",
    "BBDC4.SA", "ABEV3.SA"
]
print(f"[DEBUG] Total de ativos configurados: {len(TICKERS)}")


# PERÍODO DE ANÁLISE

# Data inicial da análise
START_DATE = "2020-01-31"

# Data final da análise
END_DATE = "2025-01-31"
print(f"[DEBUG] Período: {START_DATE} até {END_DATE}")


# JANELAS DE TREINO E TESTE

# Janela de treino (aproximadamente 2 anos de pregões)
TRAIN_WINDOW = 252 * 2

# Janela de teste (aproximadamente 1 mês)
TEST_WINDOW = 21
print(f"[DEBUG] TRAIN_WINDOW: {TRAIN_WINDOW}, TEST_WINDOW: {TEST_WINDOW}")


# PARÂMETROS FINANCEIROS

# Taxa livre de risco (ex: 2% ao ano)
RISK_FREE_RATE = 0.02

# Parâmetro de regularização para otimização de portfólio (Markowitz)
LAMBDA_REG = 0.1  # regularização Markowitz

print("[DEBUG] Configurações carregadas com sucesso.")
import os 

# =========================
# PATH OUTPUT
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA",
    "BBDC4.SA", "ABEV3.SA"
]

START_DATE = "2020-01-01"
END_DATE = "2024-01-01"

TRAIN_WINDOW = 252 * 2
TEST_WINDOW = 21

RISK_FREE_RATE = 0.02

LAMBDA_REG = 0.1  # regularização Markowitz
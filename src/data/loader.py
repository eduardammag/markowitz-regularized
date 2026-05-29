import os 
import pandas as pd
import yfinance as yf
import hashlib

from src.data.yahoo import configure_yfinance

# Função interna para gerar um nome único de arquivo de cache
def _generate_cache_name(tickers, start, end):
    # Ordena os tickers e junta em uma string única
    tickers_str = "_".join(sorted(tickers))

    # Cria uma chave única com tickers + período
    key = f"{tickers_str}_{start}_{end}"

    # Gera um hash MD5 da chave (reduz tamanho e evita nomes muito longos)
    hash_key = hashlib.md5(key.encode()).hexdigest()[:8]

    # Retorna o caminho do arquivo de cache
    return f"data/cache_{hash_key}.parquet"


# Função principal para carregar os dados
def load_data(tickers, start, end, force_download=False):
    
    # Gera o caminho do cache baseado nos parâmetros
    cache_path = _generate_cache_name(tickers, start, end)
    # LOAD CACHE
    # Verifica se o arquivo de cache existe e se não é forçado novo download
    if os.path.exists(cache_path) and not force_download:
        # Lê os dados salvos em formato parquet
        prices = pd.read_parquet(cache_path)
    else:
        print("[INFO] Baixando dados do Yahoo Finance...")
        configure_yfinance()

        # Baixa os dados históricos dos ativos
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        # Com auto_adjust=True, o Yahoo ja devolve "Close" ajustado.
        # Alguns downloads com falha ainda criam "Adj Close" vazio; por isso
        # preferimos "Close" quando ele existe.
        price_level = data.columns.get_level_values(0)
        price_type = "Close" if "Close" in price_level else "Adj Close"

        # Seleciona os preços desejados
        prices = data[price_type].copy()
        # Remove colunas completamente vazias
        prices = prices.dropna(axis=1, how="all")

        # Preenche valores faltantes para frente e remove linhas restantes com NaN
        prices = prices.ffill().dropna()
            # SAVE CACHE
            # Garante que o diretório "data" exista
        os.makedirs("data", exist_ok=True)

        # Salva os dados em formato parquet para uso futuro
        prices.to_parquet(cache_path)
    # RETURNS
    # Calcula os retornos percentuais
    returns = prices.pct_change().dropna()
    return returns


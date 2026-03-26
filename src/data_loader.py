import os 
import pandas as pd
import yfinance as yf
import hashlib


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
    
    print("[DEBUG] Iniciando load_data...")

    # Gera o caminho do cache baseado nos parâmetros
    cache_path = _generate_cache_name(tickers, start, end)
    print(f"[DEBUG] Caminho do cache: {cache_path}")

    # LOAD CACHE
    # Verifica se o arquivo de cache existe e se não é forçado novo download
    if os.path.exists(cache_path) and not force_download:
        print(f" Carregando cache: {cache_path}")

        # Lê os dados salvos em formato parquet
        prices = pd.read_parquet(cache_path)
        print(f"[DEBUG] Dados carregados do cache: {prices.shape}")

    else:
        print(" Baixando dados do Yahoo Finance...")

        # Baixa os dados históricos dos ativos
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)
        print("[DEBUG] Download concluído.")

        # Define qual tipo de preço usar (preferência por 'Adj Close')
        price_type = "Adj Close" if "Adj Close" in data.columns.levels[0] else "Close"

        # Caso não exista 'Adj Close', avisa o usuário
        if price_type != "Adj Close":
            print(" 'Adj Close' não encontrado. Usando 'Close'.")

        # Seleciona os preços desejados
        prices = data[price_type].copy()
        print(f"[DEBUG] Preços selecionados: {prices.shape}")

        # Remove colunas completamente vazias
        prices = prices.dropna(axis=1, how="all")

        # Preenche valores faltantes para frente e remove linhas restantes com NaN
        prices = prices.ffill().dropna()
        print(f"[DEBUG] Preços após limpeza: {prices.shape}")

            # SAVE CACHE
            # Garante que o diretório "data" exista
        os.makedirs("data", exist_ok=True)

        # Salva os dados em formato parquet para uso futuro
        prices.to_parquet(cache_path)
        print(f" Cache salvo em: {cache_path}")

    # RETURNS
    print("[DEBUG] Calculando retornos...")

    # Calcula os retornos percentuais
    returns = prices.pct_change().dropna()
    print(f"[DEBUG] Retornos calculados: {returns.shape}")

    return returns
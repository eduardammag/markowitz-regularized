"""
Configuracao defensiva para downloads via yfinance no Windows.

Em alguns ambientes, yfinance/curl_cffi falha quando o certificado ou o cache
ficam em caminhos com acentos, ou quando ha proxies locais invalidos herdados
do terminal. Este modulo centraliza os ajustes antes de chamar yf.download.
"""

import os
import shutil
import tempfile

import certifi
import yfinance as yf


def configure_yfinance():
    """
    Prepara cache, certificado e proxies antes de downloads do Yahoo Finance.
    """

    base_dir = os.path.join(tempfile.gettempdir(), "markowitz_yfinance")
    os.makedirs(base_dir, exist_ok=True)

    # Evita bancos SQLite do yfinance em diretorios problemáticos no Windows.
    yf.set_tz_cache_location(base_dir)

    # Evita erro de curl com certificado em caminho Unicode, como "Área".
    cert_path = os.path.join(base_dir, "cacert.pem")
    if not os.path.exists(cert_path):
        shutil.copyfile(certifi.where(), cert_path)

    os.environ.setdefault("SSL_CERT_FILE", cert_path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
    os.environ.setdefault("CURL_CA_BUNDLE", cert_path)

    # Remove apenas o proxy morto observado no ambiente, sem mexer em proxies reais.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        if os.environ.get(key) == "http://127.0.0.1:9":
            os.environ.pop(key, None)

# TCC - Markowitz + Machine Learning

## Objetivo
Combinar previsão de retornos (ML) com otimização de portfólio regularizada.

## Métodos
- Lasso, Ridge, Elastic Net
- Markowitz com regularização L1 + L2
- Covariância via Ledoit-Wolf

## Métricas
- Sharpe Ratio
- Drawdown

python -m venv venv
venv\Scripts\activate   # ou source venv/bin/activate
pip install -r requirements.txt
python main.py

Pergunta  a ser respondida: 
Modelos de ML melhoram a otimização de portfólio de Modern Portfolio Theory proposta por Harry Markowitz?

Mais especificamente:
Usar Lasso, Ridge e ElasticNet para prever retornos e depois aplicar otimização de portfólio regularizada melhora desempenho comparado a benchmarks?
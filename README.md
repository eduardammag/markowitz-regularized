# Machine Learning Portfolio Optimization

Este projeto investiga uma pergunta central:

> Modelos de Machine Learning melhoram a estimacao de retornos esperados e,
> quando usados dentro de Markowitz, geram carteiras com melhor desempenho
> ajustado ao risco do que estrategias tradicionais?

Em vez de testar uma grande quantidade de combinacoes, a versao principal do
repositorio compara um conjunto enxuto de estrategias:

- `historical_mean + Markowitz`: baseline tradicional.
- `lasso + Markowitz`: regressao regularizada L1.
- `ridge + Markowitz`: regressao regularizada L2.
- `elastic + Markowitz`: regressao regularizada L1/L2.
- `random_forest + Markowitz`: modelo nao linear baseado em arvores.
- `gradient_boosting + Markowitz`: modelo nao linear baseado em boosting.
- `xgboost + Markowitz`: modelo nao linear baseado em gradient boosting.
- `equal_weight`: carteira com pesos iguais.
- `ibov`: benchmark de mercado.

## Fluxo do Projeto

```text
precos historicos
-> retornos dos ativos
-> previsao de retorno esperado
-> estimacao de covariancia
-> otimizacao de Markowitz regularizada
-> backtest
-> metricas e graficos
```

## Metodologia

1. Coleta precos historicos via Yahoo Finance.
2. Calcula retornos dos ativos.
3. Treina modelos para prever retornos esperados.
4. Estima a matriz de covariancia com Ledoit-Wolf.
5. Resolve uma carteira de Markowitz com restricoes e regularizacao.
6. Simula a estrategia em janelas de treino e teste.
7. Compara desempenho por Sharpe, retorno acumulado, drawdown e erro de previsao.

## Configuracao Principal

O arquivo `config.py` controla o escopo do experimento.

Atualmente, a busca esta reduzida para:

```python
models = [
    "historical_mean",
    "lasso",
    "ridge",
    "elastic",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]
gammas = [1, 5, 10]
lambdas = [0.01, 0.1, 1]
```

Isso gera 63 estrategias, em vez de centenas de combinacoes, deixando a analise
mais interpretavel para o TCC.

## Estrutura

```text
main.py                      # executa o pipeline completo
config.py                    # parametros do experimento
src/data/                    # carregamento de dados e benchmarks
src/ml_models/               # modelos de previsao de retorno
src/portfolio/               # covariancia e otimizacao de carteira
src/backtesting/             # simulacao temporal
src/evaluation/              # metricas e relatorios
src/visualization/           # graficos finais
```

## Execucao

```bash
python main.py
```

## Dependencias

```bash
pip install -r requirements.txt
```

## Autora

Maria Eduarda Mesquita Magalhaes  
Data Science & Artificial Intelligence  
FGV EMAp

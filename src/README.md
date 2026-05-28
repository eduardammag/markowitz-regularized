# Estrutura do pipeline

Este diretorio esta dividido por etapas do fluxo do projeto.

## 1. data

Carrega dados historicos e cria benchmarks.

- `loader.py`: baixa ou carrega precos em cache e calcula retornos.
- `benchmarks.py`: calcula carteira com pesos iguais e retornos do IBOV.

## 2. ml_models

Contem os modelos que preveem retornos esperados.

- `historical_mean.py`: media historica.
- `linear_regression.py`: regressao linear.
- `lasso.py`: regressao Lasso.
- `ridge.py`: regressao Ridge.
- `elastic_net.py`: Elastic Net.
- `random_forest.py`: Random Forest.
- `gradient_boosting.py`: Gradient Boosting.
- `xgboost_model.py`: XGBoost.
- `feature_engineering.py`: cria lags, media movel e volatilidade.

## 3. portfolio

Transforma previsoes em pesos de carteira.

- `covariance.py`: estima matriz de covariancia via Ledoit-Wolf.
- `optimizer.py`: resolve Markowitz regularizado com restricoes.

## 4. backtesting

Simula a estrategia no tempo.

- `engine.py`: roda janelas de treino/teste, rebalanceia e calcula retornos.

## 5. experiments

Conecta modelo, covariancia, otimizacao e backtest.

- `single_experiment.py`: executa uma combinacao de modelo, gamma e lambda.

## 6. evaluation

Avalia resultados numericos.

- `prediction_metrics.py`: MSE, MAE, acuracia direcional, Sortino, Calmar e turnover.
- `performance.py`: Sharpe, max drawdown e serie de drawdown.
- `report.py`: consolida metricas em tabela.
- `statistical_tests.py`: teste Diebold-Mariano.

## 7. visualization

Gera graficos finais.

- `plots.py`: retorno acumulado, risco vs retorno, drawdown e boxplot.
- `helpers.py`: funcoes auxiliares para salvar graficos e filtrar modelos.

## Fluxo geral

`main.py`
-> `data`
-> `experiments`
-> `ml_models`
-> `portfolio`
-> `backtesting`
-> `evaluation`
-> `visualization`

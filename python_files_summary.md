# Resumo dos arquivos Python do repositório

Este arquivo lista todos os arquivos `.py` do projeto e descreve, em linhas gerais, o papel de cada um.

## Arquivos na raiz

- [config.py](./config.py): Define as configurações do experimento, como os tickers analisados, data inicial/final, tamanho das janelas de treino/teste, limite de peso por ativo, modelos habilitados e parâmetro de aversão ao risco (`gamma`).
- [diagnostics.py](./diagnostics.py): Executa diagnósticos do backtest para verificar, por exemplo, anualização correta do Sharpe, sobreposição de janelas, sensibilidade a custos de transação e concentração dos pesos.
- [main.py](./main.py): Orquestra o pipeline completo: carrega dados, executa os experimentos, calcula benchmarks, gera relatório de performance e salva os gráficos finais.

## Pacote `src`

- [src/__init__.py](./src/__init__.py): Arquivo de inicialização do pacote Python. Em geral, não contém lógica de negócio; apenas marca `src` como módulo Python.

### `src/backtesting`

- [src/backtesting/__init__.py](./src/backtesting/__init__.py): Inicializa o pacote `backtesting`.
- [src/backtesting/engine.py](./src/backtesting/engine.py): Implementa o backtest temporal: divide treino/teste, gera previsões, estima covariância, otimiza pesos e calcula retorno da carteira por janela.

### `src/data`

- [src/data/__init__.py](./src/data/__init__.py): Inicializa o pacote `data`.
- [src/data/yahoo.py](./src/data/yahoo.py): Ajusta configurações defensivas para downloads do Yahoo Finance no Windows (cache, certificados e proxies) antes de chamar `yfinance`.
- [src/data/loader.py](./src/data/loader.py): Carrega preços históricos de ativos, usa cache em formato Parquet e transforma preços em retornos percentuais.
- [src/data/benchmarks.py](./src/data/benchmarks.py): Calcula retornos de benchmarks: carteira equally weighted e índice Ibovespa (
^BVSP). Também agrega retornos por janelas de backtest.

### `src/evaluation`

- [src/evaluation/__init__.py](./src/evaluation/__init__.py): Inicializa o pacote `evaluation`.
- [src/evaluation/performance.py](./src/evaluation/performance.py): Calcula métricas financeiras como Sharpe, drawdown e desenho do drawdown ao longo do tempo.
- [src/evaluation/prediction_metrics.py](./src/evaluation/prediction_metrics.py): Mede qualidade preditiva por meio de MSE, MAE, acurácia direcional, Sortino, Calmar e turnover de pesos.
- [src/evaluation/report.py](./src/evaluation/report.py): Gera um DataFrame consolidado com métricas por estratégia para comparação dos modelos.
- [src/evaluation/statistical_tests.py](./src/evaluation/statistical_tests.py): Implementa testes estatísticos, como Diebold-Mariano, para comparar erros de previsão entre modelos.

### `src/experiments`

- [src/experiments/__init__.py](./src/experiments/__init__.py): Inicializa o pacote `experiments`.
- [src/experiments/single_experiment.py](./src/experiments/single_experiment.py): Executa um experimento individual para um dado modelo e parâmetro `gamma`, retornando métricas de previsão e performance da carteira.

### `src/ml_models`

- [src/ml_models/__init__.py](./src/ml_models/__init__.py): Centraliza o registro dos modelos treináveis e expõe a função `predict_returns(...)`.
- [src/ml_models/feature_engineering.py](./src/ml_models/feature_engineering.py): Constrói as features usadas pelos modelos supervisionados: lags, média móvel de 20 dias e volatilidade móvel de 20 dias; também monta o dataset supervisado.
- [src/ml_models/historical_mean.py](./src/ml_models/historical_mean.py): Implementa o baseline de média histórica, assumindo que a média passada do retorno é a melhor previsão para o futuro.
- [src/ml_models/lasso.py](./src/ml_models/lasso.py): Treina uma regressão Lasso com penalização L1 para prever retornos futuros.
- [src/ml_models/ridge.py](./src/ml_models/ridge.py): Treina uma regressão Ridge com penalização L2 para prever retornos futuros.
- [src/ml_models/elastic_net.py](./src/ml_models/elastic_net.py): Treina um modelo Elastic Net combinando penalizações L1 e L2.
- [src/ml_models/random_forest.py](./src/ml_models/random_forest.py): Treina um Random Forest Regressor para capturar relações não lineares entre atributos e retornos futuros.
- [src/ml_models/gradient_boosting.py](./src/ml_models/gradient_boosting.py): Treina um Gradient Boosting Regressor em um setup multioutput para prever os retornos de todos os ativos.
- [src/ml_models/xgboost_model.py](./src/ml_models/xgboost_model.py): Treina um modelo XGBoost para previsão multioutput de retornos com base em gradient boosting regularizado.

### `src/portfolio`

- [src/portfolio/__init__.py](./src/portfolio/__init__.py): Inicializa o pacote `portfolio`.
- [src/portfolio/covariance.py](./src/portfolio/covariance.py): Estima a matriz de covariância dos retornos por meio do método de Ledoit-Wolf.
- [src/portfolio/optimizer.py](./src/portfolio/optimizer.py): Resolve a otimização de Markowitz com restrições de orçamento, ausência de short e limite máximo por ativo usando CVXPY.

### `src/visualization`

- [src/visualization/__init__.py](./src/visualization/__init__.py): Inicializa o pacote `visualization`.
- [src/visualization/helpers.py](./src/visualization/helpers.py): Define funções auxiliares para salvar figuras em `output/<subfolder>`, além de utilitários para filtrar modelos.
- [src/visualization/plots.py](./src/visualization/plots.py): Produz os gráficos principais do estudo: retorno acumulado, drawdown, risco x retorno, barras de performance e pesos médios por ativo.

## Conclusão

A estrutura geral do projeto está organizada em quatro grandes blocos:

1. carregamento e tratamento de dados;
2. previsão de retornos por modelos de machine learning;
3. estimativa de risco e otimização de carteira;
4. avaliação financeira, testes estatísticos e visualização dos resultados.

Essa separação facilita a manutenção, a reprodutibilidade e a comparação de estratégias de portfólio no contexto do TCC.

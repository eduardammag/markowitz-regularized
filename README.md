# Machine Learning Portfolio Optimization

Este projeto implementa um pipeline completo de construção, previsão e
otimização de portfólios financeiros utilizando Machine Learning.

O sistema utiliza modelos de regressão regularizada para prever retornos
de ativos e aplicar otimização de portfólio baseada em média-variância
(Markowitz), avaliando o desempenho através de backtesting histórico.

O objetivo é comparar diferentes combinações de:
-   Modelos de Machine Learning
-   Parâmetros de risco
-   Regularização do portfólio

e analisar qual estratégia produz melhor desempenho ajustado ao risco.
------------------------------------------------------------------------

## Metodologia

O pipeline segue as seguintes etapas:

1.  Coleta de dados financeiros
2.  Pré-processamento e cálculo de retornos
3.  Treinamento de modelos de Machine Learning
4.  Previsão de retornos
5.  Estimativa da matriz de covariância
6.  Otimização do portfólio
7.  Backtesting da estratégia
8.  Avaliação estatística
9.  Visualização dos resultados

------------------------------------------------------------------------

## Estrutura do Projeto

    project/

    config.py
    main.py
    README.md

    src/
    data_loader.py
    ml_models.py
    optimizer.py
    covariance.py
    backtest.py
    benchmarks.py
    metrics.py
    stat_tests.py
    report.py
    analysis.py
    single_experiment.py
    utils.py

------------------------------------------------------------------------

##  Configuração do Projeto

O arquivo config.py contém os parâmetros principais do experimento:

-   Ativos analisados
-   Período de dados
-   Parâmetros do backtest
-   Janelas de treinamento

------------------------------------------------------------------------

## Etapas do Pipeline

1️⃣ Carregamento de Dados

Arquivo:

src/data/data_loader.py

Responsável por:

-   baixar preços históricos
-   limpar dados
-   calcular retornos logarítmicos ou simples

Função principal:

load_data(tickers, start_date, end_date)

Saída:

DataFrame onde: - linhas = datas - colunas = ativos - valores = retornos

------------------------------------------------------------------------

2️⃣ Engenharia de Features

Arquivo:

src/data/features.py

Aqui podem ser criadas variáveis adicionais:

-   médias móveis
-   momentum
-   volatilidade histórica
-   indicadores técnicos

------------------------------------------------------------------------

3️⃣ Modelos de Machine Learning

Arquivo:

src/models/ml_models.py

Modelos utilizados:

-   Ridge Regression (regularização L2)
-   Lasso Regression (regularização L1)
-   Elastic Net (combinação L1 + L2)

Função principal:

predict_returns(data, model_type)

------------------------------------------------------------------------

4️⃣ Estimativa de Covariância

Arquivo:

src/portfolio/covariance.py

Responsável por estimar a matriz de covariância entre os ativos,
representando o risco conjunto.

------------------------------------------------------------------------

5️⃣ Otimização de Portfólio

Arquivo:

src/portfolio/optimizer.py

Baseado na teoria de Markowitz Mean-Variance Optimization.

Objetivo:

maximizar

retorno esperado − γ * risco

onde:

γ = coeficiente de aversão ao risco

Função:

optimize_portfolio(mu, covariance, lambda_reg, gamma)

Saída:

pesos ótimos do portfólio.

------------------------------------------------------------------------

6️⃣ Backtesting

Arquivo:

src/backtesting/backtest.py

Processo:

1.  selecionar janela de treinamento
2.  treinar modelo
3.  prever retornos
4.  otimizar portfólio
5.  aplicar pesos
6.  registrar retorno

Função:

run_backtest()

Saídas:

-   retornos do portfólio
-   previsões do modelo
-   valores reais

------------------------------------------------------------------------

7️⃣ Benchmarks

Arquivo:

src/backtesting/benchmarks.py

Estratégias comparativas:

-   Equal Weight Portfolio
-   Índice IBOVESPA

------------------------------------------------------------------------

8️⃣ Métricas de Avaliação

Arquivo:

src/evaluation/metrics.py

Métricas utilizadas:

-   MSE (Mean Squared Error)
-   MAE (Mean Absolute Error)
-   Directional Accuracy

------------------------------------------------------------------------

9️⃣ Testes Estatísticos

Arquivo: src/evaluation/stat_tests.py

Implementa o teste: Diebold-Mariano
Usado para comparar desempenho preditivo entre modelos.

------------------------------------------------------------------------

## Grid Search

Experimentos variam:
Modelos: - lasso - ridge - elastic
Gamma: - 5 - 10 - 20
Lambda: - 0.01 - 0.1 - 1
Total: 3 × 3 × 3 = 27 experimentos
Cada experimento executado em paralelo.

------------------------------------------------------------------------

## Processamento Paralelo

O código utiliza multiprocessing:
Pool(3) para rodar vários experimentos simultaneamente.

------------------------------------------------------------------------

## Análise dos Resultados

Arquivo:

src/analysis/analysis.py

Visualizações:
- Retorno Cumulativo
- Mostra crescimento do capital.
- Risk vs Return

x = volatilidade
y = retorno médio

Drawdown: Mostra maiores perdas acumuladas.
Boxplot: Distribuição dos retornos.

------------------------------------------------------------------------

## Execução do Projeto

Rodar o pipeline completo:

python main.py

O script executa automaticamente:
1.  carregamento de dados
2.  execução de experimentos
3.  avaliação de performance
4.  geração de gráficos

------------------------------------------------------------------------

## Dependências

Bibliotecas principais:
-   numpy
-   pandas
-   matplotlib
-   scikit-learn
-   yfinance
-   scipy

Instalação:

pip install -r requirements.txt

------------------------------------------------------------------------

## Objetivo do Projeto

Investigar se Machine Learning melhora estratégias de portfólio
comparado a abordagens tradicionais.

------------------------------------------------------------------------

## Autora

Maria Eduarda Mesquita Magalhães
Data Science & Artificial Intelligence
FGV EMAp

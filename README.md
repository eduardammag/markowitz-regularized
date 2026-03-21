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

    markowitz-regularized/

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

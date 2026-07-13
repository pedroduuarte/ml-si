# Relatorio Critico: Comparacao dos Modelos de Classificacao

## 1. Objetivo da analise

Este relatorio tem como objetivo justificar qual modelo apresentou o melhor
resultado para o problema de classificacao de churn de clientes de uma empresa
de turismo e viagens.

Foram comparados dois modelos classicos de aprendizagem de maquina:

- KNN com `GridSearchCV`.
- Arvore de Decisao com diferentes valores de profundidade.

O problema analisado e uma classificacao binaria, em que o modelo deve prever se
um cliente apresenta ou nao risco de churn. Nesse cenario, e importante avaliar
nao apenas a acuracia geral, mas tambem a capacidade do modelo de identificar
corretamente os clientes em risco.

## 2. Modelos avaliados

### KNN

O KNN foi treinado com busca de hiperparametros usando `GridSearchCV`. Foram
testados os seguintes valores:

| Parametro | Valores testados |
|---|---|
| `n_neighbors` | 3, 5, 7 |
| `metric` | euclidean, manhattan |

A melhor configuracao encontrada para o KNN foi:

```text
n_neighbors = 5
metric = manhattan
```

### Arvore de Decisao

A Arvore de Decisao foi avaliada com diferentes valores de profundidade maxima:

| Parametro | Valores testados |
|---|---|
| `max_depth` | 3, 5, 7, None |

O modelo escolhido para comparacao final foi:

```text
Decision Tree (depth=5)
```

Essa configuracao foi utilizada por apresentar bom desempenho no conjunto de
teste, por evitar uma arvore totalmente sem limite que poderia aumentar o
risco de overfitting, e por apresentar o mesmo desempenho que depth=7 com
menor complexidade e menor gap entre treino e teste.

## 3. Tabela comparativa de metricas

As metricas obtidas no conjunto de teste foram:

| Modelo | Acuracia | Precisao | Recall | F1-Score |
|---|---:|---:|---:|---:|
| KNN (GridSearch) | 0.8603 | 0.8000 | 0.5581 | 0.6575 |
| Decision Tree (depth=5) | 0.8827 | 0.7500 | 0.7674 | 0.7586 |

Esses resultados tambem estao registrados no arquivo:

```text
results/tabela-metricas.csv
```

O projeto tambem apresenta os graficos:

```text
results/matrizes-de-confusao.png
results/comparacao-metricas.png
```

Esses graficos ajudam a visualizar a diferenca de desempenho entre os modelos,
principalmente na comparacao entre acuracia, precisao, recall e F1-Score.

## 4. Interpretacao das metricas

### Acuracia

A acuracia mede a proporcao total de acertos do modelo.

| Modelo | Acuracia |
|---|---:|
| KNN (GridSearch) | 0.8603 |
| Decision Tree (depth=5) | 0.8827 |

A Arvore de Decisao teve acuracia maior que o KNN. Isso indica que, considerando
todas as classes, ela acertou uma proporcao maior de previsoes no conjunto de
teste.

### Precisao

A precisao mede, entre os clientes classificados como churn, quantos realmente
eram churn.

| Modelo | Precisao |
|---|---:|
| KNN (GridSearch) | 0.8000 |
| Decision Tree (depth=5) | 0.7500 |

Nesse ponto, o KNN foi superior. Isso significa que, quando o KNN aponta um
cliente como churn, ele tem uma taxa de acerto maior nessa classe positiva.
Porem, analisar apenas a precisao pode ser insuficiente para esse problema.

### Recall

O recall mede, entre os clientes que realmente eram churn, quantos o modelo
conseguiu identificar.

| Modelo | Recall |
|---|---:|
| KNN (GridSearch) | 0.5581 |
| Decision Tree (depth=5) | 0.7674 |

Essa e uma metrica muito importante para o cenario escolhido. Em problemas de
churn, deixar de identificar clientes em risco pode significar perder a chance
de realizar uma acao de retencao.

Nesse criterio, a Arvore de Decisao foi claramente melhor. Ela conseguiu
identificar uma proporcao maior dos clientes que realmente apresentavam risco de
churn.

### F1-Score

O F1-Score combina precisao e recall em uma unica metrica. Ele e util quando
queremos avaliar o equilibrio entre acertar os positivos previstos e encontrar a
maior quantidade possivel de positivos reais.

| Modelo | F1-Score |
|---|---:|
| KNN (GridSearch) | 0.6575 |
| Decision Tree (depth=5) | 0.7586 |

A Arvore de Decisao apresentou F1-Score maior. Isso mostra que, mesmo tendo uma
precisao um pouco menor que o KNN, ela teve um desempenho geral mais equilibrado
por causa do recall significativamente superior.

## 5. Discussao sobre o melhor modelo

O melhor modelo para o cenario escolhido foi a Arvore de Decisao com
`max_depth=5`.

A escolha se justifica pelos seguintes pontos:

1. Teve maior acuracia geral.
2. Teve recall muito superior ao KNN.
3. Teve F1-Score superior, indicando melhor equilibrio entre precisao e recall.
4. E mais interpretavel que o KNN, pois suas decisoes podem ser explicadas por
   regras.
5. Para churn, identificar clientes em risco e mais relevante do que apenas ter
   alta precisao nos poucos casos positivos previstos.

Embora o KNN tenha apresentado melhor precisao, seu recall foi baixo. Isso
significa que ele deixou passar muitos clientes que realmente poderiam estar em
risco de churn. Em uma aplicacao de negocio, esse comportamento pode ser
problematico, pois a empresa perderia oportunidades de agir preventivamente.

A Arvore de Decisao, por outro lado, conseguiu identificar melhor os clientes da
classe positiva. Por isso, mesmo com precisao um pouco menor, ela e mais adequada
para o objetivo do projeto.

## 6. Interpretacao da matriz de confusao

A matriz de confusao permite observar os tipos de acertos e erros cometidos por
cada modelo.

No caso do KNN, o resultado mostrou boa capacidade de identificar clientes da
classe negativa, mas menor desempenho na identificacao da classe positiva. Isso
explica o recall mais baixo.

No caso da Arvore de Decisao, houve melhor identificacao da classe positiva, o
que resultou em recall e F1-Score maiores. Esse comportamento e desejavel para
um problema de churn, pois o foco e reconhecer clientes com risco de abandono.

## 7. Conclusao

Com base na tabela de metricas, nos graficos gerados e na interpretacao dos
resultados, o modelo mais adequado para o cenario escolhido foi:

```text
Decision Tree (depth=5)
```

Esse modelo apresentou melhor desempenho geral para o problema de churn,
principalmente por ter maior acuracia, maior recall e maior F1-Score em relacao
ao KNN.

Portanto, a Arvore de Decisao foi escolhida para a fase bonus de deployment e
interface web, pois atende melhor ao objetivo do sistema: identificar clientes
com maior risco de churn e permitir uma simulacao de predicao em tempo real.

# Relatorio do Projeto: Sistema Inteligente de Classificacao de Churn

## 1. Introducao

Este projeto tem como objetivo desenvolver um sistema inteligente de
classificacao capaz de prever o risco de churn de clientes de uma empresa de
turismo e viagens. Churn representa a possibilidade de um cliente abandonar ou
deixar de utilizar um servico. Nesse contexto, identificar clientes com maior
risco permite que a empresa tome decisoes preventivas, como campanhas de
retencao, ofertas personalizadas ou melhoria no relacionamento com o cliente.

O projeto foi desenvolvido com base em algoritmos classicos de aprendizagem de
maquina, utilizando KNN e Arvore de Decisao. Alem das etapas obrigatorias de
limpeza, treinamento e avaliacao, tambem foi implementada uma fase bonus com uma
interface web para simular o uso do modelo por um usuario final.

## 2. Base de dados

A base utilizada e o arquivo `Customertravel.csv`, composto por registros de
clientes de uma empresa de turismo. A variavel alvo e `Target`, que indica se o
cliente representa ou nao um caso de churn.

As principais colunas da base sao:

| Coluna | Descricao |
|---|---|
| `Age` | Idade do cliente |
| `FrequentFlyer` | Indica se o cliente e passageiro frequente |
| `AnnualIncomeClass` | Classe de renda anual |
| `ServicesOpted` | Quantidade de servicos contratados |
| `AccountSyncedToSocialMedia` | Indica se a conta esta sincronizada com rede social |
| `BookedHotelOrNot` | Indica se o cliente reservou hotel |
| `Target` | Classe alvo da classificacao |

Durante a exploracao inicial foram identificadas 954 linhas no dataset original.
Tambem foram observadas 507 linhas duplicadas, representando aproximadamente
53,14% da base. A coluna `FrequentFlyer` continha registros com o valor
`No Record`, tratados como valores ausentes.

## 3. Pre-processamento dos dados

O pre-processamento foi realizado no notebook `clean-data.ipynb` e tambem foi
replicado em `scripts/clean_data.py`, permitindo executar a limpeza pelo
terminal.

As etapas aplicadas foram:

1. Leitura do arquivo bruto `data/Customertravel.csv`.
2. Validacao das colunas obrigatorias.
3. Analise de linhas duplicadas.
4. Substituicao do valor `No Record` por nulo.
5. Remocao das linhas com valores nulos.
6. Conversao das variaveis categoricas para valores numericos.
7. Exportacao da base tratada para `data/cleaned_dataset.csv`.

O encoding foi feito da seguinte forma:

| Variavel | Mapeamento |
|---|---|
| `AnnualIncomeClass` | Low Income = 0, Middle Income = 1, High Income = 2 |
| `FrequentFlyer` | No = 0, Yes = 1 |
| `AccountSyncedToSocialMedia` | No = 0, Yes = 1 |
| `BookedHotelOrNot` | No = 0, Yes = 1 |

Para o treinamento dos modelos, os dados foram divididos em treino e teste na
proporcao 80/20. Tambem foi aplicada padronizacao com `StandardScaler`,
principalmente porque o KNN e sensivel a escala das variaveis.

## 4. Modelagem

Foram treinados e comparados dois modelos de classificacao:

### 4.1 KNN

O KNN classifica uma amostra com base nos vizinhos mais proximos. No projeto,
foi utilizado `GridSearchCV` para testar diferentes valores de `K` e diferentes
metricas de distancia.

Parametros avaliados:

- `n_neighbors`: 3, 5 e 7.
- `metric`: euclidean e manhattan.

Melhor configuracao encontrada:

```text
metric = manhattan
n_neighbors = 5
```

### 4.2 Arvore de Decisao

A Arvore de Decisao cria regras de classificacao a partir dos atributos da base.
Foram testadas diferentes profundidades para avaliar o impacto do parametro
`max_depth` e reduzir o risco de overfitting.

Profundidades avaliadas:

- 3.
- 5.
- 7.
- Sem limite.

Para a fase bonus, foi escolhida a Arvore de Decisao com `max_depth=7`, pois
apresentou melhor equilibrio geral entre as metricas avaliadas.

## 5. Avaliacao dos modelos

A avaliacao foi feita com matriz de confusao e metricas de classificacao. As
metricas consideradas foram acuracia, precisao, recall e F1-Score.

Resultados obtidos:

| Modelo | Acuracia | Precisao | Recall | F1-Score |
|---|---:|---:|---:|---:|
| KNN (GridSearch) | 0.8603 | 0.8000 | 0.5581 | 0.6575 |
| Decision Tree (depth=7) | 0.8827 | 0.7500 | 0.7674 | 0.7586 |

## 6. Interpretacao dos resultados

O KNN apresentou precisao de 0.8000, maior que a da Arvore de Decisao. Isso
significa que, quando o KNN classifica um cliente como churn, ele tende a ter
uma boa proporcao de acerto. No entanto, seu recall foi de 0.5581, indicando que
o modelo deixou de identificar uma parte relevante dos clientes que realmente
pertenciam a classe positiva.

A Arvore de Decisao apresentou acuracia de 0.8827, recall de 0.7674 e F1-Score
de 0.7586. Esses resultados indicam um desempenho geral mais equilibrado,
principalmente para um problema de churn, em que identificar clientes em risco e
mais importante do que apenas evitar falsos alertas.

Por esse motivo, a Arvore de Decisao com profundidade 7 foi escolhida como o
modelo final para a fase de deployment.

## 7. Fase bonus: interface web

A fase bonus consistiu no desenvolvimento de uma interface web simples para
simular o uso do modelo por um usuario final.

Arquivos envolvidos:

- `scripts/train_bonus_model.py`: treina o modelo final e gera o artefato.
- `src/churn_model.py`: contem funcoes de validacao, treino, carregamento e
  predicao.
- `app.py`: cria o servidor web local e expoe a rota `/api/predict`.
- `web/index.html`: estrutura da interface.
- `web/static/styles.css`: estilos visuais.
- `web/static/app.js`: chamada da API e atualizacao da tela em tempo real.

Fluxo da interface:

1. O usuario informa os dados do cliente.
2. A pagina envia os dados para a rota `/api/predict`.
3. O servidor valida os campos recebidos.
4. O modelo salvo em `models/customer_churn_decision_tree.pkl` e carregado.
5. A predicao e retornada para a interface.
6. A tela mostra o risco de churn, probabilidade e metricas do modelo.

A separacao entre treino e interface foi mantida. O modelo e treinado
previamente e a interface apenas carrega o artefato para realizar predicoes.

## 8. Como executar o projeto

### 8.1 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 8.2 Preparar os dados

Colocar o arquivo bruto em:

```text
data/Customertravel.csv
```

Executar:

```bash
python scripts/clean_data.py
```

### 8.3 Treinar e avaliar os modelos

Executar o notebook:

```text
train-models.ipynb
```

### 8.4 Treinar o modelo da fase bonus

```bash
python scripts/train_bonus_model.py
```

### 8.5 Iniciar a interface

```bash
python app.py
```

Abrir no navegador:

```text
http://127.0.0.1:8000
```

## 9. Analise pelos criterios de avaliacao

### Qualidade do codigo

O projeto apresenta boa organizacao, separando limpeza, treinamento, avaliacao e
interface. A criacao de scripts auxiliares facilita a reproducao do processo. A
fase bonus tambem segue uma boa pratica ao separar o treinamento do modelo da
interface de predicao.

### Analise cientifica e metricas

O projeto compara dois algoritmos classicos, testa hiperparametros, calcula
metricas importantes e gera graficos de comparacao. A escolha da Arvore de
Decisao foi baseada principalmente no melhor equilibrio entre recall e F1-Score.

### Relatorio e apresentacao

O projeto possui README, graficos, tabela de metricas, guia de estudo e este
relatorio. Esses materiais ajudam a explicar a implementacao, justificar as
decisoes tomadas e apresentar os resultados de forma objetiva.

## 10. Limitacoes e melhorias futuras

Alguns pontos podem ser melhorados em futuras versoes:

1. Explicar melhor a decisao sobre as linhas duplicadas ou implementar sua
   remocao no pipeline de limpeza.
2. Adicionar testes automatizados para as funcoes de limpeza, validacao e
   predicao.
3. Comparar tambem o modelo Naive Bayes, caso seja desejado ampliar a analise.
4. Adicionar validacao cruzada para a Arvore de Decisao.
5. Criar um relatorio final em PDF a partir deste documento Markdown.
6. Melhorar a interpretabilidade do modelo exibindo as regras ou importancia das
   variaveis.

## 11. Conclusao

O projeto atinge o objetivo proposto ao implementar um sistema de classificacao
para prever churn em clientes de viagens. Foram realizadas as etapas de
pre-processamento, treinamento, avaliacao e comparacao de modelos. A Arvore de
Decisao com profundidade 7 foi selecionada como modelo final por apresentar
melhor equilibrio entre as metricas, principalmente recall e F1-Score.

A interface web implementada na fase bonus torna o projeto mais completo, pois
permite simular a entrada de dados por um usuario e visualizar a predicao em
tempo real. Dessa forma, o trabalho demonstra tanto a parte cientifica de
machine learning quanto a aplicacao pratica do modelo em uma experiencia
interativa.

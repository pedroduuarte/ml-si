# Analise do Projeto e Guia de Estudo da Implementacao

## 1. Visao geral do projeto

O projeto implementa um sistema inteligente de classificacao para prever churn de
clientes de uma empresa de turismo e viagens. A base usada e o dataset
`Customertravel.csv`, que contem informacoes como idade, frequencia de viagem,
classe de renda, quantidade de servicos contratados, sincronizacao com rede
social, reserva de hotel e a variavel alvo `Target`.

O fluxo principal foi dividido em quatro partes:

1. Limpeza e preparacao dos dados.
2. Treinamento e comparacao de modelos.
3. Avaliacao por metricas e graficos.
4. Fase bonus com interface web para predicao em tempo real.

Arquivos principais:

- `clean-data.ipynb`: notebook de exploracao, limpeza e encoding.
- `scripts/clean_data.py`: versao executavel pelo terminal da limpeza.
- `train-models.ipynb`: notebook de treino, comparacao e metricas.
- `src/churn_model.py`: modulo reutilizavel para validacao, treino e predicao.
- `scripts/train_bonus_model.py`: gera o artefato do modelo usado pela interface.
- `app.py`: servidor web local com API de predicao.
- `web/`: interface do usuario.
- `results/`: tabela e graficos de avaliacao.

## 2. Criterios de avaliacao

### 2.1 Qualidade do codigo - 30%

Pontos fortes:

- O projeto separa bem os conceitos principais:
  - limpeza em `clean-data.ipynb` e `scripts/clean_data.py`;
  - treino e avaliacao em `train-models.ipynb`;
  - predicao e artefato do modelo em `src/churn_model.py`;
  - interface web em `app.py` e `web/`.
- A fase bonus respeita a recomendacao do enunciado: o treino fica separado da
  interface.
- O script `clean_data.py` permite executar a limpeza pelo terminal sem depender
  diretamente do Jupyter.
- O modulo `churn_model.py` valida os dados de entrada antes da predicao,
  evitando entradas invalidas como idade vazia, categorias desconhecidas ou
  quantidade de servicos fora do intervalo esperado.
- O modelo treinado e salvo em `models/customer_churn_decision_tree.pkl`, e os
  metadados ficam em um JSON separado.

Pontos de atencao:

- O notebook identifica 507 linhas duplicadas, mas a versao atual apenas analisa
  essas duplicadas; ela nao remove duplicatas. Como o criterio cita tratamento de
  duplicados, e importante explicar a decisao ou ajustar o codigo para remover
  duplicatas, se essa for a abordagem escolhida.
- O notebook `clean-data.ipynb` calcula `train_test_split` e `StandardScaler`,
  mas exporta o dataset limpo sem as colunas escaladas. Isso nao invalida o
  projeto, pois o treino faz novamente o split e o scaler, mas vale explicar que
  o treino definitivo ocorre em `train-models.ipynb`.
- Os notebooks ainda tem parte da analise em formato exploratorio. Para entrega
  final, e bom manter textos explicativos mais objetivos em celulas Markdown.

Resumo avaliativo:

O criterio de qualidade do codigo esta bem atendido pela separacao em scripts,
modulos e interface. O principal ajuste recomendado e formalizar melhor a
decisao sobre duplicatas.

### 2.2 Analise cientifica e metricas - 40%

Modelos implementados:

- KNN com `GridSearchCV`.
- Arvore de Decisao com teste de profundidades diferentes.

Pre-processamento usado:

- Tratamento de valores `"No Record"` como nulos.
- Remocao de linhas com valores nulos.
- Encoding manual de variaveis categoricas:
  - `AnnualIncomeClass`: Low = 0, Middle = 1, High = 2.
  - `FrequentFlyer`: No = 0, Yes = 1.
  - `AccountSyncedToSocialMedia`: No = 0, Yes = 1.
  - `BookedHotelOrNot`: No = 0, Yes = 1.
- Divisao treino/teste em 80/20.
- Padronizacao com `StandardScaler`.

Metricas geradas:

| Modelo | Acuracia | Precisao | Recall | F1-Score |
|---|---:|---:|---:|---:|
| KNN (GridSearch) | 0.8603 | 0.8000 | 0.5581 | 0.6575 |
| Decision Tree (depth=7) | 0.8827 | 0.7500 | 0.7674 | 0.7586 |

Interpretacao:

- O KNN teve precisao maior na classe positiva: quando ele indica churn, tende a
  errar menos proporcionalmente.
- A Decision Tree teve recall maior: ela identifica uma parte maior dos clientes
  que realmente podem dar churn.
- Para o problema de churn, recall e F1-Score sao especialmente importantes,
  porque deixar de identificar um cliente em risco pode significar perder uma
  oportunidade de retencao.
- A Decision Tree `max_depth=7` foi escolhida para o deploy por apresentar melhor
  equilibrio geral, com F1-Score superior ao KNN.

Resumo avaliativo:

O criterio de analise cientifica esta bem atendido porque o projeto compara dois
modelos, testa hiperparametros, gera matriz de confusao, calcula metricas e usa
graficos para comparacao. O ponto a reforcar e a interpretacao textual das
metricas no notebook ou em um relatorio final.

### 2.3 Relatorio e apresentacao - 30%

Pontos fortes:

- O README explica a estrutura do projeto e o fluxo de execucao.
- Os graficos em `results/` ajudam na apresentacao visual.
- A interface web deixa a fase bonus demonstravel para a banca/professor.
- A tabela `results/tabela-metricas.csv` facilita a comparacao objetiva dos
  modelos.

Pontos de atencao:

- O notebook `train-models.ipynb` possui uma secao de relatorio critico
  provisoria. Para uma entrega final, essa secao deve explicar claramente:
  - qual modelo foi escolhido;
  - por que ele foi escolhido;
  - quais metricas sustentam essa decisao;
  - quais limitacoes existem na base e no modelo.
- A apresentacao deve deixar claro que a predicao e uma simulacao educacional,
  nao uma decisao automatica real de negocio.

Resumo avaliativo:

O projeto tem boa base para apresentacao. Para melhorar a nota nesse criterio,
vale transformar a analise critica em um texto final mais direto e conectado aos
graficos e metricas.

### 2.4 Fase bonus: deployment e interface

Implementacao:

- `scripts/train_bonus_model.py` treina a Decision Tree e salva o artefato.
- `app.py` cria um servidor HTTP local.
- `web/index.html`, `web/static/styles.css` e `web/static/app.js` formam a
  interface.
- O usuario informa os dados do cliente e a pagina chama `/api/predict`.
- A resposta mostra:
  - classificacao: baixo ou alto risco de churn;
  - probabilidade de churn, quando disponivel;
  - modelo usado;
  - metricas principais do modelo.

Pontos fortes:

- A interface cumpre exatamente o objetivo da fase bonus: simular um usuario
  final inserindo dados e recebendo predicao em tempo real.
- A interface nao re-treina o modelo a cada predicao, o que e uma boa pratica.
- O sistema retorna mensagens claras quando o modelo ainda nao foi treinado.

## 3. Fluxo completo da implementacao

### Passo 1: baixar a base

Baixar o dataset `Customertravel.csv` e colocar em:

```text
data/Customertravel.csv
```

### Passo 2: limpar os dados

Executar:

```bash
python scripts/clean_data.py
```

Saida esperada:

```text
data/cleaned_dataset.csv
```

O script faz:

- leitura do CSV bruto;
- validacao das colunas obrigatorias;
- substituicao de `"No Record"` por nulo;
- remocao de linhas com nulos;
- encoding das variaveis categoricas;
- exportacao do CSV limpo.

### Passo 3: treinar e avaliar modelos

Executar `train-models.ipynb`.

Esse notebook faz:

- leitura de `data/cleaned_dataset.csv`;
- divisao treino/teste;
- padronizacao dos dados;
- treino do KNN;
- busca dos melhores parametros do KNN;
- treino de arvores com profundidades diferentes;
- matriz de confusao;
- tabela de metricas;
- grafico comparativo.

### Passo 4: gerar modelo para a interface

Executar:

```bash
python scripts/train_bonus_model.py
```

Saidas esperadas:

```text
models/customer_churn_decision_tree.pkl
models/customer_churn_decision_tree.metadata.json
```

### Passo 5: iniciar a interface

Executar:

```bash
python app.py
```

Abrir:

```text
http://127.0.0.1:8000
```

## 4. Como explicar o projeto em uma apresentacao

Sugestao de roteiro:

1. O problema escolhido foi prever churn em clientes de viagens.
2. A base contem dados simples de perfil e comportamento do cliente.
3. Primeiro os dados foram limpos e codificados.
4. Depois os dados foram divididos em treino e teste.
5. Foram comparados dois modelos: KNN e Arvore de Decisao.
6. O KNN foi otimizado com `GridSearchCV`.
7. A Arvore de Decisao foi avaliada com diferentes profundidades.
8. As metricas analisadas foram acuracia, precisao, recall e F1-Score.
9. A Decision Tree com profundidade 7 foi escolhida para o deploy por ter melhor
   equilibrio, principalmente em recall e F1-Score.
10. A fase bonus disponibiliza uma interface web local para simular a predicao em
    tempo real.

## 5. Conceitos importantes para estudar

### Churn

Churn significa cancelamento, abandono ou perda de cliente. Em um contexto de
turismo, prever churn ajuda a identificar clientes com risco de deixar de usar o
servico.

### Encoding

Modelos de machine learning geralmente precisam de numeros. Por isso categorias
como `"Yes"`, `"No"`, `"Low Income"` e `"High Income"` foram convertidas para
valores numericos.

### Normalizacao e padronizacao

O `StandardScaler` transforma as variaveis para uma escala comum. Isso e muito
importante para o KNN, pois esse algoritmo depende de distancia entre pontos.

### KNN

O KNN classifica uma nova amostra olhando para os vizinhos mais proximos. No
projeto, foram testados diferentes valores de `K` e diferentes metricas de
distancia.

Melhor configuracao encontrada:

```text
metric = manhattan
n_neighbors = 5
```

### Arvore de Decisao

A Arvore de Decisao cria regras do tipo "se isso, entao aquilo". O parametro
`max_depth` controla a profundidade da arvore. Profundidades muito altas podem
gerar overfitting.

No projeto, a profundidade 7 foi usada no modelo final.

### Matriz de confusao

A matriz de confusao mostra acertos e erros por classe. Em classificacao
binaria, ela ajuda a entender:

- verdadeiros negativos;
- falsos positivos;
- falsos negativos;
- verdadeiros positivos.

### Precisao

Mostra, entre os clientes previstos como churn, quantos realmente eram churn.

### Recall

Mostra, entre os clientes que realmente eram churn, quantos o modelo conseguiu
identificar.

### F1-Score

E uma media harmonica entre precisao e recall. E util quando queremos equilibrar
os dois criterios.

## 6. Perguntas provaveis e respostas curtas

**Por que usar KNN?**

Porque e um algoritmo classico de classificacao, facil de entender e baseado em
similaridade entre clientes.

**Por que usar Arvore de Decisao?**

Porque gera regras interpretaveis e permite analisar o impacto da profundidade
no desempenho.

**Por que a Decision Tree foi escolhida?**

Porque teve F1-Score maior e recall melhor que o KNN, identificando melhor os
clientes com risco de churn.

**Por que o recall importa nesse problema?**

Porque em churn e ruim deixar de identificar clientes em risco. Um falso
negativo pode representar uma oportunidade perdida de retencao.

**Por que separar treino e interface?**

Porque a interface deve apenas carregar um modelo treinado e prever. Treinar
modelo dentro da interface deixaria o sistema lento e desorganizado.

**O projeto esta pronto para producao real?**

Nao. Ele e um projeto academico. Para producao real seriam necessarios mais
testes, validacao com dados novos, monitoramento, seguranca e uma API mais
robusta.

## 7. Melhorias recomendadas antes da entrega

1. Explicar ou implementar a remocao de duplicatas.
2. Completar o relatorio critico dentro de `train-models.ipynb`.
3. Adicionar uma tabela no notebook comparando treino e teste da Arvore de
   Decisao para discutir overfitting.
4. Adicionar um pequeno texto explicando a matriz de confusao de cada modelo.
5. Garantir que o professor consiga rodar o projeto seguindo apenas o README.

## 8. Conclusao

O projeto atende bem ao objetivo do enunciado: resolve um problema real de
classificacao, usa algoritmos classicos, compara metricas e entrega uma interface
bonus funcional. A Decision Tree foi escolhida para o deploy por apresentar
melhor equilibrio entre acuracia, recall e F1-Score.

Para maximizar a avaliacao, o foco final deve ser melhorar a explicacao critica
dos resultados e deixar explicita a decisao sobre linhas duplicadas.

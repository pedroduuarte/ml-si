# Projeto de Machine Learning

Projeto pratico de classificacao para prever churn em clientes de viagens.

O projeto cobre as tres fases obrigatorias do enunciado nos notebooks e adiciona
a fase bonus com uma interface web local para simular o usuario final informando
dados e recebendo a predicao em tempo real.

## Estrutura do projeto

- `notebooks/clean-data.ipynb`: limpeza, tratamento de nulos, encoding e exportacao do dataset limpo.
- `scripts/clean_data.py`: versao executavel pelo terminal da limpeza do notebook.
- `notebooks/train-models.ipynb`: separacao treino/teste, scaling, treino e avaliacao de KNN e Decision Tree.
- `results/`: graficos e tabela de metricas gerados na avaliacao.
- `src/churn_model.py`: regras de validacao, encoding, treino do artefato e predicao.
- `scripts/train_bonus_model.py`: script separado para treinar o modelo usado no deploy.
- `app.py`: servidor web local da fase bonus.
- `web/`: pagina, estilos e JavaScript da interface.
- `models/`: destino dos artefatos gerados pelo treino.
- `data/`: datasets brutos e tratados, ignorados pelo Git.
- `docs/relatorio-projeto.md`: relatorio formal do projeto, com metodologia,
  resultados, interpretacao e conclusao.

## Configuracao do ambiente

1. Crie e ative um ambiente virtual.

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Instale as dependencias.

   ```bash
   pip install -r requirements.txt
   ```

## Como executar as fases obrigatorias

1. Coloque o arquivo bruto em `data/Customertravel.csv`.
2. Execute `notebooks/clean-data.ipynb` ou rode `python scripts/clean_data.py` para gerar `data/cleaned_dataset.csv`.
3. Execute `notebooks/train-models.ipynb` para separar treino/teste, aplicar scaling, treinar e comparar os modelos.
4. Consulte `results/tabela-metricas.csv` e os graficos em `results/`.

## Fase Bonus: deployment e interface

O treino fica separado da interface. Primeiro gere o artefato do modelo:

```bash
python scripts/train_bonus_model.py
```

Depois inicie a aplicacao web local:

```bash
python app.py
```

Abra no navegador:

```text
http://127.0.0.1:8000
```

A interface envia os dados para `/api/predict` e atualiza a predicao sem
recarregar a pagina. Se o modelo ainda nao tiver sido treinado, a tela mostra
`Modelo pendente` e a API retorna a instrucao para executar o script de treino.

## Modelo usado no deploy

A fase bonus usa `Decision Tree (depth=5)`, seguindo a avaliacao atual do
notebook: ela teve o melhor equilibrio entre F1-Score e Recall na classe
positiva em comparacao com o KNN.

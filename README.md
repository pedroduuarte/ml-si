# Projeto de Machine Learning

Este projeto contém notebooks para limpeza de dados e treinamento de modelos de Machine Learning.

## Estrutura do Projeto

- `clean-data.ipynb`: Notebook responsável pelo pré-processamento, limpeza e preparação dos dados.
- `train-models.ipynb`: Notebook focado na criação, treinamento e avaliação dos modelos preditivos.
- `data/`: Diretório onde os conjuntos de dados (datasets) brutos e processados devem ser armazenados (ignorado pelo versionamento).

## Configuração do Ambiente

1. Certifique-se de ter o Python instalado.
2. Ative o ambiente virtual (já configurado no diretório `.venv`):
   ```bash
   # No Windows
   .\.venv\Scripts\activate
   ```
3. Instale as dependências necessárias caso haja um arquivo de requisitos, ou execute as células nos notebooks que instalam os pacotes necessários (ex: pandas, scikit-learn, matplotlib).

## Como Executar

1. Abra o Jupyter Notebook ou a sua IDE (ex: VS Code) na pasta do projeto.
2. Execute primeiro o `clean-data.ipynb` para gerar a base de dados tratada.
3. Em seguida, execute o `train-models.ipynb` para treinar os modelos usando a base tratada.

# 🎬 Sistema de Recomendação de Filmes

Projeto desenvolvido para as disciplinas de **Mineração de Dados** e **Testes de Software**.

O sistema recomenda filmes personalizados com base nas avaliações feitas pelos usuários, utilizando três abordagens de aprendizado de máquina: **SVD (Filtragem Colaborativa)**, **Regressão Logística** e **Rede Neural com Embeddings**.

Dataset utilizado: [MovieLens](https://grouplens.org/datasets/movielens/)

---

## 📁 Estrutura do Projeto

```
sistema_recomendacao/
├── data_loader.py           # Carregamento e limpeza dos dados
├── preprocessamento.py      # Pré-processamento e divisão treino/teste
├── recomendacao_svd.py      # Recomendador baseado em SVD
├── recomendacao_lr.py       # Recomendador baseado em Regressão Logística
├── recomendacao_nn.py       # Recomendador baseado em Rede Neural
├── __init__.py
├── EntregaFinal.ipynb       # Notebook original da disciplina de Mineração de Dados
├── requirements.txt
├── README.md
└── tests/
    ├── test_data_loader.py        # Testes unitários — DataLoader
    ├── test_preprocessamento.py   # Testes unitários — Preprocessador
    ├── test_recomendacao_svd.py   # Testes unitários — RecomendadorSVD
    ├── test_recomendacao_lr.py    # Testes unitários — RecomendadorRegressaoLogistica
    └── test_integracao.py         # Testes de integração — Pipeline completo
```

---

## ⚙️ Como instalar

> Requer Python 3.9 ou superior.

**1. Clone o repositório:**
```bash
git clone https://github.com/<seu-usuario>/sistema_recomendacao.git
cd sistema_recomendacao
```

**2. Crie um ambiente virtual (recomendado):**
```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

**3. Instale as dependências:**
```bash
pip install -r requirements.txt
```

> ⚠️ O TensorFlow é necessário apenas para o módulo `recomendacao_nn.py`. Os demais módulos funcionam sem ele.

---

## 📦 Dataset

Baixe o dataset MovieLens e coloque os arquivos na raiz do projeto:

```
sistema_recomendacao/
├── ratings.csv
└── movies.csv
```

Download: https://grouplens.org/datasets/movielens/latest/

---

## 🚀 Como usar

```python
from data_loader import DataLoader
from preprocessamento import Preprocessador
from recomendacao_svd import RecomendadorSVD

# 1. Carregar dados
loader = DataLoader("ratings.csv", "movies.csv")
ratings_df = loader.carregar_ratings()
movies_df  = loader.carregar_movies()
movies_df  = loader.aplicar_one_hot_genres(movies_df)

# 2. Pré-processar
import pandas as pd
prep = Preprocessador()
df_merged = pd.merge(ratings_df, movies_df, on="movieId")
df_filtrado = loader.filtrar_usuarios_e_filmes(df_merged)
train, test = prep.dividir_treino_teste(df_filtrado)
matriz = prep.construir_matriz_usuario_item(train)

# 3. Treinar e recomendar
rec = RecomendadorSVD(n_components=20)
rec.treinar(matriz)

recomendacoes = rec.recomendar(user_id=1, movies_df=movies_df, n_recomendacoes=10)
print(recomendacoes)
```

---

## 🧪 Como executar os testes

**Todos os testes de uma vez:**
```bash
python3 -m unittest discover -s tests -v
```

**Por arquivo:**
```bash
# Testes unitários
python3 -m unittest tests/test_data_loader.py -v
python3 -m unittest tests/test_preprocessamento.py -v
python3 -m unittest tests/test_recomendacao_svd.py -v
python3 -m unittest tests/test_recomendacao_lr.py -v

# Testes de integração
python3 -m unittest tests/test_integracao.py -v
```

**Salvar resultado em arquivo:**
```bash
python3 -m unittest discover -s tests -v 2>&1 | tee resultado_testes.txt
```

---

## 🧩 Módulos

### `DataLoader`
Responsável por carregar e limpar os arquivos CSV do MovieLens.

| Método | Descrição |
|---|---|
| `carregar_ratings()` | Lê ratings.csv, valida colunas, remove nulos |
| `carregar_movies()` | Lê movies.csv, remove duplicatas de título, extrai ano |
| `aplicar_one_hot_genres()` | Aplica one-hot encoding nos gêneros |
| `filtrar_usuarios_e_filmes()` | Remove usuários/filmes com poucas avaliações |

### `Preprocessador`
Transforma os dados para alimentar os modelos.

| Método | Descrição |
|---|---|
| `dividir_treino_teste()` | Divide os dados em treino e teste |
| `construir_matriz_usuario_item()` | Cria matriz pivot usuário × filme |
| `criar_coluna_liked()` | Binariza a nota em 0/1 por threshold |
| `criar_mapeamentos()` | Cria índices consecutivos para embeddings |

### `RecomendadorSVD`
Filtragem colaborativa via SVD truncado (scikit-learn).

| Método | Descrição |
|---|---|
| `treinar(matriz)` | Decompõe a matriz e reconstrói notas previstas |
| `recomendar(user_id, movies_df, n)` | Retorna top-N filmes não avaliados |
| `calcular_rmse(test_data)` | Calcula o erro quadrático médio no teste |

### `RecomendadorRegressaoLogistica`
Classificação binária: prevê se o usuário vai gostar do filme.

| Método | Descrição |
|---|---|
| `treinar(X_train, y_train, ...)` | Treina o modelo de regressão logística |
| `avaliar(X_test, y_test)` | Retorna accuracy e AUC-ROC |
| `recomendar(user_id, movies_df, n)` | Recomenda filmes por probabilidade |

### `RecomendadorRedeNeural`
Rede neural com embeddings de usuário e filme (TensorFlow/Keras).

| Método | Descrição |
|---|---|
| `treinar(train_df, user_map, ...)` | Treina a rede com embeddings |
| `recomendar(user_id, movies_df, n)` | Recomenda filmes via probabilidade da rede |

---

## 🧪 Cobertura de Testes

| Arquivo de Teste | Tipo | Qtd. Testes | Mockado | Parametrizado |
|---|---|---|---|---|
| `test_data_loader.py` | Unitário | 14 | ✅ Sim | ✅ Sim |
| `test_preprocessamento.py` | Unitário | 15 | ✅ Sim | ✅ Sim |
| `test_recomendacao_svd.py` | Unitário | 12 | ✅ Sim | ✅ Sim |
| `test_recomendacao_lr.py` | Unitário | 12 | ✅ Sim | ✅ Sim |
| `test_integracao.py` | Integração | 4 | ✅ Parcial | ✅ Sim |

**Total: 57 casos de teste**

---

## 📋 Requisitos Não Funcionais

- **Desempenho:** Recomendações geradas em menos de 3 segundos para bases de até 100 mil avaliações
- **Reprodutibilidade:** Todos os modelos usam `random_state=42` para resultados consistentes
- **Manutenibilidade:** Código modularizado em classes com responsabilidade única
- **Testabilidade:** Todos os módulos são independentes de arquivos externos nos testes (mock por padrão)

---

## 👥 Equipe

Desenvolvido para as disciplinas de Mineração de Dados e Testes de Software.

---

## 📄 Licença

Este projeto é de uso acadêmico.

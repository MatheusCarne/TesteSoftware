"""
Testes de Integração — Pipeline Completo do Sistema de Recomendação
Arquivo: test_integracao.py

Testa o fluxo completo de ponta a ponta, verificando a integração entre
os módulos DataLoader, Preprocessador e os Recomendadores.

IT-01: Pipeline DataLoader → Preprocessador → RecomendadorSVD
    Verifica que o fluxo completo de carregamento, pré-processamento
    e recomendação funciona corretamente em sequência.

IT-02: Pipeline Preprocessador → RecomendadorSVD → calcular_rmse()
    Verifica que o modelo treinado gera RMSE válido sobre o conjunto de teste.

IT-03: Pipeline DataLoader → Preprocessador → RecomendadorRegressaoLogistica
    Verifica que o modelo de regressão logística é treinado e avaliado
    corretamente dentro do pipeline completo.

IT-04: Consistência das recomendações (mesmo usuário, mesmo resultado)
    Verifica determinismo: chamadas repetidas retornam os mesmos filmes.

Parametrizado: Sim (subTest para múltiplos usuários no IT-04)
Mockado: Sim para DataLoader (CSVs temporários); dados reais simulados com
         tamanho representativo (300+ registros) para testar a integração real.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader
from preprocessamento import Preprocessador
from recomendacao_svd import RecomendadorSVD
from recomendacao_lr import RecomendadorRegressaoLogistica
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# Helpers para gerar CSVs e DataFrames mock
# ─────────────────────────────────────────────

def gerar_csvs_temporarios(n_users=10, n_movies=20, seed=42):
    """
    Gera arquivos ratings.csv e movies.csv temporários com dados sintéticos
    representativos para testes de integração.
    """
    rng = np.random.default_rng(seed)
    tmp_dir = tempfile.mkdtemp()

    # ratings.csv: cada usuário avalia cada filme uma vez
    rows_r = []
    for uid in range(1, n_users + 1):
        for mid in range(201, 201 + n_movies):
            rows_r.append({
                "userId": uid,
                "movieId": mid,
                "rating": float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])),
                "timestamp": 946684800 + uid * 100 + mid,
            })
    ratings_df = pd.DataFrame(rows_r)
    ratings_path = os.path.join(tmp_dir, "ratings.csv")
    ratings_df.to_csv(ratings_path, index=False)

    # movies.csv: filmes com gêneros variados
    generos = ["Action|Sci-Fi", "Drama|Romance", "Comedy", "Thriller", "Animation|Family"]
    movies_rows = []
    for mid in range(201, 201 + n_movies):
        movies_rows.append({
            "movieId": mid,
            "title": f"Filme {mid} ({1990 + (mid - 201)})",
            "genres": rng.choice(generos),
        })
    movies_df = pd.DataFrame(movies_rows)
    movies_path = os.path.join(tmp_dir, "movies.csv")
    movies_df.to_csv(movies_path, index=False)

    return ratings_path, movies_path, tmp_dir


# ─────────────────────────────────────────────
# IT-01: DataLoader → Preprocessador → SVD
# ─────────────────────────────────────────────

class TestIntegracaoPipelineSVD(unittest.TestCase):
    """
    Teste de Integração IT-01:
    DataLoader → Preprocessador → RecomendadorSVD → recomendar()
    """

    def setUp(self):
        self.ratings_path, self.movies_path, self.tmp_dir = gerar_csvs_temporarios(
            n_users=15, n_movies=25
        )

    def test_pipeline_completo_svd_recomendacao(self):
        """
        [IT-01] O pipeline completo deve carregar dados, pré-processar,
        treinar SVD e retornar recomendações válidas para um usuário.
        """
        # 1. Carregar dados
        loader = DataLoader(self.ratings_path, self.movies_path)
        ratings_df = loader.carregar_ratings()
        movies_df = loader.carregar_movies()
        movies_df = loader.aplicar_one_hot_genres(movies_df)

        # Confirma que os DataFrames foram carregados
        self.assertFalse(ratings_df.empty, "ratings_df não deve estar vazio")
        self.assertFalse(movies_df.empty, "movies_df não deve estar vazio")

        # 2. Pré-processar
        prep = Preprocessador()
        df_merged = pd.merge(ratings_df, movies_df, on="movieId")

        # Não aplicar filtro de 50 pois temos apenas 15 usuários no mock
        train_data, test_data = prep.dividir_treino_teste(df_merged, test_size=0.2)
        matriz = prep.construir_matriz_usuario_item(train_data)

        # Confirma dimensões da matriz
        self.assertGreater(matriz.shape[0], 0, "Matriz deve ter linhas (usuários)")
        self.assertGreater(matriz.shape[1], 0, "Matriz deve ter colunas (filmes)")

        # 3. Treinar SVD
        rec = RecomendadorSVD(n_components=5, random_state=42)
        rec.treinar(matriz)

        # 4. Recomendar
        usuario = int(matriz.index[0])
        resultado = rec.recomendar(usuario, movies_df, n_recomendacoes=5)

        # Validações finais
        self.assertIsInstance(resultado, pd.DataFrame)
        self.assertFalse(resultado.empty, "Deve haver pelo menos 1 recomendação")
        self.assertIn("movieId", resultado.columns)
        self.assertIn("title", resultado.columns)
        self.assertLessEqual(len(resultado), 5)


# ─────────────────────────────────────────────
# IT-02: SVD treinado → calcular_rmse()
# ─────────────────────────────────────────────

class TestIntegracaoRMSEPipeline(unittest.TestCase):
    """
    Teste de Integração IT-02:
    Preprocessador → RecomendadorSVD → calcular_rmse()
    Verifica que o RMSE calculado sobre o teste é um valor numérico válido.
    """

    def setUp(self):
        self.ratings_path, self.movies_path, self.tmp_dir = gerar_csvs_temporarios(
            n_users=12, n_movies=20, seed=10
        )

    def test_rmse_apos_pipeline_completo(self):
        """
        [IT-02] O RMSE calculado após treino deve ser um float >= 0
        e sensato para dados de filmes (geralmente < 5.0).
        """
        loader = DataLoader(self.ratings_path, self.movies_path)
        ratings_df = loader.carregar_ratings()
        movies_df = loader.carregar_movies()
        movies_df = loader.aplicar_one_hot_genres(movies_df)

        prep = Preprocessador()
        df_merged = pd.merge(ratings_df, movies_df, on="movieId")
        train_data, test_data = prep.dividir_treino_teste(df_merged, test_size=0.2)
        matriz = prep.construir_matriz_usuario_item(train_data)

        rec = RecomendadorSVD(n_components=5, random_state=42)
        rec.treinar(matriz)

        # Filtra o test_data para pares existentes na matriz
        test_valido = test_data[
            test_data["userId"].isin(rec._reconstructed_df.index)
            & test_data["movieId"].isin(rec._reconstructed_df.columns)
        ]

        if test_valido.empty:
            self.skipTest("Nenhum par válido no test set — skip RMSE.")

        rmse = rec.calcular_rmse(test_valido)

        self.assertIsInstance(rmse, float)
        self.assertGreaterEqual(rmse, 0.0)
        self.assertLess(rmse, 5.0, "RMSE deve ser menor que 5 (escala máxima de nota)")


# ─────────────────────────────────────────────
# IT-03: DataLoader → Preprocessador → LR
# ─────────────────────────────────────────────

class TestIntegracaoPipelineLR(unittest.TestCase):
    """
    Teste de Integração IT-03:
    DataLoader → Preprocessador → RecomendadorRegressaoLogistica → avaliar() + recomendar()
    """

    def setUp(self):
        self.ratings_path, self.movies_path, self.tmp_dir = gerar_csvs_temporarios(
            n_users=15, n_movies=20, seed=5
        )

    def test_pipeline_completo_lr(self):
        """
        [IT-03] O pipeline LR deve carregar, pré-processar, treinar,
        avaliar e recomendar sem erros, com métricas válidas.
        """
        # 1. Carregar
        loader = DataLoader(self.ratings_path, self.movies_path)
        ratings_df = loader.carregar_ratings()
        movies_df = loader.carregar_movies()
        movies_df = loader.aplicar_one_hot_genres(movies_df)

        genre_columns = [c for c in movies_df.columns if c not in ["movieId", "title", "year"]]

        # 2. Pré-processar
        prep = Preprocessador()
        df_merged = pd.merge(ratings_df, movies_df, on="movieId")
        df_merged = prep.criar_coluna_liked(df_merged, threshold=4.0)

        features = ["userId", "movieId"] + genre_columns
        X = df_merged[features].copy()
        y = df_merged["liked"]

        X["userId"] = X["userId"].astype(str)
        X["movieId"] = X["movieId"].astype(str)
        X_encoded = pd.get_dummies(X, columns=["userId", "movieId"], drop_first=True)

        # Verifica que há pelo menos 2 classes (liked=0 e liked=1) para stratify
        if y.nunique() < 2:
            self.skipTest("Dados sintéticos sem variação suficiente na coluna 'liked'.")

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. Treinar
        rec = RecomendadorRegressaoLogistica(random_state=42)
        rec.treinar(X_train, y_train, features, genre_columns, df_merged)

        # 4. Avaliar
        metricas = rec.avaliar(X_test, y_test)

        self.assertIn("accuracy", metricas)
        self.assertIn("auc_roc", metricas)
        self.assertGreaterEqual(metricas["accuracy"], 0.0)
        self.assertLessEqual(metricas["accuracy"], 1.0)

        # 5. Recomendar
        usuario = int(df_merged["userId"].iloc[0])
        resultado = rec.recomendar(usuario, movies_df, n_recomendacoes=5)

        self.assertIsInstance(resultado, pd.DataFrame)
        if not resultado.empty:
            self.assertIn("probabilidade_gostar", resultado.columns)


# ─────────────────────────────────────────────
# IT-04: Determinismo das recomendações
# ─────────────────────────────────────────────

class TestIntegracaoDeterminismo(unittest.TestCase):
    """
    Teste de Integração IT-04:
    Verifica que chamadas repetidas ao recomendar() retornam os mesmos resultados.
    Garante reprodutibilidade do modelo.
    """

    def setUp(self):
        self.ratings_path, self.movies_path, self.tmp_dir = gerar_csvs_temporarios(
            n_users=10, n_movies=20, seed=99
        )

    def test_recomendacoes_sao_deterministicas(self):
        """
        [IT-04] recomendar() deve retornar os mesmos filmes em chamadas consecutivas
        para o mesmo usuário — garante reprodutibilidade.
        """
        loader = DataLoader(self.ratings_path, self.movies_path)
        ratings_df = loader.carregar_ratings()
        movies_df = loader.carregar_movies()
        movies_df = loader.aplicar_one_hot_genres(movies_df)

        prep = Preprocessador()
        df_merged = pd.merge(ratings_df, movies_df, on="movieId")
        train_data, _ = prep.dividir_treino_teste(df_merged, test_size=0.2)
        matriz = prep.construir_matriz_usuario_item(train_data)

        rec = RecomendadorSVD(n_components=5, random_state=42)
        rec.treinar(matriz)

        usuarios = list(matriz.index[:3])

        for uid in usuarios:
            with self.subTest(user_id=uid):
                r1 = rec.recomendar(uid, movies_df, n_recomendacoes=5)
                r2 = rec.recomendar(uid, movies_df, n_recomendacoes=5)
                pd.testing.assert_frame_equal(
                    r1.reset_index(drop=True),
                    r2.reset_index(drop=True),
                    check_like=True,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)

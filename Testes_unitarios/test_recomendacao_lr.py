"""
Testes Unitários — Módulo: recomendacao_lr.py
Arquivo: test_recomendacao_lr.py

Testa a classe RecomendadorRegressaoLogistica:
- treinar(): treina sem erros, falha para dados vazios
- avaliar(): retorna accuracy e auc_roc entre 0 e 1
- recomendar(): retorna DataFrame correto, valida erros de usuário e n inválido

Parametrizado: Sim (subTest para múltiplos usuários)
Mockado: Sim (dados sintéticos, sem arquivos externos)
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recomendacao_lr import RecomendadorRegressaoLogistica
from preprocessamento import Preprocessador


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def criar_ambiente_lr():
    """
    Cria e treina um RecomendadorRegressaoLogistica com dados sintéticos.
    Retorna (rec, X_train, X_test, y_train, y_test, movies_df, df_merged, features, genre_columns)
    """
    rng = np.random.default_rng(7)
    n = 300
    user_ids = rng.integers(1, 6, size=n)
    movie_ids = rng.integers(101, 111, size=n)
    ratings = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n).astype(float)

    df = pd.DataFrame({"userId": user_ids, "movieId": movie_ids, "rating": ratings})
    df = df.drop_duplicates(subset=["userId", "movieId"])

    # Filmes mock com gêneros
    movies_df = pd.DataFrame({
        "movieId": list(range(101, 111)),
        "title": [f"Filme {i} (200{i-100})" for i in range(101, 111)],
        "Action": rng.integers(0, 2, size=10),
        "Drama": rng.integers(0, 2, size=10),
        "Comedy": rng.integers(0, 2, size=10),
    })

    genre_columns = ["Action", "Drama", "Comedy"]

    # Merge
    df_merged = pd.merge(df, movies_df, on="movieId")

    prep = Preprocessador()
    df_merged = prep.criar_coluna_liked(df_merged, threshold=4.0)

    features = ["userId", "movieId"] + genre_columns
    X = df_merged[features].copy()
    y = df_merged["liked"]

    X["userId"] = X["userId"].astype(str)
    X["movieId"] = X["movieId"].astype(str)
    X_encoded = pd.get_dummies(X, columns=["userId", "movieId"], drop_first=True)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    rec = RecomendadorRegressaoLogistica(random_state=42)
    rec.treinar(X_train, y_train, features, genre_columns, df_merged)

    return rec, X_train, X_test, y_train, y_test, movies_df, df_merged, features, genre_columns


class TestRecomendadorLRTreinar(unittest.TestCase):
    """Testes para treinar()."""

    def setUp(self):
        (self.rec, self.X_train, self.X_test,
         self.y_train, self.y_test, self.movies_df,
         self.df_merged, self.features, self.genre_columns) = criar_ambiente_lr()

    # ── Teste 1 ──────────────────────────────
    def test_treinar_com_dados_validos(self):
        """[UT-42] treinar() deve completar sem erros com dados válidos."""
        self.assertIsNotNone(self.rec._model)

    # ── Teste 2 ──────────────────────────────
    def test_treinar_dados_vazios_levanta_erro(self):
        """[UT-43] treinar() deve lançar ValueError para dados de treino vazios."""
        rec_novo = RecomendadorRegressaoLogistica()
        with self.assertRaises(ValueError):
            rec_novo.treinar(
                pd.DataFrame(), pd.Series(dtype=int),
                self.features, self.genre_columns, self.df_merged,
            )


class TestRecomendadorLRAvaliar(unittest.TestCase):
    """Testes para avaliar()."""

    def setUp(self):
        (self.rec, self.X_train, self.X_test,
         self.y_train, self.y_test, self.movies_df,
         self.df_merged, self.features, self.genre_columns) = criar_ambiente_lr()

    # ── Teste 3 ──────────────────────────────
    def test_avaliar_retorna_metricas(self):
        """[UT-44] avaliar() deve retornar dicionário com 'accuracy' e 'auc_roc'."""
        metricas = self.rec.avaliar(self.X_test, self.y_test)
        self.assertIn("accuracy", metricas)
        self.assertIn("auc_roc", metricas)

    # ── Teste 4 (parametrizado) ──────────────
    def test_metricas_no_intervalo_valido(self):
        """[UT-45] accuracy e auc_roc devem estar entre 0 e 1."""
        metricas = self.rec.avaliar(self.X_test, self.y_test)
        for chave in ["accuracy", "auc_roc"]:
            with self.subTest(metrica=chave):
                self.assertGreaterEqual(metricas[chave], 0.0)
                self.assertLessEqual(metricas[chave], 1.0)

    # ── Teste 5 ──────────────────────────────
    def test_avaliar_sem_treino_levanta_erro(self):
        """[UT-46] avaliar() sem treinar() deve lançar RuntimeError."""
        rec_novo = RecomendadorRegressaoLogistica()
        with self.assertRaises(RuntimeError):
            rec_novo.avaliar(self.X_test, self.y_test)


class TestRecomendadorLRRecomendar(unittest.TestCase):
    """Testes para recomendar()."""

    def setUp(self):
        (self.rec, self.X_train, self.X_test,
         self.y_train, self.y_test, self.movies_df,
         self.df_merged, self.features, self.genre_columns) = criar_ambiente_lr()
        self.usuario_valido = int(self.df_merged["userId"].iloc[0])

    # ── Teste 6 ──────────────────────────────
    def test_recomendar_retorna_dataframe(self):
        """[UT-47] recomendar() deve retornar um DataFrame."""
        resultado = self.rec.recomendar(self.usuario_valido, self.movies_df, n_recomendacoes=5)
        self.assertIsInstance(resultado, pd.DataFrame)

    # ── Teste 7 ──────────────────────────────
    def test_recomendar_colunas_corretas(self):
        """[UT-48] recomendar() deve retornar colunas [movieId, title, probabilidade_gostar]."""
        resultado = self.rec.recomendar(self.usuario_valido, self.movies_df, n_recomendacoes=3)
        for col in ["movieId", "title", "probabilidade_gostar"]:
            with self.subTest(coluna=col):
                self.assertIn(col, resultado.columns)

    # ── Teste 8 (parametrizado) ──────────────
    def test_recomendacao_para_multiplos_usuarios(self):
        """[UT-49] recomendar() deve funcionar para todos os usuários presentes nos dados."""
        usuarios = self.df_merged["userId"].unique()[:3]
        for uid in usuarios:
            with self.subTest(user_id=int(uid)):
                resultado = self.rec.recomendar(int(uid), self.movies_df, n_recomendacoes=3)
                self.assertIsInstance(resultado, pd.DataFrame)

    # ── Teste 9 ──────────────────────────────
    def test_usuario_inexistente_levanta_erro(self):
        """[UT-50] recomendar() deve lançar ValueError para usuário inexistente."""
        with self.assertRaises(ValueError):
            self.rec.recomendar(user_id=99999, movies_df=self.movies_df)

    # ── Teste 10 ─────────────────────────────
    def test_n_invalido_levanta_erro(self):
        """[UT-51] recomendar() deve lançar ValueError para n_recomendacoes <= 0."""
        for n in [0, -1]:
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    self.rec.recomendar(self.usuario_valido, self.movies_df, n_recomendacoes=n)

    # ── Teste 11 ─────────────────────────────
    def test_sem_treino_levanta_erro(self):
        """[UT-52] recomendar() sem treinar() deve lançar RuntimeError."""
        rec_novo = RecomendadorRegressaoLogistica()
        with self.assertRaises(RuntimeError):
            rec_novo.recomendar(user_id=1, movies_df=self.movies_df)

    # ── Teste 12 ─────────────────────────────
    def test_probabilidade_entre_0_e_1(self):
        """[UT-53] Probabilidades retornadas devem estar entre 0 e 1."""
        resultado = self.rec.recomendar(self.usuario_valido, self.movies_df, n_recomendacoes=5)
        if not resultado.empty:
            self.assertTrue((resultado["probabilidade_gostar"] >= 0).all())
            self.assertTrue((resultado["probabilidade_gostar"] <= 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)

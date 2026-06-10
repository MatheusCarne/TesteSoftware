"""
Testes Unitários — Módulo: preprocessamento.py
Arquivo: test_preprocessamento.py

Testa a classe Preprocessador:
- dividir_treino_teste(): proporção e tamanho corretos
- construir_matriz_usuario_item(): estrutura da matriz pivot
- criar_coluna_liked(): binarização correta por threshold
- criar_mapeamentos(): dicionários de índices consecutivos

Parametrizado: Sim (subTest para múltiplos thresholds e proporções)
Mockado: Sim (DataFrames sintéticos em memória)
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessamento import Preprocessador


# ─────────────────────────────────────────────
# Fixture: DataFrame sintético de avaliações
# ─────────────────────────────────────────────

def criar_df_mock(n_users=5, n_movies=4, seed=42) -> pd.DataFrame:
    """Cria DataFrame sintético de avaliações para testes."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(1, n_users + 1):
        for mid in range(101, 101 + n_movies):
            rows.append({
                "userId": uid,
                "movieId": mid,
                "rating": float(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0])),
                "timestamp": 946684800,
            })
    return pd.DataFrame(rows)


class TestPreprocessadorDividirTreineTeste(unittest.TestCase):
    """Testes para dividir_treino_teste()."""

    def setUp(self):
        self.prep = Preprocessador()
        self.df = criar_df_mock()

    # ── Teste 1 ──────────────────────────────
    def test_tamanho_correto_do_split(self):
        """[UT-15] dividir_treino_teste() deve produzir tamanhos proporcionais corretos."""
        train, test = self.prep.dividir_treino_teste(self.df, test_size=0.2)
        total = len(self.df)
        self.assertEqual(len(train) + len(test), total)
        # Tolerância de ±1 registro
        self.assertAlmostEqual(len(test) / total, 0.2, delta=0.05)

    # ── Teste 2 (parametrizado) ──────────────
    def test_diferentes_proporcoes(self):
        """[UT-16] dividir_treino_teste() deve funcionar com diferentes test_size."""
        for ts in [0.1, 0.2, 0.3, 0.4]:
            with self.subTest(test_size=ts):
                train, test = self.prep.dividir_treino_teste(self.df, test_size=ts)
                self.assertGreater(len(train), 0)
                self.assertGreater(len(test), 0)

    # ── Teste 3 ──────────────────────────────
    def test_test_size_invalido_levanta_erro(self):
        """[UT-17] dividir_treino_teste() deve lançar ValueError para test_size fora de (0,1)."""
        for ts in [0, 1, -0.1, 1.5]:
            with self.subTest(test_size=ts):
                with self.assertRaises(ValueError):
                    self.prep.dividir_treino_teste(self.df, test_size=ts)

    # ── Teste 4 ──────────────────────────────
    def test_dataframe_vazio_levanta_erro(self):
        """[UT-18] dividir_treino_teste() deve lançar ValueError para DataFrame vazio."""
        with self.assertRaises(ValueError):
            self.prep.dividir_treino_teste(pd.DataFrame())


class TestPreprocessadorMatrizUsuarioItem(unittest.TestCase):
    """Testes para construir_matriz_usuario_item()."""

    def setUp(self):
        self.prep = Preprocessador()
        self.df = criar_df_mock(n_users=3, n_movies=4)

    # ── Teste 5 ──────────────────────────────
    def test_dimensoes_da_matriz(self):
        """[UT-19] construir_matriz_usuario_item() deve ter shape (n_users, n_movies)."""
        matriz = self.prep.construir_matriz_usuario_item(self.df)
        n_users = self.df["userId"].nunique()
        n_movies = self.df["movieId"].nunique()
        self.assertEqual(matriz.shape, (n_users, n_movies))

    # ── Teste 6 ──────────────────────────────
    def test_sem_valores_nulos_na_matriz(self):
        """[UT-20] A matriz deve ser preenchida com 0 onde não há avaliação (sem NaN)."""
        matriz = self.prep.construir_matriz_usuario_item(self.df)
        self.assertFalse(matriz.isnull().any().any())

    # ── Teste 7 ──────────────────────────────
    def test_coluna_ausente_levanta_erro(self):
        """[UT-21] construir_matriz_usuario_item() deve lançar ValueError se faltarem colunas."""
        df_sem_rating = self.df.drop(columns=["rating"])
        with self.assertRaises(ValueError):
            self.prep.construir_matriz_usuario_item(df_sem_rating)


class TestPreprocessadorColunaLiked(unittest.TestCase):
    """Testes para criar_coluna_liked()."""

    def setUp(self):
        self.prep = Preprocessador()
        self.df = pd.DataFrame({
            "userId": [1, 1, 2, 2],
            "movieId": [10, 20, 10, 30],
            "rating": [5.0, 3.0, 4.0, 2.5],
        })

    # ── Teste 8 ──────────────────────────────
    def test_binarizacao_correta_threshold_4(self):
        """[UT-22] criar_coluna_liked() com threshold=4.0 deve classificar corretamente."""
        resultado = self.prep.criar_coluna_liked(self.df, threshold=4.0)
        esperado = [1, 0, 1, 0]
        self.assertEqual(resultado["liked"].tolist(), esperado)

    # ── Teste 9 (parametrizado) ──────────────
    def test_diferentes_thresholds(self):
        """[UT-23] criar_coluna_liked() deve funcionar com diferentes thresholds."""
        casos = [
            (3.0, [1, 1, 1, 0]),
            (4.0, [1, 0, 1, 0]),
            (5.0, [1, 0, 0, 0]),
        ]
        for threshold, esperado in casos:
            with self.subTest(threshold=threshold):
                resultado = self.prep.criar_coluna_liked(self.df, threshold=threshold)
                self.assertEqual(resultado["liked"].tolist(), esperado)

    # ── Teste 10 ─────────────────────────────
    def test_threshold_invalido_levanta_erro(self):
        """[UT-24] criar_coluna_liked() deve lançar ValueError para threshold inválido."""
        for t in [0.0, 6.0, -1.0]:
            with self.subTest(threshold=t):
                with self.assertRaises(ValueError):
                    self.prep.criar_coluna_liked(self.df, threshold=t)

    # ── Teste 11 ─────────────────────────────
    def test_sem_coluna_rating_levanta_erro(self):
        """[UT-25] criar_coluna_liked() deve lançar ValueError se 'rating' não existir."""
        df_sem = self.df.drop(columns=["rating"])
        with self.assertRaises(ValueError):
            self.prep.criar_coluna_liked(df_sem)

    # ── Teste 12 ─────────────────────────────
    def test_nao_modifica_dataframe_original(self):
        """[UT-26] criar_coluna_liked() não deve modificar o DataFrame original."""
        df_copia = self.df.copy()
        self.prep.criar_coluna_liked(self.df, threshold=4.0)
        pd.testing.assert_frame_equal(self.df, df_copia)


class TestPreprocessadorMapeamentos(unittest.TestCase):
    """Testes para criar_mapeamentos()."""

    def setUp(self):
        self.prep = Preprocessador()
        self.df = criar_df_mock(n_users=3, n_movies=3)

    # ── Teste 13 ─────────────────────────────
    def test_mapeamentos_corretos(self):
        """[UT-27] criar_mapeamentos() deve mapear todos os usuários e filmes únicos."""
        user_map, movie_map = self.prep.criar_mapeamentos(self.df)
        self.assertEqual(len(user_map), self.df["userId"].nunique())
        self.assertEqual(len(movie_map), self.df["movieId"].nunique())

    # ── Teste 14 ─────────────────────────────
    def test_indices_sao_consecutivos(self):
        """[UT-28] Os índices dos mapeamentos devem ser consecutivos a partir de 0."""
        user_map, movie_map = self.prep.criar_mapeamentos(self.df)
        user_indices = sorted(user_map.values())
        movie_indices = sorted(movie_map.values())
        self.assertEqual(user_indices, list(range(len(user_map))))
        self.assertEqual(movie_indices, list(range(len(movie_map))))

    # ── Teste 15 ─────────────────────────────
    def test_coluna_ausente_levanta_erro(self):
        """[UT-29] criar_mapeamentos() deve lançar ValueError se colunas estiverem ausentes."""
        df_sem = self.df.drop(columns=["userId"])
        with self.assertRaises(ValueError):
            self.prep.criar_mapeamentos(df_sem)


if __name__ == "__main__":
    unittest.main(verbosity=2)

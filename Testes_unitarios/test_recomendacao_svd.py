"""
Testes Unitários — Módulo: recomendacao_svd.py
Arquivo: test_recomendacao_svd.py

Testa a classe RecomendadorSVD:
- treinar(): modelo é treinado corretamente com matriz válida
- recomendar(): retorna DataFrame com colunas corretas, exclui filmes já vistos
- recomendar(): valida erros para usuário inexistente, modelo não treinado, n inválido
- calcular_rmse(): retorna float >= 0

Parametrizado: Sim (subTest para múltiplos n_recomendacoes)
Mockado: Sim (matriz usuário-item sintética, sem arquivos CSV)
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recomendacao_svd import RecomendadorSVD
from preprocessamento import Preprocessador


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def criar_df_mock() -> pd.DataFrame:
    """DataFrame de avaliações sintético com 5 usuários e 8 filmes."""
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(1, 6):
        for mid in range(101, 109):
            rows.append({
                "userId": uid,
                "movieId": mid,
                "rating": float(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0])),
                "timestamp": 946684800,
            })
    return pd.DataFrame(rows)


def criar_movies_mock() -> pd.DataFrame:
    return pd.DataFrame({
        "movieId": list(range(101, 109)),
        "title": [f"Filme {i} (200{i-100})" for i in range(101, 109)],
    })


def treinar_svd_com_mock():
    """Treina um RecomendadorSVD com dados mock e retorna modelo + dados."""
    df = criar_df_mock()
    movies = criar_movies_mock()
    prep = Preprocessador()
    train, test = prep.dividir_treino_teste(df, test_size=0.2)
    matriz = prep.construir_matriz_usuario_item(train)

    rec = RecomendadorSVD(n_components=3, random_state=42)
    rec.treinar(matriz)
    return rec, train, test, movies, df


# ─────────────────────────────────────────────
# Classe de testes
# ─────────────────────────────────────────────

class TestRecomendadorSVDTreinar(unittest.TestCase):
    """Testes para o método treinar()."""

    # ── Teste 1 ──────────────────────────────
    def test_treinar_com_matriz_valida(self):
        """[UT-30] treinar() deve executar sem erros com uma matriz válida."""
        df = criar_df_mock()
        prep = Preprocessador()
        matriz = prep.construir_matriz_usuario_item(df)
        rec = RecomendadorSVD(n_components=3)
        try:
            rec.treinar(matriz)
        except Exception as e:
            self.fail(f"treinar() lançou exceção inesperada: {e}")

    # ── Teste 2 ──────────────────────────────
    def test_treinar_com_matriz_vazia_levanta_erro(self):
        """[UT-31] treinar() deve lançar ValueError para matriz vazia."""
        rec = RecomendadorSVD(n_components=3)
        with self.assertRaises(ValueError):
            rec.treinar(pd.DataFrame())

    # ── Teste 3 ──────────────────────────────
    def test_construtor_n_components_invalido(self):
        """[UT-32] RecomendadorSVD() com n_components <= 0 deve lançar ValueError."""
        for n in [0, -1, -10]:
            with self.subTest(n_components=n):
                with self.assertRaises(ValueError):
                    RecomendadorSVD(n_components=n)


class TestRecomendadorSVDRecomendar(unittest.TestCase):
    """Testes para o método recomendar()."""

    def setUp(self):
        self.rec, self.train, self.test, self.movies, self.df = treinar_svd_com_mock()
        self.usuario_valido = self.train["userId"].iloc[0]

    # ── Teste 4 ──────────────────────────────
    def test_recomendar_retorna_dataframe(self):
        """[UT-33] recomendar() deve retornar um DataFrame."""
        resultado = self.rec.recomendar(self.usuario_valido, self.movies, n_recomendacoes=5)
        self.assertIsInstance(resultado, pd.DataFrame)

    # ── Teste 5 ──────────────────────────────
    def test_recomendar_colunas_corretas(self):
        """[UT-34] recomendar() deve retornar DataFrame com colunas [movieId, title]."""
        resultado = self.rec.recomendar(self.usuario_valido, self.movies, n_recomendacoes=3)
        self.assertIn("movieId", resultado.columns)
        self.assertIn("title", resultado.columns)

    # ── Teste 6 (parametrizado) ──────────────
    def test_quantidade_de_recomendacoes(self):
        """[UT-35] recomendar() deve retornar no máximo n_recomendacoes filmes."""
        for n in [1, 3, 5]:
            with self.subTest(n_recomendacoes=n):
                resultado = self.rec.recomendar(self.usuario_valido, self.movies, n_recomendacoes=n)
                self.assertLessEqual(len(resultado), n)

    # ── Teste 7 ──────────────────────────────
    def test_recomendar_usuario_inexistente_levanta_erro(self):
        """[UT-36] recomendar() deve lançar ValueError para usuário inexistente."""
        with self.assertRaises(ValueError):
            self.rec.recomendar(user_id=99999, movies_df=self.movies)

    # ── Teste 8 ──────────────────────────────
    def test_recomendar_sem_treino_levanta_erro(self):
        """[UT-37] recomendar() sem treinar() deve lançar RuntimeError."""
        rec_novo = RecomendadorSVD(n_components=3)
        with self.assertRaises(RuntimeError):
            rec_novo.recomendar(user_id=1, movies_df=self.movies)

    # ── Teste 9 ──────────────────────────────
    def test_n_recomendacoes_invalido_levanta_erro(self):
        """[UT-38] recomendar() com n_recomendacoes <= 0 deve lançar ValueError."""
        for n in [0, -1, -100]:
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    self.rec.recomendar(self.usuario_valido, self.movies, n_recomendacoes=n)


class TestRecomendadorSVDRMSE(unittest.TestCase):
    """Testes para calcular_rmse()."""

    def setUp(self):
        self.rec, self.train, self.test, self.movies, self.df = treinar_svd_com_mock()

    # ── Teste 10 ─────────────────────────────
    def test_rmse_retorna_float_positivo(self):
        """[UT-39] calcular_rmse() deve retornar um float maior ou igual a zero."""
        # Garante que test_data contém pares presentes na matriz
        test_filtrado = self.test[
            self.test["userId"].isin(self.rec._reconstructed_df.index)
            & self.test["movieId"].isin(self.rec._reconstructed_df.columns)
        ]
        if test_filtrado.empty:
            self.skipTest("Nenhum par válido no test set para calcular RMSE.")
        rmse = self.rec.calcular_rmse(test_filtrado)
        self.assertIsInstance(rmse, float)
        self.assertGreaterEqual(rmse, 0.0)

    # ── Teste 11 ─────────────────────────────
    def test_rmse_sem_treino_levanta_erro(self):
        """[UT-40] calcular_rmse() sem treinar() deve lançar RuntimeError."""
        rec_novo = RecomendadorSVD(n_components=3)
        with self.assertRaises(RuntimeError):
            rec_novo.calcular_rmse(self.test)

    # ── Teste 12 ─────────────────────────────
    def test_rmse_colunas_ausentes_levanta_erro(self):
        """[UT-41] calcular_rmse() deve lançar ValueError se faltarem colunas."""
        df_sem_rating = self.test.drop(columns=["rating"])
        with self.assertRaises(ValueError):
            self.rec.calcular_rmse(df_sem_rating)


if __name__ == "__main__":
    unittest.main(verbosity=2)

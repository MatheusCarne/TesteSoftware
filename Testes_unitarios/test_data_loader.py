"""
Testes Unitários — Módulo: data_loader.py
Arquivo: test_data_loader.py

Testa a classe DataLoader:
- carregar_ratings(): leitura, validação de colunas, remoção de nulos
- carregar_movies(): leitura, remoção de duplicatas, extração de ano
- aplicar_one_hot_genres(): encoding correto de gêneros
- filtrar_usuarios_e_filmes(): filtro por mínimo de avaliações

Parametrizado: Sim (subTest para múltiplos cenários)
Mockado: Sim (usa dados sintéticos em memória, sem depender de arquivos reais)
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import sys

# Adiciona o diretório pai ao path para importar os módulos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sistema_recomendacao.data_loader import DataLoader

# ─────────────────────────────────────────────
# Dados sintéticos (mock) reutilizados nos testes
# ─────────────────────────────────────────────

def criar_ratings_csv(path: str, conteudo: str = None):
    """Cria um arquivo ratings.csv temporário com dados sintéticos."""
    if conteudo is None:
        conteudo = (
            "userId,movieId,rating,timestamp\n"
            "1,10,4.0,946684800\n"
            "1,20,3.5,946684801\n"
            "2,10,5.0,946684802\n"
            "2,30,2.0,946684803\n"
            "3,20,1.0,946684804\n"
        )
    with open(path, "w") as f:
        f.write(conteudo)


def criar_movies_csv(path: str, conteudo: str = None):
    """Cria um arquivo movies.csv temporário com dados sintéticos."""
    if conteudo is None:
        conteudo = (
            "movieId,title,genres\n"
            "10,Inception (2010),Action|Sci-Fi\n"
            "20,Titanic (1997),Drama|Romance\n"
            "30,The Matrix (1999),Action|Sci-Fi\n"
        )
    with open(path, "w") as f:
        f.write(conteudo)


# ─────────────────────────────────────────────
# Classe de testes
# ─────────────────────────────────────────────

class TestDataLoaderCarregarRatings(unittest.TestCase):
    """Testes para o método carregar_ratings()."""

    def setUp(self):
        """Cria arquivos temporários antes de cada teste."""
        self.tmp_dir = tempfile.mkdtemp()
        self.ratings_path = os.path.join(self.tmp_dir, "ratings.csv")
        self.movies_path = os.path.join(self.tmp_dir, "movies.csv")
        criar_ratings_csv(self.ratings_path)
        criar_movies_csv(self.movies_path)
        self.loader = DataLoader(self.ratings_path, self.movies_path)

    # ── Teste 1 ──────────────────────────────
    def test_carregar_ratings_retorna_dataframe(self):
        """[UT-01] carregar_ratings() deve retornar um DataFrame não vazio."""
        df = self.loader.carregar_ratings()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    # ── Teste 2 ──────────────────────────────
    def test_carregar_ratings_colunas_corretas(self):
        """[UT-02] carregar_ratings() deve conter colunas obrigatórias."""
        df = self.loader.carregar_ratings()
        for col in ["userId", "movieId", "rating", "timestamp"]:
            with self.subTest(coluna=col):
                self.assertIn(col, df.columns)

    # ── Teste 3 ──────────────────────────────
    def test_carregar_ratings_remove_nulos(self):
        """[UT-03] carregar_ratings() deve remover linhas com valores nulos."""
        conteudo_com_nulo = (
            "userId,movieId,rating,timestamp\n"
            "1,10,4.0,946684800\n"
            ",20,3.5,946684801\n"          # userId nulo
            "2,,2.0,946684803\n"            # movieId nulo
        )
        path = os.path.join(self.tmp_dir, "ratings_nulo.csv")
        with open(path, "w") as f:
            f.write(conteudo_com_nulo)
        loader = DataLoader(path, self.movies_path)
        df = loader.carregar_ratings()
        self.assertFalse(df.isnull().any().any())

    # ── Teste 4 ──────────────────────────────
    def test_carregar_ratings_coluna_ausente_levanta_erro(self):
        """[UT-04] carregar_ratings() deve lançar ValueError se coluna obrigatória estiver ausente."""
        conteudo_sem_rating = (
            "userId,movieId,timestamp\n"
            "1,10,946684800\n"
        )
        path = os.path.join(self.tmp_dir, "ratings_sem_rating.csv")
        with open(path, "w") as f:
            f.write(conteudo_sem_rating)
        loader = DataLoader(path, self.movies_path)
        with self.assertRaises(ValueError):
            loader.carregar_ratings()

    # ── Teste 5 (parametrizado via subTest) ──
    def test_tipos_das_colunas(self):
        """[UT-05] Tipos de dados de userId, movieId e rating devem ser corretos."""
        df = self.loader.carregar_ratings()
        tipos_esperados = {
            "userId": np.integer,
            "movieId": np.integer,
            "rating": np.floating,
        }
        for col, tipo in tipos_esperados.items():
            with self.subTest(coluna=col):
                self.assertTrue(
                    np.issubdtype(df[col].dtype, tipo),
                    msg=f"Coluna '{col}' deveria ser {tipo}, mas é {df[col].dtype}",
                )


class TestDataLoaderCarregarMovies(unittest.TestCase):
    """Testes para o método carregar_movies()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.ratings_path = os.path.join(self.tmp_dir, "ratings.csv")
        self.movies_path = os.path.join(self.tmp_dir, "movies.csv")
        criar_ratings_csv(self.ratings_path)
        criar_movies_csv(self.movies_path)
        self.loader = DataLoader(self.ratings_path, self.movies_path)

    # ── Teste 6 ──────────────────────────────
    def test_carregar_movies_extrai_ano(self):
        """[UT-06] carregar_movies() deve extrair o ano do título corretamente."""
        df = self.loader.carregar_movies()
        self.assertIn("year", df.columns)
        # Inception (2010) → year = 2010
        row = df[df["movieId"] == 10]
        self.assertFalse(row.empty)
        self.assertEqual(int(row["year"].values[0]), 2010)

    # ── Teste 7 ──────────────────────────────
    def test_carregar_movies_remove_duplicatas_titulo(self):
        """[UT-07] carregar_movies() deve remover filmes com títulos duplicados."""
        conteudo = (
            "movieId,title,genres\n"
            "10,Inception (2010),Action\n"
            "11,Inception (2010),Sci-Fi\n"   # duplicata de título
            "20,Titanic (1997),Drama\n"
        )
        path = os.path.join(self.tmp_dir, "movies_dup.csv")
        with open(path, "w") as f:
            f.write(conteudo)
        loader = DataLoader(self.ratings_path, path)
        df = loader.carregar_movies()
        # Inception deve ter sido removido (ambas as linhas)
        titulos = df["title"].tolist()
        self.assertNotIn("Inception (2010)", titulos)

    # ── Teste 8 ──────────────────────────────
    def test_carregar_movies_coluna_ausente_levanta_erro(self):
        """[UT-08] carregar_movies() deve lançar ValueError se 'genres' estiver ausente."""
        conteudo_sem_genres = (
            "movieId,title\n"
            "10,Inception (2010)\n"
        )
        path = os.path.join(self.tmp_dir, "movies_sem_genres.csv")
        with open(path, "w") as f:
            f.write(conteudo_sem_genres)
        loader = DataLoader(self.ratings_path, path)
        with self.assertRaises(ValueError):
            loader.carregar_movies()


class TestDataLoaderOneHotGenres(unittest.TestCase):
    """Testes para aplicar_one_hot_genres()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        ratings_path = os.path.join(self.tmp_dir, "r.csv")
        movies_path = os.path.join(self.tmp_dir, "m.csv")
        criar_ratings_csv(ratings_path)
        criar_movies_csv(movies_path)
        self.loader = DataLoader(ratings_path, movies_path)

    # ── Teste 9 ──────────────────────────────
    def test_one_hot_cria_colunas_de_genero(self):
        """[UT-09] aplicar_one_hot_genres() deve criar colunas binárias para cada gênero."""
        movies_df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A (2000)", "B (2001)"],
            "genres": ["Action|Comedy", "Drama"],
        })
        resultado = self.loader.aplicar_one_hot_genres(movies_df)
        self.assertIn("Action", resultado.columns)
        self.assertIn("Comedy", resultado.columns)
        self.assertIn("Drama", resultado.columns)
        self.assertNotIn("genres", resultado.columns)

    # ── Teste 10 ─────────────────────────────
    def test_one_hot_sem_coluna_genres_levanta_erro(self):
        """[UT-10] aplicar_one_hot_genres() deve lançar ValueError se 'genres' não existir."""
        df_sem_genres = pd.DataFrame({"movieId": [1], "title": ["Filme A"]})
        with self.assertRaises(ValueError):
            self.loader.aplicar_one_hot_genres(df_sem_genres)


class TestDataLoaderFiltrar(unittest.TestCase):
    """Testes para filtrar_usuarios_e_filmes()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        r = os.path.join(self.tmp_dir, "r.csv")
        m = os.path.join(self.tmp_dir, "m.csv")
        criar_ratings_csv(r)
        criar_movies_csv(m)
        self.loader = DataLoader(r, m)

        # DataFrame sintético com usuários e filmes de frequências variadas
        self.df = pd.DataFrame({
            "userId": [1]*60 + [2]*30 + [3]*10,
            "movieId": [10]*60 + [20]*30 + [30]*10,
            "rating": [4.0] * 100,
        })

    # ── Teste 11 ─────────────────────────────
    def test_filtrar_remove_usuarios_abaixo_do_minimo(self):
        """[UT-11] filtrar_usuarios_e_filmes() deve remover usuários com poucas avaliações."""
        resultado = self.loader.filtrar_usuarios_e_filmes(self.df, min_avaliacoes_usuario=50)
        # Usuários 2 (30) e 3 (10) devem ser removidos
        self.assertNotIn(2, resultado["userId"].values)
        self.assertNotIn(3, resultado["userId"].values)
        self.assertIn(1, resultado["userId"].values)

    # ── Teste 12 (parametrizado via subTest) ─
    def test_filtrar_minimos_negativos_levanta_erro(self):
        """[UT-12] filtrar_usuarios_e_filmes() deve lançar ValueError para mínimos negativos."""
        casos = [
            {"min_avaliacoes_usuario": -1, "min_avaliacoes_filme": 50},
            {"min_avaliacoes_usuario": 50, "min_avaliacoes_filme": -5},
        ]
        for caso in casos:
            with self.subTest(caso=caso):
                with self.assertRaises(ValueError):
                    self.loader.filtrar_usuarios_e_filmes(self.df, **caso)

    # ── Teste 13 ─────────────────────────────
    def test_filtrar_dataframe_vazio_levanta_erro(self):
        """[UT-13] filtrar_usuarios_e_filmes() deve lançar ValueError para DataFrame vazio."""
        df_vazio = pd.DataFrame(columns=["userId", "movieId", "rating"])
        with self.assertRaises(ValueError):
            self.loader.filtrar_usuarios_e_filmes(df_vazio)


class TestDataLoaderInit(unittest.TestCase):
    """Testes para o construtor de DataLoader."""

    # ── Teste 14 ─────────────────────────────
    def test_construtor_caminhos_vazios_levanta_erro(self):
        """[UT-14] DataLoader() com caminhos vazios deve lançar ValueError."""
        with self.assertRaises(ValueError):
            DataLoader("", "movies.csv")
        with self.assertRaises(ValueError):
            DataLoader("ratings.csv", "")


if __name__ == "__main__":
    unittest.main(verbosity=2)

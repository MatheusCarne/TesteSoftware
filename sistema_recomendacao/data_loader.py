"""
Módulo responsável pelo carregamento e limpeza dos dados do sistema de recomendação.
Dataset: MovieLens (ratings.csv, movies.csv)
"""

import pandas as pd


class DataLoader:
    """
    Carrega e pré-processa os arquivos CSV do MovieLens.

    Attributes:
        ratings_path (str): Caminho para o arquivo ratings.csv
        movies_path  (str): Caminho para o arquivo movies.csv
    """

    def __init__(self, ratings_path: str, movies_path: str):
        if not ratings_path or not movies_path:
            raise ValueError("Os caminhos dos arquivos não podem ser vazios.")
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self._ratings_df = None
        self._movies_df = None

    def carregar_ratings(self) -> pd.DataFrame:
        """
        Lê o CSV de avaliações, remove nulos e garante tipos corretos.

        Returns:
            pd.DataFrame: DataFrame limpo com colunas [userId, movieId, rating, timestamp]

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        df = pd.read_csv(self.ratings_path)
        colunas_obrigatorias = {"userId", "movieId", "rating", "timestamp"}
        if not colunas_obrigatorias.issubset(df.columns):
            raise ValueError(
                f"Colunas obrigatórias ausentes em ratings: "
                f"{colunas_obrigatorias - set(df.columns)}"
            )
        df.dropna(inplace=True)
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        df["rating"] = df["rating"].astype(float)
        self._ratings_df = df
        return df

    def carregar_movies(self) -> pd.DataFrame:
        """
        Lê o CSV de filmes, remove nulos, duplicatas de título e extrai o ano.

        Returns:
            pd.DataFrame: DataFrame limpo com colunas [movieId, title, genres, year, ...]

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        df = pd.read_csv(self.movies_path)
        colunas_obrigatorias = {"movieId", "title", "genres"}
        if not colunas_obrigatorias.issubset(df.columns):
            raise ValueError(
                f"Colunas obrigatórias ausentes em movies: "
                f"{colunas_obrigatorias - set(df.columns)}"
            )
        df.dropna(inplace=True)

        # Remove títulos duplicados
        duplicatas = df[df.duplicated("title", keep=False)]["title"]
        df = df[~df["title"].isin(duplicatas)]

        # Extrai ano do título
        df = df.copy()
        df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype(float)

        self._movies_df = df
        return df

    def aplicar_one_hot_genres(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica one-hot encoding na coluna 'genres' e a remove do DataFrame.

        Args:
            movies_df (pd.DataFrame): DataFrame de filmes com coluna 'genres'.

        Returns:
            pd.DataFrame: DataFrame com colunas binárias por gênero.

        Raises:
            ValueError: Se a coluna 'genres' não existir.
        """
        if "genres" not in movies_df.columns:
            raise ValueError("Coluna 'genres' não encontrada no DataFrame.")
        genres_dummies = movies_df["genres"].str.get_dummies(sep="|")
        df = pd.concat([movies_df, genres_dummies], axis=1)
        df = df.drop("genres", axis=1)
        return df

    def filtrar_usuarios_e_filmes(
        self,
        df_merged: pd.DataFrame,
        min_avaliacoes_usuario: int = 50,
        min_avaliacoes_filme: int = 50,
    ) -> pd.DataFrame:
        """
        Mantém apenas usuários e filmes com número mínimo de avaliações.

        Args:
            df_merged (pd.DataFrame): DataFrame combinado ratings + movies.
            min_avaliacoes_usuario (int): Mínimo de avaliações por usuário.
            min_avaliacoes_filme (int): Mínimo de avaliações por filme.

        Returns:
            pd.DataFrame: DataFrame filtrado.

        Raises:
            ValueError: Se os mínimos forem negativos ou DataFrame vazio.
        """
        if min_avaliacoes_usuario < 0 or min_avaliacoes_filme < 0:
            raise ValueError("Mínimos de avaliações não podem ser negativos.")
        if df_merged.empty:
            raise ValueError("O DataFrame fornecido está vazio.")

        user_counts = df_merged["userId"].value_counts()
        movie_counts = df_merged["movieId"].value_counts()

        df_filtrado = df_merged[
            df_merged["userId"].isin(user_counts[user_counts >= min_avaliacoes_usuario].index)
        ]
        df_filtrado = df_filtrado[
            df_filtrado["movieId"].isin(movie_counts[movie_counts >= min_avaliacoes_filme].index)
        ]
        return df_filtrado

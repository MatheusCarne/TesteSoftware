"""
Módulo de pré-processamento: criação da matriz usuário-item e divisão treino/teste.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


class Preprocessador:
    """
    Constrói a matriz usuário-item e realiza a divisão treino/teste.
    """

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.user_item_matrix = None

    def dividir_treino_teste(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple:
        """
        Divide o DataFrame em conjuntos de treino e teste.

        Args:
            df (pd.DataFrame): DataFrame com avaliações.
            test_size (float): Proporção dos dados de teste (0 < test_size < 1).
            random_state (int): Semente aleatória para reprodutibilidade.

        Returns:
            tuple: (train_data, test_data) como DataFrames.

        Raises:
            ValueError: Se test_size estiver fora do intervalo (0, 1).
            ValueError: Se o DataFrame estiver vazio.
        """
        if not (0 < test_size < 1):
            raise ValueError("test_size deve estar entre 0 e 1 (exclusivo).")
        if df.empty:
            raise ValueError("O DataFrame fornecido está vazio.")

        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
        self.train_data = train
        self.test_data = test
        return train, test

    def construir_matriz_usuario_item(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria uma matriz usuário x filme com as notas como valores.
        Células sem avaliação são preenchidas com 0.

        Args:
            df (pd.DataFrame): DataFrame com colunas [userId, movieId, rating].

        Returns:
            pd.DataFrame: Matriz pivot usuário x filme.

        Raises:
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        colunas = {"userId", "movieId", "rating"}
        if not colunas.issubset(df.columns):
            raise ValueError(f"Colunas obrigatórias ausentes: {colunas - set(df.columns)}")

        matriz = df.pivot_table(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)
        self.user_item_matrix = matriz
        return matriz

    def criar_coluna_liked(self, df: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
        """
        Cria coluna binária 'liked': 1 se rating >= threshold, 0 caso contrário.

        Args:
            df (pd.DataFrame): DataFrame com coluna 'rating'.
            threshold (float): Nota mínima para classificar como 'gostou'.

        Returns:
            pd.DataFrame: DataFrame com a coluna 'liked' adicionada.

        Raises:
            ValueError: Se 'rating' não existir no DataFrame.
            ValueError: Se threshold estiver fora do intervalo válido [0.5, 5.0].
        """
        if "rating" not in df.columns:
            raise ValueError("Coluna 'rating' não encontrada no DataFrame.")
        if not (0.5 <= threshold <= 5.0):
            raise ValueError("Threshold deve estar entre 0.5 e 5.0.")

        df = df.copy()
        df["liked"] = (df["rating"] >= threshold).astype(int)
        return df

    def criar_mapeamentos(self, df: pd.DataFrame) -> tuple:
        """
        Cria dicionários de mapeamento userId/movieId para índices consecutivos,
        necessários para embeddings em redes neurais.

        Args:
            df (pd.DataFrame): DataFrame com colunas userId e movieId.

        Returns:
            tuple: (user_map, movie_map) — dicts {id_original: índice_consecutivo}

        Raises:
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        if "userId" not in df.columns or "movieId" not in df.columns:
            raise ValueError("DataFrame deve conter colunas 'userId' e 'movieId'.")

        user_map = {u: i for i, u in enumerate(df["userId"].unique())}
        movie_map = {m: i for i, m in enumerate(df["movieId"].unique())}
        return user_map, movie_map

"""
Módulo de recomendação baseado em SVD (Decomposição em Valores Singulares).
Utiliza filtragem colaborativa para prever notas de filmes não avaliados.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


class RecomendadorSVD:
    """
    Sistema de recomendação baseado em SVD truncado.

    O modelo decompõe a matriz usuário-item em fatores latentes e reconstrói
    as notas previstas para itens não avaliados.

    Attributes:
        n_components (int): Número de fatores latentes.
        random_state (int): Semente para reprodutibilidade.
    """

    def __init__(self, n_components: int = 20, random_state: int = 42):
        if n_components <= 0:
            raise ValueError("n_components deve ser maior que zero.")
        self.n_components = n_components
        self.random_state = random_state
        self._svd = None
        self._reconstructed_df = None
        self._train_matrix = None

    def treinar(self, user_item_matrix: pd.DataFrame) -> None:
        """
        Treina o modelo SVD sobre a matriz usuário-item.

        Args:
            user_item_matrix (pd.DataFrame): Matriz pivot com userId nos índices
                e movieId nas colunas, preenchida com 0 onde não há avaliação.

        Raises:
            ValueError: Se a matriz estiver vazia.
        """
        if user_item_matrix.empty:
            raise ValueError("A matriz usuário-item está vazia.")

        self._train_matrix = user_item_matrix
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        latent_matrix = svd.fit_transform(user_item_matrix)
        self._svd = svd

        reconstructed = np.dot(latent_matrix, svd.components_)
        self._reconstructed_df = pd.DataFrame(
            reconstructed,
            index=user_item_matrix.index,
            columns=user_item_matrix.columns,
        )

    def recomendar(
        self,
        user_id: int,
        movies_df: pd.DataFrame,
        n_recomendacoes: int = 10,
    ) -> pd.DataFrame:
        """
        Recomenda os N filmes com maior nota prevista para o usuário,
        excluindo filmes que ele já avaliou.

        Args:
            user_id (int): ID do usuário alvo.
            movies_df (pd.DataFrame): DataFrame com colunas [movieId, title].
            n_recomendacoes (int): Quantidade de filmes a recomendar.

        Returns:
            pd.DataFrame: DataFrame com colunas [movieId, title] dos filmes recomendados.

        Raises:
            RuntimeError: Se o modelo não foi treinado ainda.
            ValueError: Se o user_id não existir na matriz de treino.
            ValueError: Se n_recomendacoes for menor ou igual a zero.
        """
        if self._reconstructed_df is None or self._train_matrix is None:
            raise RuntimeError("Modelo não treinado. Chame treinar() antes de recomendar().")
        if n_recomendacoes <= 0:
            raise ValueError("n_recomendacoes deve ser maior que zero.")
        if user_id not in self._train_matrix.index:
            raise ValueError(f"Usuário {user_id} não encontrado na matriz de treino.")

        # Filmes ainda não avaliados pelo usuário (nota original = 0)
        notas_originais = self._train_matrix.loc[user_id]
        notas_previstas = self._reconstructed_df.loc[user_id]
        nao_avaliados = notas_originais[notas_originais == 0].index

        top = notas_previstas[nao_avaliados].sort_values(ascending=False).head(n_recomendacoes)
        recomendados = movies_df[movies_df["movieId"].isin(top.index)][["movieId", "title"]]
        return recomendados.reset_index(drop=True)

    def calcular_rmse(self, test_data: pd.DataFrame) -> float:
        """
        Calcula o RMSE entre notas reais e previstas sobre o conjunto de teste.

        Args:
            test_data (pd.DataFrame): DataFrame com colunas [userId, movieId, rating].

        Returns:
            float: Valor do RMSE.

        Raises:
            RuntimeError: Se o modelo não foi treinado.
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        if self._reconstructed_df is None:
            raise RuntimeError("Modelo não treinado. Chame treinar() antes.")
        colunas = {"userId", "movieId", "rating"}
        if not colunas.issubset(test_data.columns):
            raise ValueError(f"Colunas obrigatórias ausentes: {colunas - set(test_data.columns)}")

        # Filtra apenas pares (user, movie) que existem na matriz reconstruída
        test_valido = test_data[
            test_data["userId"].isin(self._reconstructed_df.index)
            & test_data["movieId"].isin(self._reconstructed_df.columns)
        ]

        if test_valido.empty:
            raise ValueError("Nenhum par (userId, movieId) do teste existe na matriz reconstruída.")

        predicoes = test_valido.apply(
            lambda row: self._reconstructed_df.loc[row["userId"], row["movieId"]], axis=1
        )
        erros_quadraticos = (test_valido["rating"] - predicoes) ** 2
        rmse = float(np.sqrt(erros_quadraticos.mean()))
        return rmse

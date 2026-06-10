"""
Módulo de recomendação baseado em Rede Neural com Embeddings (TensorFlow/Keras).
Usa embeddings de usuários e filmes + features de gênero para prever 'liked'.
"""

import pandas as pd
import numpy as np


class RecomendadorRedeNeural:
    """
    Sistema de recomendação usando Rede Neural com Embeddings.

    Requer TensorFlow instalado. O modelo aprende representações densas
    (embeddings) para usuários e filmes combinadas com features de gênero.

    Attributes:
        embedding_dim (int): Dimensão dos vetores de embedding.
        epochs (int): Número de épocas de treinamento.
        batch_size (int): Tamanho do batch.
        random_state (int): Semente para reprodutibilidade.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        epochs: int = 5,
        batch_size: int = 512,
        random_state: int = 42,
    ):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim deve ser maior que zero.")
        if epochs <= 0:
            raise ValueError("epochs deve ser maior que zero.")
        if batch_size <= 0:
            raise ValueError("batch_size deve ser maior que zero.")

        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self._model = None
        self._user_map = None
        self._movie_map = None
        self._genre_columns = None
        self._df_merged = None

    def _construir_modelo(self, num_users: int, num_movies: int, num_genres: int):
        """Constrói a arquitetura da rede neural com embeddings."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model
        except ImportError:
            raise ImportError("TensorFlow não está instalado. Instale com: pip install tensorflow")

        tf.random.set_seed(self.random_state)

        user_input = layers.Input(shape=(1,), name="userId")
        movie_input = layers.Input(shape=(1,), name="movieId")
        genres_input = layers.Input(shape=(num_genres,), name="genres")

        user_emb = layers.Embedding(num_users, self.embedding_dim)(user_input)
        movie_emb = layers.Embedding(num_movies, self.embedding_dim)(movie_input)

        user_vec = layers.Flatten()(user_emb)
        movie_vec = layers.Flatten()(movie_emb)

        concat = layers.Concatenate()([user_vec, movie_vec, genres_input])
        dense1 = layers.Dense(64, activation="relu")(concat)
        dense2 = layers.Dense(32, activation="relu")(dense1)
        output = layers.Dense(1, activation="sigmoid")(dense2)

        model = Model(inputs=[user_input, movie_input, genres_input], outputs=output)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def treinar(
        self,
        train_df: pd.DataFrame,
        user_map: dict,
        movie_map: dict,
        genre_columns: list,
        df_merged: pd.DataFrame,
        validation_split: float = 0.1,
    ) -> None:
        """
        Treina a rede neural.

        Args:
            train_df (pd.DataFrame): Dados de treino com colunas [userId, movieId, liked, <genres>].
            user_map (dict): Mapeamento userId -> índice.
            movie_map (dict): Mapeamento movieId -> índice.
            genre_columns (list): Colunas de gênero (one-hot).
            df_merged (pd.DataFrame): DataFrame completo para consultas nas recomendações.
            validation_split (float): Fração dos dados de treino usada para validação.

        Raises:
            ValueError: Se 'liked' não existir no train_df.
            ValueError: Se train_df estiver vazio.
        """
        if train_df.empty:
            raise ValueError("O DataFrame de treino está vazio.")
        if "liked" not in train_df.columns:
            raise ValueError("Coluna 'liked' ausente. Chame criar_coluna_liked() antes.")

        self._user_map = user_map
        self._movie_map = movie_map
        self._genre_columns = genre_columns
        self._df_merged = df_merged

        X_user = train_df["userId"].map(user_map).values
        X_movie = train_df["movieId"].map(movie_map).values
        X_genres = train_df[genre_columns].values.astype(np.float32)
        y = train_df["liked"].values

        num_users = len(user_map)
        num_movies = len(movie_map)
        num_genres = len(genre_columns)

        self._model = self._construir_modelo(num_users, num_movies, num_genres)
        self._model.fit(
            [X_user, X_movie, X_genres],
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=1,
        )

    def recomendar(
        self,
        user_id: int,
        movies_df: pd.DataFrame,
        n_recomendacoes: int = 10,
    ) -> pd.DataFrame:
        """
        Recomenda filmes para um usuário usando probabilidades da rede neural.

        Args:
            user_id (int): ID do usuário alvo.
            movies_df (pd.DataFrame): DataFrame de filmes com colunas de gênero.
            n_recomendacoes (int): Quantidade de filmes a recomendar.

        Returns:
            pd.DataFrame: Filmes recomendados com [movieId, title, probabilidade_gostar].

        Raises:
            RuntimeError: Se o modelo não foi treinado.
            ValueError: Se o usuário não estiver no mapeamento de treino.
            ValueError: Se n_recomendacoes for menor ou igual a zero.
        """
        if self._model is None:
            raise RuntimeError("Modelo não treinado. Chame treinar() antes.")
        if n_recomendacoes <= 0:
            raise ValueError("n_recomendacoes deve ser maior que zero.")
        if user_id not in self._user_map:
            raise ValueError(
                f"Usuário {user_id} não está no conjunto de treino. Não é possível recomendar."
            )

        # Filmes ainda não avaliados
        filmes_avaliados = self._df_merged[self._df_merged["userId"] == user_id]["movieId"].unique()
        nao_avaliados = movies_df[~movies_df["movieId"].isin(filmes_avaliados)].copy()
        nao_avaliados = nao_avaliados[nao_avaliados["movieId"].isin(self._movie_map.keys())]

        if nao_avaliados.empty:
            return pd.DataFrame(columns=["movieId", "title", "probabilidade_gostar"])

        user_idx = np.full(len(nao_avaliados), self._user_map[user_id], dtype=np.int32)
        movie_idx = nao_avaliados["movieId"].map(self._movie_map).astype(np.int32).values
        genres_data = nao_avaliados[self._genre_columns].values.astype(np.float32)

        probs = self._model.predict([user_idx, movie_idx, genres_data], verbose=0).flatten()
        nao_avaliados = nao_avaliados.copy()
        nao_avaliados["probabilidade_gostar"] = probs

        resultado = nao_avaliados.sort_values("probabilidade_gostar", ascending=False)
        return resultado.head(n_recomendacoes)[["movieId", "title", "probabilidade_gostar"]].reset_index(drop=True)

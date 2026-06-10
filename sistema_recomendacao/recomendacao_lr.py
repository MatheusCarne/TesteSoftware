"""
Módulo de recomendação baseado em Regressão Logística.
Classifica cada par (usuário, filme) como 'gostou' (1) ou 'não gostou' (0).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


class RecomendadorRegressaoLogistica:
    """
    Sistema de recomendação usando Regressão Logística binária.

    O modelo é treinado sobre features de usuário, filme e gênero,
    e prevê a probabilidade de um usuário gostar de um filme.

    Attributes:
        random_state (int): Semente para reprodutibilidade.
        solver (str): Solver do scikit-learn para Regressão Logística.
    """

    def __init__(self, random_state: int = 42, solver: str = "liblinear"):
        self.random_state = random_state
        self.solver = solver
        self._model = None
        self._feature_columns = None
        self._genre_columns = None
        self._df_merged = None

    def treinar(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_columns: list,
        genre_columns: list,
        df_merged: pd.DataFrame,
    ) -> None:
        """
        Treina o modelo de Regressão Logística.

        Args:
            X_train (pd.DataFrame): Features de treino (já com one-hot encoding).
            y_train (pd.Series): Coluna alvo binária (liked).
            feature_columns (list): Lista de colunas de features brutas.
            genre_columns (list): Colunas de gênero (one-hot).
            df_merged (pd.DataFrame): DataFrame completo para consulta nas recomendações.

        Raises:
            ValueError: Se X_train ou y_train estiverem vazios.
        """
        if X_train.empty or y_train.empty:
            raise ValueError("Os dados de treino não podem estar vazios.")

        self._model = LogisticRegression(
            random_state=self.random_state,
            solver=self.solver,
            max_iter=1000,
        )
        self._model.fit(X_train, y_train)
        self._feature_columns = feature_columns
        self._genre_columns = genre_columns
        self._df_merged = df_merged
        self._X_train_columns = X_train.columns.tolist()

    def avaliar(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Avalia o modelo no conjunto de teste.

        Args:
            X_test (pd.DataFrame): Features de teste.
            y_test (pd.Series): Coluna alvo real.

        Returns:
            dict: {'accuracy': float, 'auc_roc': float}

        Raises:
            RuntimeError: Se o modelo não foi treinado.
        """
        if self._model is None:
            raise RuntimeError("Modelo não treinado. Chame treinar() antes.")

        y_pred = self._model.predict(X_test)
        y_prob = self._model.predict_proba(X_test)[:, 1]
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, y_prob)),
        }

    def recomendar(
        self,
        user_id: int,
        movies_df: pd.DataFrame,
        n_recomendacoes: int = 10,
    ) -> pd.DataFrame:
        """
        Recomenda filmes para um usuário usando probabilidades previstas pelo modelo.

        Args:
            user_id (int): ID do usuário alvo.
            movies_df (pd.DataFrame): DataFrame de filmes com colunas de gênero.
            n_recomendacoes (int): Quantidade de filmes a recomendar.

        Returns:
            pd.DataFrame: Filmes recomendados com colunas [movieId, title, probabilidade_gostar].

        Raises:
            RuntimeError: Se o modelo não foi treinado.
            ValueError: Se n_recomendacoes for menor ou igual a zero.
            ValueError: Se o usuário não existir nos dados.
        """
        if self._model is None:
            raise RuntimeError("Modelo não treinado. Chame treinar() antes.")
        if n_recomendacoes <= 0:
            raise ValueError("n_recomendacoes deve ser maior que zero.")
        if user_id not in self._df_merged["userId"].values:
            raise ValueError(f"Usuário {user_id} não encontrado nos dados.")

        # Filmes que o usuário ainda não avaliou
        filmes_avaliados = self._df_merged[self._df_merged["userId"] == user_id]["movieId"].unique()
        nao_avaliados = movies_df[~movies_df["movieId"].isin(filmes_avaliados)].copy()

        if nao_avaliados.empty:
            return pd.DataFrame(columns=["movieId", "title", "probabilidade_gostar"])

        # Monta features para predição
        predict_data = nao_avaliados[["movieId"] + list(self._genre_columns)].copy()
        predict_data["userId"] = user_id

        X_pred = predict_data[self._feature_columns].copy()
        X_pred["userId"] = X_pred["userId"].astype(str)
        X_pred["movieId"] = X_pred["movieId"].astype(str)

        X_pred_encoded = pd.get_dummies(X_pred, columns=["userId", "movieId"])
        X_pred_encoded = X_pred_encoded.reindex(columns=self._X_train_columns, fill_value=0)

        probs = self._model.predict_proba(X_pred_encoded)[:, 1]
        nao_avaliados = nao_avaliados.copy()
        nao_avaliados["probabilidade_gostar"] = probs

        resultado = nao_avaliados.sort_values("probabilidade_gostar", ascending=False)
        return resultado.head(n_recomendacoes)[["movieId", "title", "probabilidade_gostar"]].reset_index(drop=True)

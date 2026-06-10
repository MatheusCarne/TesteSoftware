"""
Sistema de Recomendação de Filmes — MovieLens
Disciplina: Mineração de Dados / Testes de Software
"""

from .data_loader import DataLoader
from .preprocessamento import Preprocessador
from .recomendacao_svd import RecomendadorSVD
from .recomendacao_lr import RecomendadorRegressaoLogistica
from .recomendacao_nn import RecomendadorRedeNeural

__all__ = [
    "DataLoader",
    "Preprocessador",
    "RecomendadorSVD",
    "RecomendadorRegressaoLogistica",
    "RecomendadorRedeNeural",
]

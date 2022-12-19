import numpy as np
from sentence_transformers import SentenceTransformer

from src.models.embedders.model import Embedder


class SentenceEmbedder(Embedder):
    def __init__(self):
        self._model = SentenceTransformer('sdadas/st-polish-paraphrase-from-distilroberta')

    def get_embedding(self, message: str) -> np.array:
        return self._model.encode(message)


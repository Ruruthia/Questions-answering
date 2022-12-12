import os
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from src.models.embedders.model import Embedder
from src.models.retrieval.model import RetrievalModel


def preprocess_question(question: str) -> str:
    return " ".join(question.split()[3:])[:-1]


class DenseRetrievalModel(RetrievalModel):

    def __init__(self, definitions_path: str, embeddings_model: Embedder, definitions_embeddings_path: str) -> None:
        super().__init__(definitions_path)
        self._embeddings_model = embeddings_model
        if Path(definitions_embeddings_path).is_file():
            with open(definitions_embeddings_path, 'rb') as f:
                self._definitions_embeddings = np.load(f)
        else:
            print("Embedding definitions...")
            self._definitions_embeddings = []
            for definition in tqdm(self._definitions):
                self._definitions_embeddings.append(self._embeddings_model.get_embedding(definition))
            self._definitions_embeddings = np.array(self._definitions_embeddings)

            with open(definitions_embeddings_path, 'wb') as f:
                np.save(f, self._definitions_embeddings)

        print("Definitions embedded!")

    def _match_question(self, question: str, definitions: list[str]) -> int:
        # TODO: How to combine dense + sparse model elegantly? Probably run answer questions differently
        question_embedding = self._embeddings_model.get_embedding(preprocess_question(question))
        closest_definition = (self._definitions_embeddings @ question_embedding) / \
                             (norm(self._definitions_embeddings) * norm(question_embedding) + 1e-10)

        return int(np.argmax(closest_definition))

import os
from pathlib import Path
from stempel import StempelStemmer
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import pandas as pd

from collections import defaultdict as dd

from src.models.embedders.model import Embedder
from src.models.retrieval.model import RetrievalModel


def preprocess_question(question: str) -> str:
    return " ".join(question.split()[3:])[:-1]


class SparseRetrievalModel(RetrievalModel):

    def __init__(self, definitions_path: str) -> None:
        super().__init__(definitions_path)
        self._index = dd(list)

        # Probable extensions: lemmatizing
        self._stemmer = StempelStemmer.polimorf()

        for i in tqdm(range(len(self._definitions)), "Indexing"):
            for word in self._definitions[i].split(" "):
                self._index[self._stemmer.stem(word)].append(i)




    def _match_question(self, question: str, max_answers: int = 10) -> list[int]:
        matches = dd(lambda: 0)
        for word in question.split(" "):
            for i in self._index[self._stemmer.stem(word)]:
                matches[i] += 1
        return pd.Series(matches).sort_values(ascending=False).iloc[:max_answers].index.values.tolist()

if __name__ == "__main__":
    model = SparseRetrievalModel(Path(__file__).parents[3] / 'data' / 'retrieval' / 'plwiktionary.txt')
    print(model._match_question("Jak nazywa się królowa angielska?"))

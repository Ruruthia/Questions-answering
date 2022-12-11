import re
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

# Downloaded from https://github.com/sdadas/polish-nlp-resources
from src.models.embedders.model import Embedder

WORD_TO_VEC_PATH = str(Path(__file__).parents[3] / "data" / "word2vec" / "word2vec_100_3_polish.bin")


def _get_message_tokens(message: str) -> set[str]:
    # This regex splits lowercased message on whitespaces and peels off punctuation
    tokens = re.findall(r"[\w'\"]+|[,.!?]", message.lower())
    # Longer words seem to have more meaningful embeddings
    tokens = [token for token in tokens if len(token) > 4]
    return set(tokens)


class Word2Vec(Embedder):
    def __init__(self, path=WORD_TO_VEC_PATH):
        self._embeddings = KeyedVectors.load(path)

    def get_embedding(self, message: str) -> np.array:
        message_tokens = _get_message_tokens(message)
        tokens_embeddings = []
        for token in message_tokens:
            try:
                token_embedding = self._embeddings.get_vector(token)
            except KeyError:
                continue
            tokens_embeddings.append(token_embedding)
        if len(tokens_embeddings) == 0:
            return np.zeros(100)
        return np.array(tokens_embeddings).mean(axis=0)

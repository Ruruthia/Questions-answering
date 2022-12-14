import os
from pathlib import Path

import pandas as pd
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

import spacy
from stempel import StempelStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict as dd

from src.models.embedders.model import Embedder
from src.models.retrieval.model import RetrievalModel


def preprocess_question(question: str) -> str:
    return " ".join(question.split()[3:])[:-1]


class DenseRetrievalModel(RetrievalModel):

    def __init__(self, definitions_path: str, embeddings_model: Embedder, index_path: str) -> None:
        super().__init__(definitions_path)
        self._embeddings_model = embeddings_model
        # #  probably we could embed only the needed definitions
        # if Path(definitions_embeddings_path).is_file():
        #     with open(definitions_embeddings_path, 'rb') as f:
        #         self._definitions_embeddings = np.load(f)
        # else:
        #     print("Embedding definitions...")
        #     self._definitions_embeddings = []
        #     for definition in tqdm(self._definitions):
        #         self._definitions_embeddings.append(self._embeddings_model.get_embedding(definition))
        #     self._definitions_embeddings = np.array(self._definitions_embeddings)
        #
        #     with open(definitions_embeddings_path, 'wb') as f:
        #         np.save(f, self._definitions_embeddings)
        #
        # print("Definitions embedded!")

        self._stemmer = StempelStemmer.polimorf()
        self._nlp = spacy.load("pl_core_news_md")

        if Path(index_path).is_file():
            with open(index_path, 'rb') as f:
                self._index = np.load(f)
        else:

            self._index = dd(list)
            def my_stem(word: str) -> str:
                res = self._stemmer.stem(word)
                return res if res is not None else ""

            stemmed_definitions = [" ".join([my_stem(word) for word in definition.split(" ")]) for definition in self._definitions]
            # tfidf = TfidfVectorizer().fit(stemmed_definitions)


            for i in tqdm(range(len(self._definitions)), "Indexing"):
                definition = stemmed_definitions[i]
                # tfidf_scores = tfidf.transform([definition]).todense()
                tokens = self._nlp(definition)
                definition_words = definition.split(" ")
                for j in range(len(definition_words)):
                    if j < len(tokens):
                    # if definition_words[j] in tfidf.vocabulary_.keys():
                        # self._index[tokens[j].lemma_].append((i, tfidf_scores[0, tfidf.vocabulary_[definition_words[j]]]))
                        self._index[tokens[j].lemma_].append((i, 1))

            with open(index_path, 'wb') as f:
                np.save(f, self._index)

        print("Index provided!")

    def _get_probable_answers(self, question: str, max_answers: int = 10) -> list[int]:
        matches = dd(lambda: 0)
        for word in question.split(" "):
            for i, score in self._index[self._nlp(self._stemmer.stem(word))[0].lemma_]:
                matches[i] += score
        return pd.Series(matches).sort_values(ascending=False).iloc[:max_answers].index.values.tolist()


    def _match_question(self, question: str, probable_answers: list[int]) -> int:
        question_embedding = self._embeddings_model.get_embedding(preprocess_question(question))

        definitions_embeddings = []
        for i in probable_answers:
            definitions_embeddings.append(self._embeddings_model.get_embedding(self._definitions[i]))
        definitions_embeddings = np.array(definitions_embeddings)

        closest_definition = (definitions_embeddings @ question_embedding) / \
                             (norm(definitions_embeddings) * norm(question_embedding) + 1e-10)

        return probable_answers[int(np.argmax(closest_definition))]

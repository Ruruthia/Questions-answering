import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, List

from sklearn.cluster import KMeans

from src.models.questions_clusterer.model import AnsweredQuestion, QuestionsClusterer
from src.models.word2vec import Word2Vec

QUESTIONS_ANSWERS_PATH = Path(__file__).parents[3] / 'data' / 'def_question.tsv'


def _read_qa_tsv(questions_answers_path: str):
    with open(questions_answers_path, 'r') as in_file:
        csv_file = csv.reader(in_file, delimiter='\t')
        qa_dict = {line[0]: line[1:] for line in csv_file}
    return qa_dict


class Word2VecClusterer(QuestionsClusterer):
    def __init__(self, questions_answers_path=QUESTIONS_ANSWERS_PATH, n_clusters=5):
        self._embeddings_model = Word2Vec()
        self._clustering_model = KMeans(n_clusters=n_clusters)
        self._cluster_to_answered_questions = None

        self._cluster_questions(str(questions_answers_path))

    def _cluster_questions(self, questions_answers_path: str) -> None:
        qa_dict = _read_qa_tsv(questions_answers_path)
        questions = list(qa_dict.keys())
        question_embeddings = [self._embeddings_model.get_embedding(question) for question in questions]

        clusters = self._clustering_model.fit_predict(question_embeddings)

        answered_questions = [AnsweredQuestion(question, answers) for question, answers in qa_dict.items()]

        self._cluster_to_answered_questions = defaultdict(list)
        for cluster, answered_question in zip(clusters, answered_questions):
            self._cluster_to_answered_questions[cluster].append(answered_question)

    def cluster_single_question(self, question: str) -> int:
        embedding = self._embeddings_model.get_embedding(question)
        return self._clustering_model.predict([embedding])[0]

    def sample_questions_from_cluster(
            self,
            cluster_id: int,
            num_questions_to_get: Optional[int] = 5
    ) -> List[AnsweredQuestion]:
        return random.sample(self._cluster_to_answered_questions[cluster_id], k=num_questions_to_get)

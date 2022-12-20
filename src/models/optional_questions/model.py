import numpy as np
from scipy.spatial.distance import cosine


def _process_question(question: str) -> (str, list[str]):
    question = question.lower()
    if question[-1] == '?':
        question = question[:-1]
    if question.startswith('czy'):
        question = question[3:]
    subject, options = question.split(' to ')
    options_a, option_b = options.split(' czy ')
    options = options_a.split(",") + [option_b]
    subject = subject.strip()
    options = [option.strip() for option in options]
    return subject, options


class ABQuestionsAnswerer:
    def __init__(self, embedder):
        self._embedder = embedder

    def answer_question(self, question: str) -> str:
        subject, options = _process_question(question)
        subject_embedding = self._embedder.get_embedding(subject)
        distances = np.array([cosine(subject_embedding, self._embedder.get_embedding(option)) for option in options])
        idx = np.argmin(distances)
        return options[idx]

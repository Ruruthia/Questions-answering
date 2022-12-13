from scipy.spatial.distance import cosine

from src.models.embedders.model import Embedder
from src.models.embedders.sentence_embedder import SentenceEmbedder
from src.models.embedders.word2vec import Word2Vec

EMBEDDER = SentenceEmbedder()
# EMBEDDER = Word2Vec()

QUESTIONS = [
    "Czy bobry to ssaki czy ryby?",
    "Czy cebula to owoc czy warzywo?",
    "Czy hades to bóg umarłych czy bóg morza?",
    "Czy hades to bóg śmierci czy bóg morza?",
    "Czy żelazo to metal czy roślina?",
    "Czy sód to metal czy niemetal?",
]
ANSWERS = [
    'ssaki',
    'warzywo',
    'bóg umarłych',
    'bóg śmierci',
    'metal',
    'metal',
]


def process_question(question: str) -> (str, str, str):
    question = question.lower()
    if question[-1] == '?':
        question = question[:-1]
    if question.startswith('czy'):
        question = question[3:]
    subject, options = question.split('to')
    option_a, option_b = options.split('czy')
    subject = subject.strip()
    option_a = option_a.strip()
    option_b = option_b.strip()
    return subject, option_a, option_b


def answer_question(embedder: Embedder, question: str) -> str:
    subject, option_a, option_b = process_question(question)
    dist_a = cosine(embedder.get_embedding(subject), embedder.get_embedding(option_a))
    dist_b = cosine(embedder.get_embedding(subject), embedder.get_embedding(option_b))
    if dist_a < dist_b:
        return option_a
    return option_b


def main():

    for question, answer in zip(QUESTIONS, ANSWERS):
        print(question)
        print(f"Answer of the model: {answer_question(EMBEDDER, question)}")
        print(f"Correct answer: {answer}")
        print()

    while True:
        question = input()
        print(answer_question(EMBEDDER, question))
        print()


if __name__ == '__main__':
    main()
    
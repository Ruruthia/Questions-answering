from scipy.spatial.distance import cosine

from src.models.embedders.sentence_embedder import SentenceEmbedder

EMBEDDER = SentenceEmbedder()
QUESTIONS = [
    "Czy bobry to ssaki czy ryby?",
    "Czy cebula to owoc czy warzywo?",
    "Czy hades to bóg umarłych czy bóg morza?",
    "Czy żelazo to metal czy roślina?",
    "Czy sód to metal czy nie metal?",
]
ANSWERS = [
    'ssaki',
    'warzywo',
    'bóg umarłych',
    'metal',
    'metal',
]


def process_question(question):
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


def answer_question(embedder, question):
    subject, option_a, option_b = process_question(question)
    dist_a = cosine(embedder.get_embedding(subject), embedder.get_embedding(option_a))
    dist_b = cosine(embedder.get_embedding(subject), embedder.get_embedding(option_b))
    if dist_a < dist_b:
        return option_a
    return option_b


def main():

    for question, answer in zip(QUESTIONS, ANSWERS):
        print(question)
        print(answer_question(EMBEDDER, question))
        print(answer)
        print()

    while True:
        question = input()
        print(answer_question(EMBEDDER, question))
        print()


if __name__ == '__main__':
    main()
    
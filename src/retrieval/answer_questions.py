from pathlib import Path

from src.models.embedders.sentence_embedder import SentenceEmbedder
from src.models.embedders.word2vec import Word2Vec
from src.models.retrieval.dense_model import DenseRetrievalModel
from src.models.retrieval.model import RetrievalModel
from src.utils import read_qa_tsv, match

MODEL_TYPE = "BERT"
DEFINITIONS_PATH = Path(__file__).parents[2] / 'data' / 'retrieval' / 'plwiktionary.txt'
QUESTIONS_ANSWERS_PATH = Path(__file__).parents[2] / 'data' / 'questions_answers' / 'def_question.tsv'


def create_model() -> RetrievalModel:
    # More embeddings can be found here: https://github.com/ksopyla/awesome-nlp-polish
    if MODEL_TYPE == "W2V":
        definitions_embeddings_path = Path(__file__).parents[2] / 'data' / 'retrieval' / 'w2v_embeddings.npz'
        model = DenseRetrievalModel(
            definitions_path=str(DEFINITIONS_PATH),
            definitions_embeddings_path=str(definitions_embeddings_path),
            embeddings_model=Word2Vec())
    elif MODEL_TYPE == "BERT":
        definitions_embeddings_path = Path(__file__).parents[2] / 'data' / 'retrieval' / 'roberta_embeddings.npz'
        model = DenseRetrievalModel(
            definitions_path=str(DEFINITIONS_PATH),
            definitions_embeddings_path=str(definitions_embeddings_path),
            embeddings_model=SentenceEmbedder())
    else:
        raise NotImplementedError()
    return model


def main():
    data = read_qa_tsv(questions_answers_path=str(QUESTIONS_ANSWERS_PATH))
    model = create_model()
    score = 0

    for question, correct_answers in data.items():
        model_answer = model.answer_question(question)
        if match(model_answer, correct_answers):
            print(question)
            print(correct_answers)
            print(model_answer)
            score += 1

    print(f"Accuracy: {100 * score / len(data):.2f} %")
    # W2V: 4.45%
    # BERT: 19.84%


if __name__ == '__main__':
    main()

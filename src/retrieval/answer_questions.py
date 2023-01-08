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
        index_path = Path(__file__).parents[2] / 'data' / 'retrieval' / 'index.npz'
        definitions_embeddings_path = Path(__file__).parents[2] / 'data' / 'retrieval' / 'w2v_embeddings.npz'
        model = DenseRetrievalModel(
            definitions_path=str(DEFINITIONS_PATH),
            index_path=str(index_path),
            definitions_embeddings_path=definitions_embeddings_path,
            embeddings_model=Word2Vec())
    elif MODEL_TYPE == "BERT":
        index_path = Path(__file__).parents[2] / 'data' / 'retrieval' / 'index.npz'
        definitions_embeddings_path = Path(__file__).parents[2] / 'data' / 'retrieval' / 'roberta_embeddings.npz'
        model = DenseRetrievalModel(
            definitions_path=str(DEFINITIONS_PATH),
            index_path=str(index_path),
            definitions_embeddings_path=definitions_embeddings_path,
            embeddings_model=SentenceEmbedder())
    else:
        raise NotImplementedError()
    return model


def main():
    data = read_qa_tsv(questions_answers_path=str(QUESTIONS_ANSWERS_PATH))
    model = create_model()
    score = 0

    for question, correct_answers in data.items():
        model_answer = model.answer_question(question, 2)
        if match(model_answer, correct_answers):
            score += 1
        print(question)
        print(correct_answers)
        print(model_answer)


    print(f"Accuracy: {100 * score / len(data):.2f} %")
    # W2V: 4.45%  (+ Sparse 10: 11.34 %, 3: 10.12 %,
    # SPARSE (with stemmer): 5.26 %
    # BERT: 19.84% (+ Sparse 10: 13.77 %, 3: 10.12 %, 25: 17.41 %)


if __name__ == '__main__':
    main()

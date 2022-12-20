import re
from pathlib import Path

from tqdm import tqdm

from src.models.embedders.sentence_embedder import SentenceEmbedder
from src.models.language_model_question_answerer.model import LanguageModelQuestionAnswerer
from src.models.optional_questions.model import ABQuestionsAnswerer
from src.models.papugapt import PapuGaPT2
from src.models.questions_clusterer.w2v_clusterer import Word2VecClusterer
from src.models.retrieval.dense_model import DenseRetrievalModel
from src.utils import read_qa_tsv, scaled_editdist, match

TRAIN_QUESTIONS_ANSWERS_PATH = Path(__file__).parents[2] / 'data' / 'questions_answers' / 'def_question.tsv'
QUESTIONS_ANSWERS_PATH = Path(__file__).parents[2] / 'data' / 'qa_competition' / 'task2_questions_with_answers.tsv'
DEFINITIONS_PATH = Path(__file__).parents[2] / 'data' / 'retrieval' / 'plwiktionary.txt'
DEFINITIONS_EMBEDDINGS_PATH = Path(__file__).parents[2] / 'data' / 'retrieval' / 'roberta_embeddings.npz'
KNOWLEDGE_RETRIEVAL_PREFIXES = [
    "Jak nazywa się",
    "Jak potocznie nazywamy",
    "Jak nazywamy",
    "Jak z {} nazywamy",
]

PAPUGAPT_GENERATION_CONFIG = {
    "do_sample": True,
    "top_k": 100,
    "top_p": 0.75,
    "max_new_tokens": 10,
    "num_beams": 2,
    "no_repeat_ngram_size": 2,
}

MODELS = {
    "KR": DenseRetrievalModel(
        definitions_path=str(DEFINITIONS_PATH),
        definitions_embeddings_path=str(DEFINITIONS_EMBEDDINGS_PATH),
        embeddings_model=SentenceEmbedder()),
    "AB": ABQuestionsAnswerer(
        embedder=SentenceEmbedder()
    ),
    "YN": PapuGaPT2(),
    "G": LanguageModelQuestionAnswerer(
        language_model=PapuGaPT2(),
        clusterer=Word2VecClusterer(
            qa_dict=read_qa_tsv(questions_answers_path=str(TRAIN_QUESTIONS_ANSWERS_PATH)),
            n_clusters=15,
        ),
        num_questions_to_sample=3,
        generation_config=PAPUGAPT_GENERATION_CONFIG
    )
}


def is_knowledge_retrieval_question(question: str) -> bool:
    question = question.split()
    for prefix in KNOWLEDGE_RETRIEVAL_PREFIXES:
        if scaled_editdist(" ".join(question[:len(prefix.split())]), prefix) < 0.2:
            return True
    if question[0] == "Jak" and question[1] == "z" and question[3] in ["nazywamy", "nazywa", "nazywał"]:
        return True
    return False


def is_a_or_b_question(question: str) -> bool:
    return bool(re.match(".* to .* czy .*", question))


def is_a_yes_no_question(question: str) -> bool:
    return question.split()[0] == "Czy"


def choose_model(question: str):
    if is_knowledge_retrieval_question(question):
        return MODELS["KR"]
    elif is_a_or_b_question(question):
        return MODELS["AB"]
    elif is_a_yes_no_question(question):
        return MODELS["YN"]
    else:
        return MODELS["G"]


def main():
    data = read_qa_tsv(questions_answers_path=str(QUESTIONS_ANSWERS_PATH))
    answers = []
    all_correct_answers = []
    for question, correct_answers in tqdm(list(data.items())):
        current_model = choose_model(question)
        if isinstance(current_model, PapuGaPT2):
            answer = current_model.respond_to_yes_no_question(question)
        else:
            answer = current_model.answer_question(question)
        answers.append(answer.split('\n')[0])
        all_correct_answers.append(correct_answers)

    with open(Path(__file__).parents[2] / 'data' / 'qa_competition' / 'answers.txt', 'w') as f:
        for answer in answers:
            f.write(f"{answer}\n")
    with open(Path(__file__).parents[2] / 'data' / 'qa_competition' / 'correct_answers.txt', 'w') as f:
        for answer in all_correct_answers:
            line = "\t".join(answer)
            f.write(f"{line}\n")


if __name__ == '__main__':
    main()

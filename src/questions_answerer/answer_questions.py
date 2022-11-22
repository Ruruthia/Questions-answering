import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.papugapt import PapuGaPT2
from src.models.questions_clusterer.model import QuestionsClusterer
from src.models.questions_clusterer.w2v_clusterer import read_qa_tsv, Word2VecClusterer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

QUESTIONS_ANSWERS_PATH = Path(__file__).parents[2] / 'data' / 'questions_answers' / 'def_question.tsv'
GENERATION_CONFIG = {
    "do_sample": True,
    "top_k": 100,
    "top_p": 0.75,
    "max_new_tokens": 5,
    "num_beams": 2,
    "no_repeat_ngram_size": 2,
}
NUM_CLUSTERS = 15
QA_SAMPLED_FROM_CLUSTER = 10


def split_data(
        questions_answers_path: str = str(QUESTIONS_ANSWERS_PATH),
) -> (dict[str, List[str]], dict[str, List[str]]):
    data = read_qa_tsv(questions_answers_path=questions_answers_path)
    data_pd = pd.Series(data)
    return [i.to_dict() for i in train_test_split(data_pd, train_size=0.8)]


def prepare_prompt(
        question: str,
        clusterer: QuestionsClusterer
) -> str:
    """
    Gets questions similar to the question
    and prepares the prompt by processing the questions & answers and joining them using the EOS token.
    """
    cluster_id = clusterer.cluster_single_question(question)
    sampled_qas = clusterer.sample_questions_from_cluster(cluster_id, QA_SAMPLED_FROM_CLUSTER)
    return " ### ".join(
        # From "Jak nazywa się ...?" to "... nazywa się ..."
        [f"{(' '.join(qa.question.split()[3:]))[:-1]} nazywa się {qa.answers[0]}" for qa in sampled_qas]
        + [f"{(' '.join(question.split()[3:]))[:-1]} nazywa się "]
    )


print("Splitting data...")
train_data, test_data = split_data()
print("Initializing clusters...")
# TODO: Switch between models
w2v_clusterer = Word2VecClusterer(qa_dict=train_data, n_clusters=NUM_CLUSTERS)
print("Clusters initialized!")
questions_answerer = PapuGaPT2()

for (q, a) in test_data.items():
    print(f"Pytanie: {q}")
    prompt = prepare_prompt(q, w2v_clusterer)
    response = questions_answerer.respond_to_prompt(
        prompt=prompt,
        generation_config=GENERATION_CONFIG,
        end_sequence="###",
    )[0]
    # Get first word of the response
    response = response[len(prompt):].split(' ')[0]
    print(f"Odpowiedź: {response}")

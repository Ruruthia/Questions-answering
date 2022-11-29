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

# QUESTIONS_ANSWERS_PATH = Path(__file__).parents[2] / 'data' / 'questions_answers' / 'def_question.tsv'
QUESTIONS_ANSWERS_PATH_TRAIN = Path(__file__).parents[2] / 'data' / 'QA' / '2021-question-answering' / 'dev-0'
QUESTIONS_ANSWERS_PATH_TEST = Path(__file__).parents[2] / 'data' / 'QA' / '2021-question-answering' / 'test-A'
GENERATION_CONFIG = {
    "do_sample": True,
    "top_k": 100,
    "top_p": 0.75,
    "max_new_tokens": 5,
    "num_beams": 2,
    "no_repeat_ngram_size": 2,
}
QA_SAMPLED_FROM_CLUSTER = 10

def prepare_prompt(
        question: str,
        clusterer: QuestionsClusterer
) -> str:
    """
    Gets questions similar to the question
    and prepares the prompt by processing the questions & answers and joining them using the EOS token.
    """
    cluster_id = clusterer.cluster_single_question(question)
    sampled_qas = clusterer.sample_questions_from_cluster(cluster_id, QA_SAMPLED_FROM_CLUSTER, question)
    return " ### ".join([" ".join(reversed(
        q.question.replace("\n",  "").split(" "))) + ":  " + q.answers[0] for q in sampled_qas]) + \
           " ### " + " ".join(reversed(question.replace("\n",  "").split(" "))) + ":  "


with open(f"{QUESTIONS_ANSWERS_PATH_TEST}/expected.tsv", 'r') as f:
    answers_test = f.readlines()
    answers_test = [a.strip().split("\t") for a in answers_test]

with open(f"{QUESTIONS_ANSWERS_PATH_TEST}/in.tsv", 'r') as f:
    questions_test = f.readlines()

test_data = {questions_test[i]: answers_test[i] for i in range(len(answers_test))}

print("Initializing clusters...")
clusterer = QuestionsClusterer()
clusterer._cluster_questions(QUESTIONS_ANSWERS_PATH_TRAIN)
print("Clusters initialized!")
questions_answerer = PapuGaPT2()



for (q, a) in test_data.items():
    print(f"Pytanie: {q}")
    prompt = prepare_prompt(q, clusterer)
    response = questions_answerer.respond_to_prompt(
        prompt=prompt,
        generation_config=GENERATION_CONFIG,
        end_sequence="###",
    )[0]
    # Get first word of the response
    response = " ".join(response[len(prompt):].split(' ')[:3])
    print(f"Odpowiedź: {response}")

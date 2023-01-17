import re
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.language_model_text_generator.model import LanguageModelTextGenerator
from src.models.rule_based.model import TaskOrientedChatbot
from src.models.embedders.word2vec import Word2Vec

END_OF_CONVERSATION_PROMPT = "Do widzenia!"
GENERATION_CONFIG = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "num_beams": 3,
    "num_return_sequences": 5,
    "no_repeat_ngram_size": 2,
    "max_new_tokens": 30,
}

CONVERSATION_SAMPLES = """A: Jak się masz?
B: Dobrze, a Ty?
A: Też nieźle. Co dziś robisz?
B: Idę do biblioteki, muszę uczyć się na egzamin w przyszłym tygodniu.
A: O nie. Ok, w takim razie porozmawiamy później. Powodzenia!
B: Dzięki. Do zobaczenia.
###
A: Cześć!
B: Hej!
A: Czy lubisz oglądać filmy?
B: Lubię, zwłaszcza w kinie.
A: Jaki jest Twój ulubiony gatunek?
B: Najbardziej lubię filmy akcji.
###"""
NUM_CONVERSATION_SAMPLES = len(CONVERSATION_SAMPLES.split('###')) - 1
VERBOSE = False

model = LanguageModelTextGenerator(
            model=AutoModelForCausalLM.from_pretrained('flax-community/papuGaPT2'),
            tokenizer=AutoTokenizer.from_pretrained('flax-community/papuGaPT2')
        )
embeddings_model = Word2Vec()
task_model = TaskOrientedChatbot(Path(__file__).parent.parent)


def modify_prompt(raw_prompt: str, conversation_history: str) -> str:
    prompt_start = f"""
A: {raw_prompt}
B:"""

    return conversation_history + prompt_start


def modify_response(response: str, history_len: int) -> str:
    # Take the first block after our samples
    # We can't simply take the last, as the model would sometimes generate '###' in response
    response = response.split('###')[NUM_CONVERSATION_SAMPLES]
    # Leave only the part after the appropriate 'B:'
    response = response.split('B:')[history_len + 1]
    # Leave only the first line
    response = response.partition('\n')[0]
    # Remove everything after the last '.', '!', or '?'.
    last_sentence_end_index = max(response.rfind('.'), response.rfind('?'), response.rfind('!'))
    if last_sentence_end_index != -1:  # -1 means none were found
        response = response[:last_sentence_end_index + 1]
    return response.strip()


def get_best_response(responses_list: list[str], history_len: int, prompt_embedding: np.array) -> str:
    responses = [modify_response(response, history_len) for response in responses_list]
    scored_responses = score_responses(prompt_embedding, responses)
    if VERBOSE:
        print("----SCORED RESPONSES-----")
        for response, score in scored_responses:
            print(f'{response}: {score}')
        print("---------")
    return scored_responses[0][0]


def score_responses(prompt_embedding: np.array, responses: list[str]) -> list[(str, float)]:
    scored_responses = []
    for response in responses:
        response_embedding = embeddings_model.get_embedding(response)
        score = (prompt_embedding @ response_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(response_embedding) + 1e-100)
        scored_responses.append((response, score))
    return sorted(scored_responses, key=lambda x: x[1], reverse=True)


def detect_task(raw_prompt: str) -> bool:
    pattern = re.compile(".*(gdzie|w której sali).*(zajęcia|ćwiczenia|wykład|laboratoria).*")
    res = pattern.search(raw_prompt.lower())
    return res is not None


if __name__ == "__main__":
    prompt = None
    continue_task = False
    history = ""
    history_len = 0
    print("Przywitaj się")
    while True:
        prompt = input()
        if prompt == END_OF_CONVERSATION_PROMPT:
            break
        if continue_task or detect_task(prompt):
            response = task_model.interact(prompt)
            continue_task = not task_model.is_completed()
            print(response)
        else:
            prompt_embedding = embeddings_model.get_embedding(prompt)
            prompt = modify_prompt(prompt, history)
            responses = model.respond_to_prompt(
                prompt=CONVERSATION_SAMPLES + prompt,
                generation_config=GENERATION_CONFIG,
            )
            response = get_best_response(
                responses_list=responses,
                history_len=history_len,
                prompt_embedding=prompt_embedding
            )
            print(response)
            history_len += 1
            history = f"{prompt} {response}"

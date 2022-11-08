from src.models.papugapt import PapuGaPT2
from src.models.rule_based.model import TaskOrientedChatbot
import re
from pathlib import Path

END_OF_CONVERSATION_PROMPT = "Do widzenia!"
GENERATION_CONFIG = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
    "no_repeat_ngram_size": 3,
    "max_new_tokens": 50,
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
    if last_sentence_end_index != -1:   # -1 means none were found
        response = response[:last_sentence_end_index+1]
    return response.strip()


def get_best_response(responses_list: list[str], history_len: int) -> str:
    # TODO
    return modify_response(responses_list[0], history_len)


def detect_task(raw_prompt: str) -> bool:
    pattern = re.compile(".*(gdzie|w której sali).*(zajęcia|ćwiczenia|wykład|laboratoria).*")
    res = pattern.search(raw_prompt.lower())
    return res is not None

model = PapuGaPT2()
task_model = TaskOrientedChatbot(Path(__file__).parent.parent)

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
          prompt = modify_prompt(prompt, history)
          responses = model.respond_to_prompt(
              prompt=CONVERSATION_SAMPLES + prompt,
              generation_config=GENERATION_CONFIG,
              end_sequence="###",
          )
          response = get_best_response(responses, history_len)
          print(response)
          history_len += 1
          history = f"{prompt} {response}"
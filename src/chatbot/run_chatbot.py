from src.models.papugapt import PapuGaPT2
from src.models.rule_based.model import TaskOrientedChatbot

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
###
"""
NUM_CONVERSATION_SAMPLES = len(CONVERSATION_SAMPLES.split('###')) - 1

def modify_prompt(raw_prompt: str) -> str:

    prompt_start = f"""A: {raw_prompt}
B:"""

    return CONVERSATION_SAMPLES + prompt_start


def modify_response(response: str) -> str:
    # Take the first block after our samples
    # We can't simply take the last, as the model would sometimes generate '###' in response
    response = response.split('###')[NUM_CONVERSATION_SAMPLES]
    # Leave only the part after the first 'B:'
    response = response.partition('B:')[2]
    # Leave only the first line
    response = response.partition('\n')[0]
    # Remove everything after the last '.', '!', or '?'.
    last_sentence_end_index = max(response.rfind('.'), response.rfind('?'), response.rfind('!'))
    if last_sentence_end_index != -1:   # -1 means none were found
        response = response[:last_sentence_end_index+1]
    return response.strip()


def get_best_response(responses_list: List[str]) -> str:
    # TODO
    return modify_response(responses_list[0])


def detect_task(raw_prompt: str) -> bool:
    # TODO
    return False

model = PapuGaPT2()
task_model = TaskOrientedChatbot()

if __name__ == "__main__":
    prompt = None
    continue_task = False
    print("Przywitaj się")
    while prompt != END_OF_CONVERSATION_PROMPT:
        prompt = input()
        if continue_task or detect_task(prompt):
            response = task_model.interact(prompt)
            continue_task = not task_model.is_completed()
            print(response)
        else:
          responses = model.respond_to_prompt(
              prompt=modify_prompt(prompt),
              generation_config=GENERATION_CONFIG,
              end_sequence="###",
          )
          print(get_best_response(responses))


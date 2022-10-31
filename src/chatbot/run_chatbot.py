from src.models.papugapt import PapuGaPT2

END_OF_CONVERSATION_PROMPT = "Do widzenia!"
GENERATION_CONFIG = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
    "no_repeat_ngram_size": 3,
}


def modify_prompt(raw_prompt: str) -> str:
    conversation_samples = """A: Jak się masz?
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

    prompt_start = f"""A: {raw_prompt}
    B:"""

    return conversation_samples + prompt_start


def modify_response(response: str) -> str:
    # TODO
    return response


def get_best_response(responses_list: list[str]) -> str:
    # TODO
    return modify_response(responses_list[0])


model = PapuGaPT2()

if __name__ == "__main__":
    prompt = None
    print("Przywitaj się")
    while prompt != END_OF_CONVERSATION_PROMPT:
        prompt = input()
        responses = model.respond_to_prompt(
            prompt=modify_prompt(prompt),
            generation_config=GENERATION_CONFIG,
            end_sequence="###",
            max_response_length=20,
        )
        print(get_best_response(responses))

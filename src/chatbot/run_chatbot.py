from src.models.papugapt import PapuGaPT2

END_OF_CONVERSATION_PROMPT = "Do widzenia!"
GENERATION_CONFIG = {
    "do_sample": True,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
}


def modify_prompt(raw_prompt: str) -> str:
    new_prompt = f"""
    Pytanie: {raw_prompt}
    Odpowiedź: 
    """
    return new_prompt


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
        )
        print(get_best_response(responses))

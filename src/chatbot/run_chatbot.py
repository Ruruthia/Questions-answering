from typing import List

from src.models.papugapt import PapugaPT2

END_OF_CONVERSATION_PROMPT = "Do widzenia!"
GENERATE_CONFIG = {
    "do_sample": True,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
}


def modify_prompt(raw_prompt: str) -> str:
    # TODO
    return raw_prompt


def modify_response(response: str) -> str:
    # TODO
    return response


def get_best_response(responses_list: List[str]) -> str:
    # TODO
    return modify_response(responses_list[0])


model = PapugaPT2()

if __name__ == "__main__":
    while True:
        prompt = input()
        responses = model.respond_to_prompt(
            prompt=modify_prompt(prompt),
            generate_config=GENERATE_CONFIG,
        )
        print(get_best_response(responses))
        if prompt == END_OF_CONVERSATION_PROMPT:
            break

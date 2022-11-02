from src.models.papugapt import PapuGaPT2
from src.models.rule_based.model import TaskOrientedChatbot
from typing import List

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
            )
            print(get_best_response(responses))

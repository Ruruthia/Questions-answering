from src.models.papugapt import PapugaPT2

GENERATE_CONFIG = {
    "do_sample": True,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
}

PROMPT = "Hej! Jak Ci mija dzień?"

if __name__ == "__main__":

    model = PapugaPT2()
    responses = model.respond_to_prompt(PROMPT, GENERATE_CONFIG)
    for i in range(GENERATE_CONFIG["num_return_sequences"]):
        print(f'{i}: {responses[i]} \n')

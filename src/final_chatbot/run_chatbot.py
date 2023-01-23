import argparse

from awscli.compat import raw_input
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, PreTrainedTokenizer

from src.models.language_model_text_generator.model import LanguageModelTextGenerator
from src.models.translator.model import Translator

END_OF_CONVERSATION_PROMPT = "Do widzenia!"

GENERATION_CONFIG = {
    "max_new_tokens": 50
}


def modify_prompt(raw_prompt: str, history: list[str], tokenizer: PreTrainedTokenizer) -> str:

    def crop_history() -> list[str]:
        cropped_history = []
        cropped_history_token_len = 0
        prompt_token_len = len(tokenizer.encode(raw_prompt))
        i = 1
        while i <= len(history):
            cropped_history_token_len += len(tokenizer.encode("</s> <s>" + history[-i]))
            if cropped_history_token_len >= 127 - prompt_token_len:
                break
            cropped_history.append(history[-i])
            i += 1
        print(cropped_history_token_len)
        cropped_history.reverse()
        print(cropped_history)
        return cropped_history

    if len(history):
        return "</s> <s>".join(crop_history() + [raw_prompt])
    else:
        return raw_prompt


def modify_response(response: str) -> str:
    # Leave only the first line
    response = response.partition('\n')[0]
    # Remove everything after the last '.', '!', or '?'.
    last_sentence_end_index = max(response.rfind('.'), response.rfind('?'), response.rfind('!'))
    if last_sentence_end_index != -1:  # -1 means none were found
        response = response[:last_sentence_end_index + 1]
    return response.strip()


def run_chatbot(model: LanguageModelTextGenerator, translator: Translator) -> None:
    history = []
    print("Przywitaj się!")
    while True:
        prompt = translator.translate(raw_input(), source_lang="PL", target_lang="EN-GB")
        if prompt == END_OF_CONVERSATION_PROMPT:
            break
        history.append(prompt)
        prompt = modify_prompt(prompt, history, model.tokenizer)
        response = model.respond_to_prompt(
            prompt=prompt,
            generation_config=GENERATION_CONFIG,
        )[0]
        response = modify_response(
            response=response,
        )
        print("Odpowiedź:", translator.translate(response, source_lang="EN", target_lang="PL"))
        history.append(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Chatbot based on DeepL translator + Blenderbot model + ChatScript",
        description="Script to run final chatbot.",
    )
    parser.add_argument("--auth-key", help="API key for DeepL", required=True)
    args = parser.parse_args()

    model = LanguageModelTextGenerator(
        model=BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill"),
        tokenizer=BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    )
    translator = Translator(auth_key=args.auth_key)
    run_chatbot(model=model, translator=translator)

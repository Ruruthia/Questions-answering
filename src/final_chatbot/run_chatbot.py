import argparse

from awscli.compat import raw_input
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

from src.models.language_model_text_generator.model import LanguageModelTextGenerator
from src.models.translator.model import Translator
from src.utils import modify_prompt, modify_response

END_OF_CONVERSATION_PROMPT = "Do widzenia!"

GENERATION_CONFIG = {
    "max_new_tokens": 50
}


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

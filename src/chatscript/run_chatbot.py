import click
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

from src.chatscript.client import ChatScriptClient
from src.models.language_model_text_generator.model import LanguageModelTextGenerator
from src.models.translator.model import Translator
from src.utils import modify_prompt, modify_response

FALLBACK_MESSAGE = ' '
END_OF_CONVERSATION_PROMPT = ':quit'

GENERATION_CONFIG = {
    "max_new_tokens": 50
}


def run_chatbot(
        name: str,
        bot: str,
        cs_client: ChatScriptClient,
        model: LanguageModelTextGenerator,
        translator: Translator
) -> None:
    history: list[str] = []

    print("Hi " + name + ", enter ':quit' to end this session")

    prompt = input(f"[{name}]: ").lower().strip()
    while prompt != END_OF_CONVERSATION_PROMPT:
        # Ensure empty strings are padded with at least one space before sending to the
        # server, as per the required protocol
        if prompt == "":
            prompt = " "

        prompt = translator.translate(prompt, source_lang="PL", target_lang="EN-GB")
        # Send this to the server and print the response
        # Put in null terminations as required
        msg = u'%s\u0000%s\u0000%s\u0000' % (name, bot, prompt)
        msg = str.encode(msg)
        response = cs_client.send_and_receive_message(msg)

        if response is None:
            raise RuntimeError("Error communicating with Chat Server")

        elif response == FALLBACK_MESSAGE:
            print("FALLBACK")
            modified_prompt = modify_prompt(prompt, history, model.tokenizer)
            print(modified_prompt)
            response = model.respond_to_prompt(
                prompt=modified_prompt,
                generation_config=GENERATION_CONFIG,
            )[0]
            response = modify_response(
                response=response,
            )

        history.append(prompt)
        history.append(response)

        response = translator.translate(response, source_lang="EN", target_lang="PL")
        print(f"[{bot}]: {response}")

        prompt = input(f"[{name}]: ").lower().strip()


@click.command()
@click.argument('name', type=click.STRING)
@click.argument('deepl_auth_key', type=click.STRING)
@click.option('--server', type=click.STRING, default='127.0.0.1')
@click.option('--port', type=click.INT, default=1024)
@click.option('--bot', type=click.STRING, default='CriStian')
def main(name: str, deepl_auth_key: str, server: str, port: int, bot: str):
    cs_client = ChatScriptClient(server, port)
    translator = Translator(deepl_auth_key)
    model = LanguageModelTextGenerator(
        model=BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill"),
        tokenizer=BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    )

    run_chatbot(name, bot, cs_client, model, translator)


if __name__ == '__main__':
    main()

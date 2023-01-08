import click
from awscli.compat import raw_input

from src.models.translator.model import Translator


@click.command()
@click.option('--auth-key', help="Key to DeepL API.")
def translate_prompt(auth_key: str):
    translator = Translator(auth_key=auth_key)
    while True:
        # raw_input is needed to handle polish characters
        prompt = raw_input("Sentence to translate (POLISH TO ENGLISH): ")
        print(translator.translate(prompt, source_lang="PL", target_lang="EN-GB"))
        prompt = raw_input("Sentence to translate (ENGLISH TO POLISH): ")
        print(translator.translate(prompt, source_lang="EN", target_lang="PL"))


if __name__ == '__main__':
    translate_prompt()

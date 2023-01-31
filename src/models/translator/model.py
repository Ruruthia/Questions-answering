import deepl


class Translator:
    def __init__(self, auth_key: str):
        self._translator = deepl.Translator(auth_key=auth_key)

    def translate(self, prompt: str, source_lang: str = "PL", target_lang: str = "EN-GB") -> str:
        return self._translator.translate_text(prompt, source_lang=source_lang, target_lang=target_lang).text

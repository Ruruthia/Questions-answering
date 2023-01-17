from typing import Any

from pytorch_lightning import LightningModule
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel


class LanguageModelTextGenerator(LightningModule):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        set_seed(42)

    def respond_to_yes_no_question(self, question: str) -> str:
        encoded_question = self._tokenizer.encode(question, return_tensors="pt")
        continuation_logits = self._model(encoded_question).logits[0, -1]

        yes_idx = self._tokenizer.encode("tak", return_tensors="pt")
        no_idx = self._tokenizer.encode("nie", return_tensors="pt")
        yes_score, no_score = continuation_logits[yes_idx], continuation_logits[no_idx]
        if yes_score >= no_score:
            return "tak"
        else:
            return "nie"

    def respond_to_prompt(
            self,
            prompt: str,
            generation_config: dict[str, Any],
            end_sequence: str,
    ) -> list[str]:
        input_ids = self._tokenizer.encode(prompt, return_tensors='pt')
        output = self._model.generate(
            input_ids,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.convert_tokens_to_ids(end_sequence),
            **generation_config
        )
        return list(map(lambda x: self._tokenizer.decode(x, skip_special_tokens=True), output))

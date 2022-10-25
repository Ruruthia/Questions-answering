from typing import Any, Dict, List

from pytorch_lightning import LightningModule
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM


class PapuGaPT2(LightningModule):
    def __init__(self):
        super().__init__()
        self._model = AutoModelForCausalLM.from_pretrained('flax-community/papuGaPT2')
        self._tokenizer = AutoTokenizer.from_pretrained('flax-community/papuGaPT2')
        set_seed(42)

    def respond_to_prompt(self, prompt: str, generation_config: Dict[str, Any]) -> List[str]:
        input_ids = self._tokenizer.encode(prompt, return_tensors='pt')
        output = self._model.generate(
            input_ids,
            pad_token_id=self._tokenizer.eos_token_id,
            **generation_config
        )
        return list(map(lambda x: self._tokenizer.decode(x, skip_special_tokens=True), output))

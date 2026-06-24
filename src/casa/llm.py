import os
import json
from pathlib import Path
from typing import Optional, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class LLM:
    """Wrapper around a HuggingFace causal LM used for constrained generation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_id: str,
        is_chat_model: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = model.device
        self.is_chat_model = is_chat_model

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        is_chat_model: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        hf_token: Optional[str] = None,
        **model_kwargs,
    ) -> 'LLM':
        """Load a model and tokenizer from HuggingFace.

        The HF token is taken from hf_token, then $HF_TOKEN, then a local
        secrets.json ({"HF_TOKEN": ...}) if present.
        """
        if hf_token is not None:
            os.environ["HF_TOKEN"] = hf_token
        elif "HF_TOKEN" not in os.environ:
            secrets_path = Path("secrets.json")
            if secrets_path.exists():
                with open(secrets_path) as f:
                    secrets = json.load(f)
                    if "HF_TOKEN" in secrets and secrets["HF_TOKEN"] != "your_token":
                        os.environ["HF_TOKEN"] = secrets["HF_TOKEN"]

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
            **model_kwargs,
        )
        model.eval()

        return cls(model=model, tokenizer=tokenizer, model_id=model_id, is_chat_model=is_chat_model)

    def format_prompt(self, prompt: str) -> str:
        """Apply the chat template for instruct models, else return as-is."""
        if self.is_chat_model:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            assert isinstance(formatted, str)
            return formatted
        return prompt

    def encode(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors=return_tensors)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def batch_decode(self, token_ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)

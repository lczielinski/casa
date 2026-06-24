from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import torch
from casa.llm import LLM
from casa.grammar import Grammar


@dataclass
class SamplingResult:
    """A single generated program and its log probabilities."""
    tokens: List[str]
    token_ids: List[int]
    text: str
    raw_logprob: float
    constrained_logprob: Optional[float] = None
    success: bool = True
    attempts: int = 1


class BaseSampler(ABC):
    """Base class for samplers; subclasses implement sample()."""

    def __init__(self, llm, grammar, max_new_tokens: int = 512):
        if not isinstance(llm, LLM):
            raise TypeError(f"llm must be an LLM instance, got {type(llm)}")
        if not isinstance(grammar, Grammar):
            raise TypeError(f"grammar must be a Grammar instance, got {type(grammar)}")

        self.llm = llm
        self.grammar = grammar
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def sample(self, prompt: str, n_samples: int = 1, **kwargs) -> List[SamplingResult]:
        ...

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Format and tokenize the prompt onto the model device."""
        formatted_prompt = self.llm.format_prompt(prompt)
        return self.llm.tokenizer.encode(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.llm.device)

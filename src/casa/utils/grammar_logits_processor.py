"""Grammar-constrained logits processor for MCMC sampling."""

import torch
import xgrammar
from transformers.generation.logits_process import LogitsProcessor


class GrammarLogitsProcessor(LogitsProcessor):
    """Masks logits each step to the tokens the grammar still allows."""

    def __init__(
        self,
        tokenizer,
        grammar_constraint,
        device: torch.device,
        prompt_length: int,
    ):
        self.tokenizer = tokenizer
        self.grammar_constraint = grammar_constraint
        self.device = device
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        generated_tokens = self._get_generated_tokens(input_ids)

        if not self.grammar_constraint.try_advance_token_ids(generated_tokens):
            raise ValueError(
                f"Grammar constraint violated at tokens: {generated_tokens}"
            )

        acceptance = self.grammar_constraint.filter_vocab()

        scores = scores.clone()
        xgrammar.apply_token_bitmask_inplace(
            scores,
            acceptance.to(scores.device, non_blocking=True),
        )

        # Mask tokens beyond the grammar tokenizer's vocab.
        scores[0, self.grammar_constraint.ll_tokenizer.vocab_size:] = float('-inf')

        return scores

    def _get_generated_tokens(self, input_ids: torch.LongTensor) -> torch.Tensor:
        assert input_ids.shape[0] == 1, "Batch size must be 1"
        return input_ids[0, self.prompt_length:]

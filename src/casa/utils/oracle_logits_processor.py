import torch
import xgrammar
from typing import Optional
from transformers.generation.logits_process import LogitsProcessor

from casa.utils.oracle_trie import Trie


class OracleLogitsProcessor(LogitsProcessor):
    """Grammar-constrained logits processor backed by an oracle trie.

    The trie caches per-prefix model logprobs and learned constraint masks
    (log_theta) across attempts, enabling adaptive rejection sampling. The
    behaviour is controlled by learn_level (how much to learn from rejections)
    and constrain_first (whether to constrain the very first token).
    """

    def __init__(
        self,
        tokenizer,
        grammar_constraint,
        device: torch.device,
        learn_level: int = 3,
        constrain_first: bool = False,
        temperature: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.grammar_constraint = grammar_constraint
        self.learn_level = learn_level
        self.constrain_first = constrain_first
        self.temperature = temperature
        self.device = device

        self.oracle_trie = Trie()
        self.reset()

    def reset(self) -> None:
        """Reset parser and trie cursor for a new generation."""
        self.grammar_constraint.reset()
        self.generate_start_index: Optional[int] = None
        self.generated_tokens: Optional[torch.Tensor] = None
        self.oracle_node = self.oracle_trie.root
        self.oracle_node_depth = 0
        self.recompute_needed = False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        self._set_generated_tokens(input_ids)
        is_root = len(self.generated_tokens) == 0

        scores = scores / self.temperature if self.temperature != 1.0 else scores.clone()
        vocab_size = self.grammar_constraint.ll_tokenizer.vocab_size
        scores[0, vocab_size:] = float('-inf')

        # Advance the parser (skipped at level 1, which samples fully unconstrained).
        if self.learn_level != 1:
            if not self.grammar_constraint.try_advance_token_ids(self.generated_tokens):
                self._generation_failed()

        # Descend into the trie node for the last token, creating it if needed.
        if not is_root:
            assert len(self.generated_tokens) == self.oracle_node_depth + 1
            last_token = self.generated_tokens[-1].item()
            if last_token not in self.oracle_node.children:
                self.oracle_node.create_child(last_token)
            self.oracle_node = self.oracle_node.children[last_token]
            self.oracle_node_depth += 1

        # First visit to this node: cache its logprobs and constraint mask.
        if self.oracle_node.raw_logprob is None:
            self.oracle_node.raw_logprob = torch.log_softmax(scores, dim=-1).cpu()
            self.oracle_node.log_theta = torch.zeros(1, scores.size(1))

            adjust_scores = is_root and self.constrain_first
            if self.learn_level >= 3 or adjust_scores:
                acceptance = self.grammar_constraint.filter_vocab()
                xgrammar.apply_token_bitmask_inplace(self.oracle_node.log_theta, acceptance)
                self.recompute_needed = True
        else:
            adjust_scores = True

        if adjust_scores:
            scores += self.oracle_node.log_theta.to(self.device, non_blocking=True)

        if not torch.isfinite(scores[0, :vocab_size]).any():
            raise ValueError(f"No valid continuation at tokens: {self.generated_tokens}")

        return scores

    def _set_generated_tokens(self, input_ids: torch.LongTensor) -> None:
        assert len(input_ids) == 1, "Batch size must be 1"

        if self.generate_start_index is None:
            self.generate_start_index = input_ids.size(1)

        self.generated_tokens = input_ids[0, self.generate_start_index:]

    def _generation_failed(self) -> None:
        """Mark the offending token impossible in the trie, then raise."""
        assert len(self.generated_tokens) == self.oracle_node_depth + 1
        if self.learn_level >= 1:
            self.oracle_node.log_theta[0, self.generated_tokens[-1]] = -float('inf')
            self._recompute_in_trie()

        raise ValueError(f"Generation failed at tokens: {self.generated_tokens}")

    def _recompute_in_trie(self) -> None:
        """Propagate updated log_theta values up the trie to the root."""
        node = self.oracle_node
        depth = self.oracle_node_depth

        while depth > 0:
            new_log_theta = torch.log(
                torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum()
            )
            depth -= 1
            node = node.parent
            node.log_theta[0, self.generated_tokens[depth]] = new_log_theta

    def generation_ended(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Finalize a generation and return its log probability.

        Raises ValueError if the sequence violates the grammar or stops without
        reaching an accepting state.
        """
        self._set_generated_tokens(input_ids)
        assert len(self.generated_tokens) == self.oracle_node_depth + 1

        if not self.grammar_constraint.try_advance_token_ids(self.generated_tokens):
            self._generation_failed()

        # If it didn't end on EOS, the parser must still be in an accepting state.
        if self.generated_tokens[-1] != self.tokenizer.eos_token_id:
            if not self.grammar_constraint.ll_matcher.is_accepting():
                self._generation_failed()

        if self.recompute_needed:
            self._recompute_in_trie()

        return self.get_logprob()

    def remove_generated(self) -> None:
        """Exclude the most recent generation from all future proposals."""
        assert len(self.generated_tokens) == self.oracle_node_depth + 1
        assert self.oracle_node.log_theta is not None
        self.oracle_node.log_theta[0, self.generated_tokens[-1]] = -float('inf')
        self._recompute_in_trie()

    def get_logprob(self) -> torch.Tensor:
        """Sum the cached raw logprobs along the path to the current node."""
        assert len(self.generated_tokens) == self.oracle_node_depth + 1

        logprobs = []
        node = self.oracle_node
        depth = self.oracle_node_depth

        while depth >= 0:
            logprobs.append(node.raw_logprob[0, self.generated_tokens[depth]])
            depth -= 1
            node = node.parent

        return torch.tensor(logprobs).flip(0).sum()

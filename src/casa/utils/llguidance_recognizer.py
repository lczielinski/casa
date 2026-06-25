import os
import llguidance
import llguidance.hf
import llguidance.torch


class LlguidanceTokenRecognizer:
    """Tracks a grammar parse over a token stream via llguidance."""

    def __init__(self, grammar_str: str, tokenizer):
        ll_grammar = llguidance.grammar_from("grammar", grammar_str)
        self.ll_tokenizer = llguidance.hf.from_tokenizer(tokenizer)

        limits = llguidance.LLParserLimits(
            max_items_in_row=int(os.environ.get("CASA_MAX_ITEMS_IN_ROW", "200000")),
        )

        err = llguidance.LLMatcher.validate_grammar(
            ll_grammar, self.ll_tokenizer, limits=limits
        )
        if err:
            raise ValueError(f"Grammar error: {err}")

        log_level = int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1"))
        self.ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            ll_grammar,
            log_level=log_level,
            limits=limits,
        )

        self.current_index = 0
        self._grammar_bitmask = llguidance.torch.allocate_token_bitmask(
            1,
            self.ll_tokenizer.vocab_size,
        )

    def reset(self) -> None:
        self.ll_matcher.reset()
        self.current_index = 0

    def try_advance_token_ids(self, token_ids) -> bool:
        """Consume tokens not yet seen; return True if all were accepted."""
        new_tokens = token_ids[self.current_index:].tolist()
        consumed = self.ll_matcher.try_consume_tokens(new_tokens)

        # A lone EOS in an accepting state counts as consumed.
        if (consumed == 0 and
            len(new_tokens) == 1 and
            new_tokens[0] == self.ll_tokenizer.eos_token and
            self.ll_matcher.is_accepting()):
            consumed = 1

        self.current_index += consumed
        return consumed == len(new_tokens)

    def filter_vocab(self):
        """Return the bitmask of tokens valid at the current position."""
        llguidance.torch.fill_next_token_bitmask(
            self.ll_matcher,
            self._grammar_bitmask,
            0,
        )
        return self._grammar_bitmask

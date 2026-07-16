import xgrammar


class XGrammarTokenRecognizer:
    """Token recognizer using xgrammar for grammar constraints.

    Attributes:
        tokenizer_info: xgrammar tokenizer wrapper.
        matcher: xgrammar matcher for grammar validation.
        current_index: Current token position in sequence.
    """

    def __init__(self, grammar_str: str, tokenizer):
        """Initialize recognizer.

        Args:
            grammar_str: Grammar specification in GBNF/EBNF format.
            tokenizer: HuggingFace tokenizer.

        Raises:
            ValueError: If the tokenizer's EOS is not a registered stop token.
            RuntimeError: If the grammar is invalid.
        """
        self.tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer)
        self.vocab_size = self.tokenizer_info.vocab_size

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and eos_token_id not in self.tokenizer_info.stop_token_ids:
            raise ValueError(
                f"xgrammar engine: tokenizer eos_token_id={eos_token_id} is not a registered "
                f"stop token (stop_token_ids={list(self.tokenizer_info.stop_token_ids)}); "
                f"grammar termination on EOS would fail. Use engine='llguidance' instead."
            )

        compiler = xgrammar.GrammarCompiler(self.tokenizer_info)
        self.compiled_grammar = compiler.compile_grammar(
            xgrammar.Grammar.from_ebnf(grammar_str)
        )
        self.matcher = xgrammar.GrammarMatcher(self.compiled_grammar)

        self.current_index = 0
        self._grammar_bitmask = xgrammar.allocate_token_bitmask(1, self.vocab_size)

    def reset(self) -> None:
        """Reset matcher state."""
        self.matcher.reset()
        self.current_index = 0

    def try_advance_token_ids(self, token_ids) -> bool:
        """Try to advance parser with new tokens.

        Args:
            token_ids: Token IDs to consume.

        Returns:
            True if all tokens were successfully consumed.
        """
        new_tokens = token_ids[self.current_index:].tolist()
        consumed = 0
        for tok in new_tokens:
            if self.matcher.accept_token(tok):
                consumed += 1
            else:
                break
        self.current_index += consumed
        return consumed == len(new_tokens)

    def is_accepting(self) -> bool:
        """Whether the grammar is currently in an accepting state."""
        return self.matcher.is_completed()

    def filter_vocab(self):
        """Get bitmask of valid tokens at current position.

        Returns:
            Token bitmask tensor.
        """
        self.matcher.fill_next_token_bitmask(self._grammar_bitmask, 0)
        return self._grammar_bitmask

    def apply_token_bitmask(self, logits, bitmask) -> None:
        """Apply a next-token bitmask to logits in place."""
        xgrammar.apply_token_bitmask_inplace(logits, bitmask)

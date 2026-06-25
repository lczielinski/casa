import os
import llguidance
import llguidance.hf
import llguidance.torch


class LlguidanceTokenRecognizer:
    """Token recognizer using llguidance for grammar constraints.
    
    Attributes:
        ll_tokenizer: llguidance tokenizer wrapper.
        ll_matcher: llguidance matcher for grammar validation.
        current_index: Current token position in sequence.
    """
    
    def __init__(self, grammar_str: str, tokenizer):
        """Initialize recognizer.
        
        Args:
            grammar_str: Grammar specification string.
            tokenizer: HuggingFace tokenizer.
            
        Raises:
            ValueError: If grammar is invalid.
        """
        ll_grammar = llguidance.grammar_from("grammar", grammar_str)
        self.ll_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
        
        # Validate grammar
        err = llguidance.LLMatcher.validate_grammar(ll_grammar, self.ll_tokenizer)
        if err:
            raise ValueError(f"Grammar error: {err}")
        
        # Create matcher
        log_level = int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1"))
        self.ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            ll_grammar,
            log_level=log_level,
        )
        
        self.current_index = 0
        # Number of real tokens (used for masking padding/phantom ids and bitmasks).
        self.vocab_size = self.ll_tokenizer.vocab_size
        self._grammar_bitmask = llguidance.torch.allocate_token_bitmask(
            1,
            self.vocab_size,
        )

    def is_accepting(self) -> bool:
        """Whether the grammar is currently in an accepting (complete) state."""
        return self.ll_matcher.is_accepting()

    def apply_token_bitmask(self, logits, bitmask) -> None:
        """Apply a next-token bitmask to ``logits`` in place (llguidance backend)."""
        llguidance.torch.apply_token_bitmask_inplace(logits, bitmask)

    def reset(self) -> None:
        """Reset matcher state."""
        self.ll_matcher.reset()
        self.current_index = 0
    
    def try_advance_token_ids(self, token_ids) -> bool:
        """Try to advance parser with new tokens.
        
        Args:
            token_ids: Token IDs to consume.
            
        Returns:
            True if all tokens were successfully consumed.
        """
        new_tokens = token_ids[self.current_index:].tolist()
        consumed = self.ll_matcher.try_consume_tokens(new_tokens)
        
        # Handle EOS token special case
        if (consumed == 0 and 
            len(new_tokens) == 1 and 
            new_tokens[0] == self.ll_tokenizer.eos_token and 
            self.ll_matcher.is_accepting()):
            consumed = 1
        
        self.current_index += consumed
        return consumed == len(new_tokens)
    
    def filter_vocab(self):
        """Get bitmask of valid tokens at current position.
        
        Returns:
            Token bitmask tensor.
        """
        llguidance.torch.fill_next_token_bitmask(
            self.ll_matcher,
            self._grammar_bitmask,
            0,
        )
        return self._grammar_bitmask
from transformers import PreTrainedTokenizer

from casa.utils.llguidance_recognizer import LlguidanceTokenRecognizer


class Grammar:
    """A grammar constraint backed by an llguidance recognizer."""

    def __init__(self, grammar_str: str, tokenizer: PreTrainedTokenizer):
        self.grammar_str = grammar_str
        self.recognizer = LlguidanceTokenRecognizer(grammar_str, tokenizer)

    @classmethod
    def from_file(cls, path: str, tokenizer: PreTrainedTokenizer) -> 'Grammar':
        with open(path, 'r') as f:
            grammar_str = f.read()
        return cls(grammar_str, tokenizer)

    @classmethod
    def from_string(cls, grammar_str: str, tokenizer: PreTrainedTokenizer) -> 'Grammar':
        return cls(grammar_str, tokenizer)

    def reset(self) -> None:
        self.recognizer.reset()

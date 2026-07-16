from transformers import PreTrainedTokenizer

from casa.utils.llguidance_recognizer import LlguidanceTokenRecognizer
from casa.utils.xgrammar_recognizer import XGrammarTokenRecognizer


ENGINES = ("llguidance", "xgrammar")


def _build_recognizer(engine: str, grammar_str: str, tokenizer):
    if engine == "llguidance":
        return LlguidanceTokenRecognizer(grammar_str, tokenizer)
    if engine == "xgrammar":
        return XGrammarTokenRecognizer(grammar_str, tokenizer)
    raise ValueError(f"Unknown grammar engine {engine!r}; choose one of {ENGINES}")


class Grammar:
    """Grammar constraint for structured generation.

    Attributes:
        grammar_str: Grammar specification string.
        engine: Grammar backend ("llguidance" or "xgrammar").
        recognizer: Underlying grammar recognizer.
    """

    def __init__(self, grammar_str: str, tokenizer: PreTrainedTokenizer,
                 engine: str = "llguidance"):
        """Initialize grammar constraint.

        Args:
            grammar_str: Grammar specification in EBNF or Lark format.
            tokenizer: Tokenizer associated with the language model.
            engine: Grammar backend, "llguidance" or "xgrammar".
        """
        self.grammar_str = grammar_str
        self.engine = engine
        self.recognizer = _build_recognizer(engine, grammar_str, tokenizer)

    @classmethod
    def from_file(cls, path: str, tokenizer: PreTrainedTokenizer,
                  engine: str = "llguidance") -> 'Grammar':
        """Load grammar from file.

        Args:
            path: Path to grammar file (.ebnf or .lark).
            tokenizer: Tokenizer associated with the language model.
            engine: Grammar backend, "llguidance" or "xgrammar".

        Returns:
            Grammar instance.
        """
        with open(path, 'r') as f:
            grammar_str = f.read()
        return cls(grammar_str, tokenizer, engine=engine)

    @classmethod
    def from_string(cls, grammar_str: str, tokenizer: PreTrainedTokenizer,
                    engine: str = "llguidance") -> 'Grammar':
        """Create grammar from string.

        Args:
            grammar_str: Grammar specification string.
            tokenizer: Tokenizer associated with the language model.
            engine: Grammar backend, "llguidance" or "xgrammar".

        Returns:
            Grammar instance.
        """
        return cls(grammar_str, tokenizer, engine=engine)

    def reset(self) -> None:
        """Reset the grammar recognizer state."""
        self.recognizer.reset()

from abc import ABC, abstractmethod
import regex as re


class PreTokenizer(ABC):
    """Abstract base class defining the pre-tokenization interface."""

    @abstractmethod
    def pre_tokenize(self, text: str) -> list[str]:
        """
        Splits input text into pre-tokenized string pieces.

        Args:
            text: Input text string to be split.

        Returns:
            List of pre-tokenized string pieces.
        """
        pass


class RegexPreTokenizer(PreTokenizer):
    """Pre-tokenizer using regular expressions to split text into word-like chunks."""

    GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    QWEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    def __init__(self, pattern: str | None = None):
        """
        Initializes RegexPreTokenizer with a custom or default pattern.

        Args:
            pattern: Custom regex pattern string. Defaults to GPT2_PATTERN if None.
        """
        selected_pattern = pattern if pattern is not None else self.GPT2_PATTERN
        self.pattern = re.compile(selected_pattern)

    def pre_tokenize(self, text: str) -> list[str]:
        """
        Pre-tokenizes text based on the configured regex pattern.

        Args:
            text: Input string to be pre-tokenized.

        Returns:
            List of string pieces matching the regex pattern.
        """
        return self.pattern.findall(text)


class WhitespacePreTokenizer(PreTokenizer):
    """Simple pre-tokenizer that splits text on whitespace."""

    def pre_tokenize(self, text: str) -> list[str]:
        """
        Splits text by whitespace characters.

        Args:
            text: Input string to split.

        Returns:
            List of whitespace-separated substring pieces.
        """
        return text.split()
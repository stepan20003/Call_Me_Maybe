"""Custom Byte-Level BPE Tokenizer implementation for LLM models."""

import json
import re
from typing import Dict, List, Optional, Set, Tuple


def bytes_to_unicode() -> Dict[int, str]:
    """Map ASCII and Unicode bytes to visual representation strings.

    Returns:
        Dict[int, str]: A mapping from byte integers to unicode characters.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs_str = [chr(x) for x in cs]
    return dict(zip(bs, cs_str))


class CustomTokenizer:
    """Custom BPE Tokenizer for encoding and decoding text without external LLM libs."""

    def __init__(self, vocab_file_path: str, merges_file_path: Optional[str] = None) -> None:
        """Initialize the tokenizer from a vocab or tokenizer.json file.

        Args:
            vocab_file_path: Path to the vocabulary JSON file provided by the SDK.
        """
        self.byte_encoder: Dict[int, str] = bytes_to_unicode()
        self.byte_decoder: Dict[str, int] = {
            v: k for k, v in self.byte_encoder.items()
        }
        
        self.vocab: Dict[str, int] = {}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

        with open(vocab_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ստուգում ենք՝ արդյոք data-ն HuggingFace tokenizer.json է
        if isinstance(data, dict) and "model" in data and isinstance(data["model"], dict):
            if "vocab" in data["model"] and isinstance(data["model"]["vocab"], dict):
                self.vocab = data["model"]["vocab"]
                merges = data["model"].get("merges", [])
                for idx, merge_str in enumerate(merges):
                    if isinstance(merge_str, str):
                        parts = tuple(merge_str.split())
                        if len(parts) == 2:
                            self.bpe_ranks[(parts[0], parts[1])] = idx
        # Եթե data-ն ուղղակի {"token_str": id} կամ {id: "token_str"} բառարան է
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, int):
                    self.vocab[k] = v
                elif isinstance(v, str):
                    self.vocab[v] = int(k)

        self.index_to_token: Dict[int, str] = {
            v: k for k, v in self.vocab.items()
        }

        # Regex pre-tokenization pattern
        # Standard Python re compatible pattern for Unicode pre-tokenization
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\s\w\d]+|\s+(?!\S)|\s+| ?\w+| ?\d+""",
            re.UNICODE,
        )

    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        """Extract adjacent pairs of symbols from a word.

        Args:
            word: List of string symbols.

        Returns:
            Set[Tuple[str, str]]: Set of symbol pairs.
        """
        pairs: Set[Tuple[str, str]] = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs

    def _bpe(self, token: str) -> List[str]:
        """Apply BPE merge operations to a single token string.

        Args:
            token: Raw string token.

        Returns:
            List[str]: List of merged subword tokens.
        """
        token_bytes = [self.byte_encoder[b] for b in token.encode("utf-8")]
        word = token_bytes
        pairs = self._get_pairs(word)

        if not pairs:
            return word

        while True:
            # Գտնում ենք ամենափոքր rank (ամենաբարձր առաջնահերթություն) ունեցող զույգը
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float("inf")),
            )
            if bigram not in self.bpe_ranks:
                break

            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            pairs = self._get_pairs(word)
            if not pairs:
                break

        return word

    def encode(self, text: str) -> List[int]:
        """Encode string text into a list of token IDs."""
        if not text:
            return []

        bpe_tokens: List[int] = []
        matches = self.pat.findall(text)
        for match in matches:
            for bpe_token in self._bpe(match):
                if bpe_token in self.vocab:
                    bpe_tokens.append(self.vocab[bpe_token])

        return bpe_tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into a UTF-8 string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            str: Decoded UTF-8 string.
        """
        text = "".join([self.index_to_token.get(tid, "") for tid in token_ids])
        byte_list = [
            self.byte_decoder[c] if c in self.byte_decoder else ord(c)
            for c in text
        ]
        return bytes(byte_list).decode("utf-8", errors="replace")
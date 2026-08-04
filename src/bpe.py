"""Byte-Pair Encoding (BPE) merger engine with bounded LRU caching."""

from functools import lru_cache


class BPE:
    """Byte-Pair Encoding (BPE) merger engine."""

    def __init__(self, merges: list[tuple[str, str]]):
        """Initializes the BPE engine with merge rank rules."""
        self.bpe_ranks: dict[tuple[str, str], int] = {
            pair: i for i, pair in enumerate(merges)
        }

    @staticmethod
    def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
        """Extracts adjacent character pairs from a word representation."""
        if len(word) < 2:
            return set()

        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    @lru_cache(maxsize=50000)
    def encode_piece(self, piece: str) -> list[str]:
        """Applies iterative BPE merge rules with bounded LRU caching."""
        word = tuple(piece)
        pairs = self.get_pairs(word)

        if not pairs:
            return [piece]

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word: list[str] = []
            i = 0

            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == first
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)

        return list(word)
# bpe.py
class BPE:
    """Byte-Pair Encoding (BPE) merger engine."""

    def __init__(self, merges: list[tuple[str, str]]):
        """
        Initializes the BPE engine with merge rank rules.

        Args:
            merges: Ordered list of symbol pairs defining BPE merge operations.
        """
        self.bpe_ranks: dict[tuple[str, str], int] = {
            pair: i for i, pair in enumerate(merges)
        }
        self.cache: dict[str, list[str]] = {}

    @staticmethod
    def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
        """
        Extracts adjacent character pairs from a word representation.

        Args:
            word: Tuple of symbols representing a word piece.

        Returns:
            Set of consecutive symbol tuples.
        """
        if len(word) < 2:
            return set()

        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def encode_piece(self, piece: str) -> list[str]:
        """
        Applies iterative BPE merge rules to a single byte-encoded string piece.

        Args:
            piece: Byte-encoded text piece.

        Returns:
            List of merged BPE sub-word tokens.
        """
        if piece in self.cache:
            return self.cache[piece]

        word = tuple(piece)
        pairs = self.get_pairs(word)

        if not pairs:
            return [piece]

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
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
            else:
                pairs = self.get_pairs(word)

        result = list(word)
        self.cache[piece] = result
        return result
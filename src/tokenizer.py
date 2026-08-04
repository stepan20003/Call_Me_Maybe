# tokenizer.py
import json
from .unicode_map import bytes_to_unicode
from .pretokenizer import PreTokenizer, RegexPreTokenizer
from .bpe import BPE


class ByteLevelBPETokenizer:
    """Byte-Level BPE Tokenizer compatible with Hugging Face models using vocab.json and merges.txt."""

    def __init__(
        self,
        vocab_path: str,
        merges_path: str,
        pre_tokenizer: PreTokenizer | None = None,
        unk_token: str | None = None,
    ):
        """
        Initializes the ByteLevelBPETokenizer instance.

        Args:
            vocab_path: Path to JSON file containing token-to-ID mappings.
            merges_path: Path to text file containing BPE merge rules.
            pre_tokenizer: Custom PreTokenizer strategy. Defaults to RegexPreTokenizer.
            unk_token: Fallback token representation for unknown tokens.
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id: dict[str, int] = json.load(f)

        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        self.bpe = BPE(merges)
        self.pre_tokenizer = pre_tokenizer or RegexPreTokenizer()
        self.unk_token = unk_token

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.byte_piece_cache: dict[str, str] = {}
        self.unk_token_id = self.token_to_id.get(unk_token) if unk_token else None

    def _piece_to_byte_string(self, piece: str) -> str:
        """
        Encodes a string piece into a byte-level representation mapping using byte_encoder.

        Args:
            piece: Substring piece to be byte-encoded.

        Returns:
            Mapped byte-level unicode string representation.
        """
        if piece not in self.byte_piece_cache:
            self.byte_piece_cache[piece] = "".join(
                self.byte_encoder[b] for b in piece.encode("utf-8")
            )
        return self.byte_piece_cache[piece]

    def encode(self, text: str) -> list[int]:
        """
        Encodes raw text string into a list of integer token IDs.

        Args:
            text: Raw input text string.

        Returns:
            List of integer token IDs.

        Raises:
            KeyError: If a generated token is not present in the vocabulary and no unk_token is provided.
        """
        ids: list[int] = []
        pieces = self.pre_tokenizer.pre_tokenize(text)

        for piece in pieces:
            byte_piece = self._piece_to_byte_string(piece)
            bpe_tokens = self.bpe.encode_piece(byte_piece)

            for token in bpe_tokens:
                token_id = self.token_to_id.get(token)
                if token_id is not None:
                    ids.append(token_id)
                elif self.unk_token_id is not None:
                    ids.append(self.unk_token_id)
                else:
                    raise KeyError(f"Token '{token}' not found in vocabulary.")

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a reconstructed text string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded UTF-8 text string.

        Raises:
            KeyError: If a token ID is not recognized or mapped byte cannot be decoded.
        """
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i)
            if token is not None:
                tokens.append(token)
            else:
                raise KeyError(f"ID {i} not found in vocabulary.")

        text_encoded = "".join(tokens)

        try:
            raw_bytes = bytes(self.byte_decoder[c] for c in text_encoded)
        except KeyError as e:
            raise KeyError(f"Character {e} not found in byte decoder.") from e

        return raw_bytes.decode("utf-8", errors="replace")
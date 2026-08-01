"""Constrained decoding implementation for structured JSON generation."""

import json
from typing import List, Dict, Set
import numpy as np
from src.models import FunctionDefinition
from src.tokenizer import CustomTokenizer


class JSONConstraintDecoder:
    """Fast dynamic constraint decoder for function calling."""

    def __init__(self, vocab: Dict[str, int],
                 functions_def: List[FunctionDefinition],
                 tokenizer: CustomTokenizer):
        """Initialize with vocabulary mapping, schemas, and indexed token pools."""
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.functions_def = functions_def
        self.valid_names = {fn.name for fn in functions_def}

        # 1. Pre-decode tokens into cache
        self.decoded_tokens: Dict[int, str] = {}
        for token_str, token_id in vocab.items():
            self.decoded_tokens[token_id] = self.tokenizer.decode([token_id])

        # 2. Filter candidate token IDs (only valid printable text & JSON syntax)
        self.valid_candidate_ids: List[int] = []
        for token_id, clean_str in self.decoded_tokens.items():
            if not clean_str:
                continue
            if any(c.isalnum() or c in '{}"[]:,. -_\n\t' for c in clean_str):
                self.valid_candidate_ids.append(token_id)

    def mask_logits(self, logits: List[float],
                    generated_text: str) -> List[float]:
        """Filter out tokens that violate JSON syntax or function schemas."""
        valid_token_ids = self.get_valid_token_ids(generated_text)
        if not valid_token_ids:
            return logits

        masked = [-float("inf")] * len(logits)
        for token_id in valid_token_ids:
            if token_id < len(logits):
                masked[token_id] = logits[token_id]

        return masked

    def get_valid_token_ids(self, current_text: str) -> Set[int]:
        """Determine allowed next token IDs quickly."""
        if not current_text.strip():
            return {tid for tid in self.valid_candidate_ids if "{" in self.decoded_tokens[tid]}

        valid_ids: Set[int] = set()

        # Fast structure checks instead of thousands of json.loads
        for token_id in self.valid_candidate_ids:
            clean_str = self.decoded_tokens[token_id]
            candidate = current_text + clean_str

            # Fast bracket balance check
            if candidate.count("{") < candidate.count("}"):
                continue

            # If function name is being generated, validate against valid_names
            if '"name"' in candidate and '"parameters"' not in candidate:
                if any(name in candidate for name in self.valid_names) or any(c in clean_str for c in '":, {}\n\t'):
                    valid_ids.add(token_id)
            else:
                valid_ids.add(token_id)

        return valid_ids if valid_ids else set(self.valid_candidate_ids)
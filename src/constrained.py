"""Constrained decoding implementation for structured JSON generation with schema type enforcement."""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.models import FunctionDefinition
from src.tokenizer import ByteLevelBPETokenizer


class JSONConstraintDecoder:
    """Schema-level JSON constraint decoder with robust state extraction and type checks."""

    NUMBER_PATTERN = re.compile(r"^-?\d*\.?\d*$")

    def __init__(
        self,
        vocab: Dict[str, int],
        functions_def: List[FunctionDefinition],
        tokenizer: ByteLevelBPETokenizer,
        top_k: int = 100,
    ) -> None:
        """Initialize vocabulary, function schemas, and token decoding maps."""
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.functions_def = {fn.name: fn for fn in functions_def}
        self.valid_names = set(self.functions_def.keys())
        self.top_k = top_k

        # Pre-cache single token IDs to string mappings for O(1) lookup
        self.id_to_str: Dict[int, str] = {}
        for _, token_id in vocab.items():
            try:
                self.id_to_str[token_id] = self.tokenizer.decode([token_id])
            except Exception:
                self.id_to_str[token_id] = ""

        self.valid_cache: Dict[Tuple[str, Tuple[int, ...]], Set[int]] = {}

    def mask_logits(self, logits: List[float], generated_text: str) -> List[float]:
        """Filter logits by evaluating candidates against JSON syntax and schema types."""
        valid_ids = self.get_valid_token_ids(generated_text, logits)

        if not valid_ids:
            return logits

        masked_logits = np.full(len(logits), -np.inf, dtype=np.float32)
        for token_id in valid_ids:
            if token_id < len(masked_logits):
                masked_logits[token_id] = logits[token_id]

        return masked_logits.tolist()  # type: ignore[no-any-return]

    def get_valid_token_ids(self, current_text: str, logits: List[float]) -> Set[int]:
        """Filter candidate tokens using Top-K ranking, caching, and schema state checks."""
        top_k_indices = np.argpartition(logits, -self.top_k)[-self.top_k:]
        top_k_tuple = tuple(int(idx) for idx in top_k_indices)

        cache_key = (current_text, top_k_tuple)

        if cache_key in self.valid_cache:
            return self.valid_cache[cache_key]

        valid_ids: Set[int] = set()

        for token_id in top_k_indices:
            token_str = self.id_to_str.get(int(token_id), "")
            if not token_str:
                continue

            candidate = current_text + token_str

            if self._is_valid_schema_prefix(candidate):
                valid_ids.add(int(token_id))

        self.valid_cache[cache_key] = valid_ids
        return valid_ids

    def _extract_selected_function(self, text: str) -> Optional[str]:
        """Extract function name currently specified in the JSON text."""
        if '"name":' not in text:
            return None
        try:
            after_name = text.split('"name":')[1].strip()
            if after_name.startswith('"'):
                parts = after_name.split('"')
                if len(parts) >= 2:
                    return parts[1]
        except Exception:
            pass
        return None

    def _extract_active_parameter_state(self, text: str) -> Tuple[Optional[str], str]:
        """Extract active parameter key name and its current unparsed value snippet."""
        if '"parameters":' not in text:
            return None, ""

        params_blob = text.split('"parameters":')[1].strip()
        if not params_blob.startswith("{"):
            return None, ""

        # Find the last key-value pair being written
        matches = list(re.finditer(r'"([a-zA-Z0-9_]+)"\s*:\s*', params_blob))
        if not matches:
            return None, ""

        last_match = matches[-1]
        param_name = last_match.group(1)
        value_snippet = params_blob[last_match.end() :].strip()

        return param_name, value_snippet

    def _get_parameter_type_schema(
        self, fn_schema: FunctionDefinition, param_name: str
    ) -> Optional[str]:
        """Extract type string safely regardless of Pydantic model parameter structure."""
        params_def = fn_schema.parameters
        if isinstance(params_def, dict):
            param_info = params_def.get(param_name)
            if isinstance(param_info, dict):
                return str(param_info.get("type", ""))
            elif hasattr(param_info, "type"):
                return str(getattr(param_info, "type"))
        return None

    def _validate_parameter_type(self, val_str: str, expected_type: str) -> bool:
        """Validate if active parameter value snippet satisfies primitive schema constraints."""
        if not val_str:
            return True

        if expected_type == "number":
            # Disallow string quotes
            if val_str.startswith('"'):
                return False
            # Check prefix against number regex
            clean_val = val_str.rstrip(",}").strip()
            if clean_val and not self.NUMBER_PATTERN.match(clean_val):
                return False

        elif expected_type == "string":
            # String value must start with quote once typing has begun
            if not val_str.startswith('"'):
                return False

        elif expected_type == "boolean":
            clean_val = val_str.rstrip(",}").strip()
            if clean_val:
                if not ("true".startswith(clean_val) or "false".startswith(clean_val)):
                    return False

        return True

    def _is_valid_schema_prefix(self, text: str) -> bool:
        """Validate candidate text against JSON syntax and active function parameter types."""
        stripped = text.strip()
        if not stripped:
            return True
        if not stripped.startswith("{"):
            return False

        # 1. Function name validity check
        selected_fn_name = self._extract_selected_function(text)
        if selected_fn_name and selected_fn_name not in self.valid_names:
            # Check if fn_name is a prefix of any valid function name
            if not any(fn.startswith(selected_fn_name) for fn in self.valid_names):
                return False

        # 2. Parameter Type Enforcement
        if selected_fn_name and selected_fn_name in self.functions_def:
            fn_schema = self.functions_def[selected_fn_name]
            param_name, val_snippet = self._extract_active_parameter_state(text)

            if param_name:
                expected_type = self._get_parameter_type_schema(fn_schema, param_name)
                if expected_type and not self._validate_parameter_type(val_snippet, expected_type):
                    return False

        # 3. Overall JSON syntax prefix validation
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError as e:
            msg = str(e)
            if "Unterminated string" in msg or "Expecting" in msg or "end of file" in msg:
                return True
            return False
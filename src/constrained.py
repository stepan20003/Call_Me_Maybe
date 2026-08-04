"""Constrained decoding implementation for structured JSON generation with state machine, cache, and schema-level type enforcement."""

import json
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.models import FunctionDefinition
from src.tokenizer import ByteLevelBPETokenizer


class JSONStateValidator:
    """Pushdown Automaton State Validator for JSON syntax and primitive schema types."""

    NUMBER_PREFIX = re.compile(r"^-?\d*\.?\d*$")

    @staticmethod
    def extract_function(text: str) -> Optional[str]:
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

    @staticmethod
    def get_active_parameter(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract the last active parameter key name and its current unparsed value snippet."""
        if '"parameters":' not in text:
            return None, None

        params_blob = text.split('"parameters":', 1)[1]
        matches = list(re.finditer(r'"([^"]+)"\s*:\s*', params_blob))

        if not matches:
            return None, None

        last_match = matches[-1]
        param_name = last_match.group(1)
        value_snippet = params_blob[last_match.end() :].strip()

        return param_name, value_snippet

    @staticmethod
    def validate_parameter(value: str, expected_type: str) -> bool:
        """Validate if active parameter value snippet satisfies primitive schema constraints."""
        val = value.strip()
        if not val:
            return True

        if expected_type == "number":
            if val.startswith('"'):
                return False
            val_clean = val.rstrip(",}").strip()
            return bool(JSONStateValidator.NUMBER_PREFIX.match(val_clean))

        if expected_type == "string":
            return val.startswith('"')

        if expected_type == "boolean":
            val_clean = val.rstrip(",}").strip()
            return "true".startswith(val_clean) or "false".startswith(val_clean)

        return True

    @staticmethod
    def validate_prefix(
        text: str, functions: Dict[str, FunctionDefinition]
    ) -> bool:
        """Validate candidate text against JSON syntax and active function parameter types."""
        stripped = text.strip()
        if not stripped:
            return True
        if not stripped.startswith("{"):
            return False

        # 1. Structural Bracket Balance & String Escape Check
        in_string = False
        escape = False
        stack: list[str] = []

        for char in text:
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack:
                    return False
                top = stack.pop()
                if (char == "}" and top != "{") or (char == "]" and top != "["):
                    return False

        # 2. Function Name Constraint Validation
        valid_fn_names = set(functions.keys())
        fn_name = JSONStateValidator.extract_function(text)

        if fn_name:
            if fn_name not in valid_fn_names:
                if not any(f.startswith(fn_name) for f in valid_fn_names):
                    return False

        # 3. Parameter Schema Type Validation (Number / String / Boolean)
        if fn_name and fn_name in functions:
            param_name, val_snippet = JSONStateValidator.get_active_parameter(
                text
            )
            if param_name and val_snippet is not None:
                schema = functions[fn_name]
                if (
                    schema.parameters
                    and param_name in schema.parameters
                ):
                    param_type = schema.parameters[param_name].type
                    if not JSONStateValidator.validate_parameter(
                        val_snippet, param_type
                    ):
                        return False

        # 4. JSON Syntax Prefix Check
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError as e:
            msg = str(e)
            return any(
                term in msg
                for term in ["Unterminated string", "Expecting", "end of file"]
            )


class JSONConstraintDecoder:
    """Fast Top-K schema-aware JSON constraint decoder with state tracking and caching."""

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
        self.top_k = top_k

        self.id_to_str: Dict[int, str] = {}
        for _, token_id in vocab.items():
            try:
                self.id_to_str[token_id] = self.tokenizer.decode([token_id])
            except Exception:
                self.id_to_str[token_id] = ""

        self.valid_cache: Dict[Tuple[str, Tuple[int, ...]], Set[int]] = {}
        self.prefix_cache: Dict[str, bool] = {}

    def validate_prefix_cached(self, candidate: str) -> bool:
        """Check and cache candidate text validity to optimize PDA character scanning."""
        if candidate in self.prefix_cache:
            return self.prefix_cache[candidate]

        is_valid = JSONStateValidator.validate_prefix(
            candidate, self.functions_def
        )
        self.prefix_cache[candidate] = is_valid
        return is_valid

    def mask_logits(
        self, logits: List[float], generated_text: str
    ) -> List[float]:
        """Filter logits by evaluating candidate tokens against JSON state machine and parameter schema."""
        valid_ids = self.get_valid_token_ids(generated_text, logits)

        if not valid_ids:
            return logits

        masked_logits = np.full(len(logits), -np.inf, dtype=np.float32)
        for token_id in valid_ids:
            if token_id < len(masked_logits):
                masked_logits[token_id] = logits[token_id]

        return masked_logits.tolist()  # type: ignore[no-any-return]

    def get_valid_token_ids(
        self, current_text: str, logits: List[float]
    ) -> Set[int]:
        """Filter candidate tokens using Top-K ranking, state caching, and schema checks."""
        top_k_indices = np.argpartition(logits, -self.top_k)[-self.top_k :]
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

            if self.validate_prefix_cached(candidate):
                valid_ids.add(int(token_id))

        self.valid_cache[cache_key] = valid_ids
        return valid_ids
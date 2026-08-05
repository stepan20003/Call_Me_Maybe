"""O(1) State-Driven Incremental Pushdown Automaton JSON Decoder with Ultra-Optimizations."""

from enum import Enum, auto
from typing import Dict, List, Optional
import numpy as np

from src.models import FunctionDefinition
from src.tokenizer import ByteLevelBPETokenizer


class ParserMode(Enum):
    EXPECT_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    AFTER_VALUE = auto()
    AFTER_COMMA = auto()


class FnTrieNode:
    __slots__ = ['children']
    def __init__(self) -> None:
        self.children: Dict[str, 'FnTrieNode'] = {}


class FnNameTrie:
    def __init__(self, names: List[str]) -> None:
        self.root = FnTrieNode()
        for name in names:
            node = self.root
            for char in name:
                if char not in node.children:
                    node.children[char] = FnTrieNode()
                node = node.children[char]

    def has_prefix(self, prefix_list: List[str]) -> bool:
        node = self.root
        for char in prefix_list:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


class IncrementalPDAState:
    __slots__ = [
        'mode', 'stack', 'in_string', 'escape', 'buffer',
        'current_key', 'current_val_buffer', 'fn_name',
        'parameters_depth', 'is_invalid'
    ]

    def __init__(self) -> None:
        self.mode: ParserMode = ParserMode.EXPECT_KEY
        self.stack: List[str] = []
        self.in_string: bool = False
        self.escape: bool = False
        self.buffer: List[str] = []
        self.current_key: Optional[str] = None
        self.current_val_buffer: List[str] = []
        self.fn_name: Optional[str] = None
        self.parameters_depth: int = -1
        self.is_invalid: bool = False

    def clone(self) -> "IncrementalPDAState":
        new_state = IncrementalPDAState.__new__(IncrementalPDAState)
        new_state.mode = self.mode
        new_state.stack = self.stack.copy()
        new_state.in_string = self.in_string
        new_state.escape = self.escape
        new_state.buffer = self.buffer.copy()
        new_state.current_key = self.current_key
        new_state.current_val_buffer = self.current_val_buffer.copy()
        new_state.fn_name = self.fn_name
        new_state.parameters_depth = self.parameters_depth
        new_state.is_invalid = self.is_invalid
        return new_state

    def feed_char(self, char: str, functions: Dict[str, FunctionDefinition], fn_trie: FnNameTrie) -> bool:
        if self.is_invalid:
            return False

        if self.escape:
            self.escape = False
            if self.in_string:
                self.buffer.append(char)
                if self.mode == ParserMode.EXPECT_VALUE:
                    self.current_val_buffer.append(char)
            return True

        if char == "\\":
            self.escape = True
            if self.in_string:
                self.buffer.append(char)
                if self.mode == ParserMode.EXPECT_VALUE:
                    self.current_val_buffer.append(char)
            return True

        mode = self.mode
        in_str = self.in_string
        fn_name = self.fn_name
        curr_key = self.current_key

        if char == '"' and mode == ParserMode.EXPECT_VALUE and fn_name and curr_key:
            schema = functions.get(fn_name)
            if schema and schema.parameters and curr_key in schema.parameters:
                if schema.parameters[curr_key].type in ("number", "boolean"):
                    self.is_invalid = True
                    return False

        if char == '"':
            self.in_string = not in_str
            if self.in_string:
                self.buffer.clear()
            else:
                completed_str = "".join(self.buffer)
                self.buffer.clear()
                if mode in (ParserMode.EXPECT_KEY, ParserMode.AFTER_COMMA):
                    self.current_key = completed_str
                    self.mode = ParserMode.EXPECT_COLON
                elif mode == ParserMode.EXPECT_VALUE:
                    if curr_key == "name":
                        self.fn_name = completed_str.strip()
                    self.mode = ParserMode.AFTER_VALUE
            return True

        if self.in_string:
            self.buffer.append(char)
            if mode == ParserMode.EXPECT_KEY and len(self.stack) == 1:
                if len(self.buffer) <= 10:
                    b_str = "".join(self.buffer)
                    if not ("name".startswith(b_str) or "parameters".startswith(b_str)):
                        self.is_invalid = True
                        return False
            if mode == ParserMode.EXPECT_VALUE:
                self.current_val_buffer.append(char)
                if curr_key == "name":
                    if not fn_trie.has_prefix(self.current_val_buffer):
                        self.is_invalid = True
                        return False
            return True

        if char == "{":
            self.stack.append("{")
            if curr_key == "parameters":
                self.parameters_depth = len(self.stack)
            self.mode = ParserMode.EXPECT_KEY
            return True
        elif char == "[":
            self.stack.append("[")
            self.mode = ParserMode.EXPECT_VALUE
            return True
        elif char in "}]":
            if not self.stack:
                self.is_invalid = True
                return False
            top = self.stack.pop()
            if (char == "}" and top != "{") or (char == "]" and top != "["):
                self.is_invalid = True
                return False
            if self.parameters_depth != -1 and len(self.stack) < self.parameters_depth:
                self.parameters_depth = -1
            self.mode = ParserMode.AFTER_VALUE
            self.current_key = None
            self.current_val_buffer.clear()
            return True

        if char == ":":
            if mode == ParserMode.EXPECT_COLON:
                self.mode = ParserMode.EXPECT_VALUE
                self.current_val_buffer.clear()
                return True
            self.is_invalid = True
            return False

        if char == ",":
            if mode == ParserMode.AFTER_VALUE:
                self.mode = ParserMode.AFTER_COMMA
                self.current_key = None
                self.current_val_buffer.clear()
                return True
            self.is_invalid = True
            return False

        if char in " \t\n\r":
            return True

        if mode == ParserMode.EXPECT_VALUE:
            self.current_val_buffer.append(char)
            if fn_name and curr_key:
                schema = functions.get(fn_name)
                if schema and schema.parameters and curr_key in schema.parameters:
                    exp_type = schema.parameters[curr_key].type
                    
                    if exp_type == "number":
                        dot_seen = False
                        decimals = 0
                        for i, c in enumerate(self.current_val_buffer):
                            if c in ",}\n\t ": 
                                continue
                            if c == '-':
                                if i != 0: 
                                    self.is_invalid = True
                                    return False
                            elif c == '.':
                                if dot_seen:
                                    self.is_invalid = True
                                    return False
                                dot_seen = True
                            elif c.isdigit():
                                if dot_seen:
                                    decimals += 1
                                    if decimals > 1:
                                        self.is_invalid = True
                                        return False
                            else:
                                self.is_invalid = True
                                return False
                                
                    elif exp_type == "boolean":
                        b_str = "".join(self.current_val_buffer).rstrip(",} \n\t")
                        if b_str and not ("true".startswith(b_str) or "false".startswith(b_str)):
                            self.is_invalid = True
                            return False
        return True

    def feed_str(self, token_str: str, functions: Dict[str, FunctionDefinition], fn_trie: FnNameTrie) -> bool:
        for char in token_str:
            if not self.feed_char(char, functions, fn_trie):
                return False
        return True


class JSONConstraintDecoder:
    def __init__(
        self,
        vocab: Dict[str, int],
        functions_def: List[FunctionDefinition],
        tokenizer: ByteLevelBPETokenizer,
        top_k: int = 3, # 🚀 Իջեցված է 3-ի մաքսիմալ արագության համար
    ) -> None:
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.functions_def = {fn.name: fn for fn in functions_def}
        self.top_k = top_k
        self.id_to_str: Dict[int, str] = {}
        
        self.fn_trie = FnNameTrie(list(self.functions_def.keys()))
        
        for _, token_id in vocab.items():
            try:
                self.id_to_str[token_id] = self.tokenizer.decode([token_id])
            except Exception:
                self.id_to_str[token_id] = ""

        self.root_state = IncrementalPDAState()

    def reset(self) -> None:
        self.root_state = IncrementalPDAState()

    def advance_base_state(self, new_token_str: str) -> None:
        if not self.root_state.feed_str(new_token_str, self.functions_def, self.fn_trie):
            raise RuntimeError(f"Parser became invalid after token {new_token_str!r}")

    def mask_logits(self, logits: List[float]) -> List[float]:
        masked_logits = np.full(len(logits), -np.inf, dtype=np.float32)
        logits_arr = np.array(logits)
        
        # 🚀 ՕՊՏԻՄԻԶԱՑԻԱ: Վերցնում ենք միայն ամենահավանական 32-ը (նախկին 64-ի փոխարեն)
        k = min(32, len(logits_arr))
        top_k_idx = np.argpartition(logits_arr, -k)[-k:]
        sorted_indices = top_k_idx[np.argsort(logits_arr[top_k_idx])[::-1]]
        
        valid_found = 0
        for token_id in sorted_indices:
            token_str = self.id_to_str.get(token_id, "")
            if not token_str:
                continue
                
            branch_state = self.root_state.clone()
            
            if branch_state.feed_str(token_str, self.functions_def, self.fn_trie):
                masked_logits[token_id] = logits[token_id]
                valid_found += 1
                
                if valid_found >= self.top_k:
                    break
                    
        if valid_found == 0:
            raise RuntimeError("No valid tokens found for current parser state.")
            
        return masked_logits.tolist()
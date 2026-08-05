"""Main entry point for function calling CLI with Safe Parallel Execution."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

from llm_sdk.llm_sdk import Small_LLM_Model
from src.constrained import JSONConstraintDecoder
from src.models import FunctionCallResult, FunctionDefinition, TestCase
from src.pretokenizer import RegexPreTokenizer
from src.tokenizer import ByteLevelBPETokenizer


def load_functions_definition(path_str: str) -> List[FunctionDefinition]:
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [FunctionDefinition.model_validate(item) for item in raw]
    except Exception as e:
        print(f"Error loading functions: {e}", file=sys.stderr)
        sys.exit(1)


def load_test_cases(path_str: str) -> List[TestCase]:
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [TestCase.model_validate(item) for item in raw]
    except Exception as e:
        print(f"Error loading inputs: {e}", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call Me Maybe")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    return parser.parse_args()


def clean_parsed_parameters(parsed: Dict[str, Any], functions_def: List[FunctionDefinition]) -> Dict[str, Any]:
    fn_name = parsed.get("name")
    params = parsed.get("parameters", {})
    if not isinstance(params, dict):
        params = {}

    fn_spec = next((f for f in functions_def if f.name == fn_name), None)
    if not fn_spec or not fn_spec.parameters:
        return params

    cleaned = {}
    for k, param_schema in fn_spec.parameters.items():
        expected_type = param_schema.type
        v = params.get(k)
        if v is None and len(params) == 1:
            v = list(params.values()[0]) if hasattr(list(params.values())[0], '__iter__') else list(params.values())[0]

        if v is not None:
            if expected_type == "number":
                try:
                    cleaned[k] = float(v) if "." in str(v) else int(v)
                except ValueError:
                    cleaned[k] = v
            else:
                cleaned[k] = v
        else:
            if expected_type == "number":
                cleaned[k] = 0.0
    return cleaned


def extract_json_object_brace_counter(text: str) -> Dict[str, Any] | None:
    start = text.find("{")
    if start == -1: return None
    depth, in_string, escape = 0, False, False
    for i in range(start, len(text)):
        char = text[i]
        if escape: escape = False; continue
        if char == "\\" and in_string: escape = True; continue
        if char == '"': in_string = not in_string; continue
        if not in_string:
            if char == "{": depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, dict) and "name" in parsed and "parameters" in parsed:
                            return parsed
                    except json.JSONDecodeError: return None
    return None


def process_single_test(
    test_index: int, 
    test: TestCase, 
    prefix_ids: List[int], 
    functions_def: List[FunctionDefinition], 
    vocab: Dict, 
    tokenizer: ByteLevelBPETokenizer,
    llm: Small_LLM_Model
) -> Dict[str, Any]:
    
    decoder = JSONConstraintDecoder(vocab, functions_def, tokenizer, top_k=10)
    
    test_text = f"User request: {test.prompt}\nJSON:\n{{\"name\":\""
    encoded_test = llm.encode(test_text)
    test_ids = encoded_test.squeeze(0).tolist() if isinstance(encoded_test, torch.Tensor) else list(encoded_test)
    
    input_ids = prefix_ids + test_ids

    pieces = ['{"name":"']
    try:
        decoder.advance_base_state(pieces[0])
    except RuntimeError:
        return None

    brace_depth = 1
    t_llm_total = 0.0
    t_mask_total = 0.0

    for step in range(40):
        t0 = time.perf_counter()
        raw_logits = llm.get_logits_from_input_ids(input_ids)
        t1 = time.perf_counter()
        t_llm_total += (t1 - t0)
        
        t2 = time.perf_counter()
        masked_logits = decoder.mask_logits(raw_logits)
        t3 = time.perf_counter()
        t_mask_total += (t3 - t2)
        
        next_token_id = int(np.argmax(masked_logits))
        next_token_str = decoder.id_to_str.get(next_token_id, "")

        if not next_token_str:
            break

        pieces.append(next_token_str)
        input_ids.append(next_token_id)

        try:
            decoder.advance_base_state(next_token_str)
        except RuntimeError:
            break

        brace_depth += next_token_str.count('{')
        brace_depth -= next_token_str.count('}')
        
        generated_text = "".join(pieces)
        if brace_depth <= 0 and generated_text.strip().endswith("}"):
            break

    print(f"\n[{test_index}] {generated_text}")
    print(f"   ⏱ LLM Time: {t_llm_total:.4f}s | Mask Time: {t_mask_total:.4f}s")

    parsed = extract_json_object_brace_counter(generated_text)
    if parsed is not None:
        cleaned_params = clean_parsed_parameters(parsed, functions_def)
        return {
            "prompt": test.prompt,
            "name": parsed["name"],
            "parameters": cleaned_params,
        }
    return None


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    functions_def = load_functions_definition(args.functions_definition)
    test_cases = load_test_cases(args.input)

    llm = Small_LLM_Model()
    vocab_path = llm.get_path_to_vocab_file()
    merges_path = vocab_path.replace("vocab.json", "merges.txt")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    tokenizer = ByteLevelBPETokenizer(
        vocab_path, merges_path, pre_tokenizer=RegexPreTokenizer(RegexPreTokenizer.QWEN_PATTERN)
    )

    mini_fns = []
    for f in functions_def:
        fn_dict = f.model_dump(exclude_none=True)
        fn_dict.pop("description", None)
        for param in fn_dict.get("parameters", {}).values():
            param.pop("description", None)
        mini_fns.append(fn_dict)
    
    fns_dump = json.dumps(mini_fns, separators=(',', ':'))

    prefix_text = f"Available functions: {fns_dump}\n"
    encoded_prefix = llm.encode(prefix_text)
    prefix_ids = encoded_prefix.squeeze(0).tolist() if isinstance(encoded_prefix, torch.Tensor) else list(encoded_prefix)

    results = [None] * len(test_cases)
    
    # 🚀 Զուգահեռ գործարկում (Max 4 հոսք անվտանգության համար)
    num_workers = min(4, os.cpu_count() or 4)
    print(f"⚡ Զուգահեռ աշխատանք {num_workers} հոսքով...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(
                process_single_test, 
                i + 1, test, prefix_ids, functions_def, vocab, tokenizer, llm
            ): i 
            for i, test in enumerate(test_cases)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                res = future.result()
                if res:
                    results[idx] = res
            except Exception as e:
                print(f"Error in test {idx+1}: {e}", file=sys.stderr)

    results = [r for r in results if r is not None]

    output_path = Path(args.output)
    output_path.output_parent.mkdir(parents=True, exist_ok=True) if hasattr(output_path, 'output_parent') else output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\n⚡ Ավարտվեց {len(test_cases)} թեստ {time.perf_counter() - start_time:.2f} վայրկյանում!")


if __name__ == "__main__":
    main()
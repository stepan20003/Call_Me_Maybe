"""Main entry point for function calling CLI."""

import argparse
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
        return params

    fn_spec = next((f for f in functions_def if f.name == fn_name), None)
    if not fn_spec or not fn_spec.parameters:
        return params

    cleaned = {}
    for k, v in params.items():
        if k in fn_spec.parameters:
            expected_type = fn_spec.parameters[k].type
            if expected_type == "number" and isinstance(v, str):
                try:
                    cleaned[k] = float(v) if "." in v else int(v)
                except ValueError:
                    cleaned[k] = v
            else:
                cleaned[k] = v
        else:
            cleaned[k] = v
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


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    functions_def = load_functions_definition(args.functions_definition)
    test_cases = load_test_cases(args.input)

    # Թողնում ենք անվտանգ քանակի միջուկներ
    safe_cores = max(1, min(3, (os.cpu_count() or 4) - 1))
    torch.set_num_threads(safe_cores)

    llm = Small_LLM_Model()
    vocab_path = llm.get_path_to_vocab_file()
    merges_path = vocab_path.replace("vocab.json", "merges.txt")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    tokenizer = ByteLevelBPETokenizer(
        vocab_path, merges_path, pre_tokenizer=RegexPreTokenizer(RegexPreTokenizer.QWEN_PATTERN)
    )

    decoder = JSONConstraintDecoder(vocab, functions_def, tokenizer, top_k=15)
    results = []

    # Ստեղծում ենք base string ֆունկցիաների համար միայն մեկ անգամ
    mini_fns = []
    for f in functions_def:
        fn_dict = f.model_dump(exclude_none=True)
        fn_dict.pop("description", None)
        for param in fn_dict.get("parameters", {}).values():
            param.pop("description", None)
        mini_fns.append(fn_dict)
    
    fns_dump = json.dumps(mini_fns, separators=(',', ':'))

    for i, test in enumerate(test_cases, 1):
        prompt_text = test.prompt
        
        full_prompt = (
            f"Available functions: {fns_dump}\n"
            f"User request: {prompt_text}\n"
            f"JSON:\n{{\"name\":\""
        )

        encoded = llm.encode(full_prompt)
        input_ids = encoded.squeeze(0).tolist() if isinstance(encoded, torch.Tensor) else list(encoded)

        decoder.reset()
        
        # 🚀 ՕՊՏԻՄԻԶԱՑԻԱ: Հեռացված է ծանր .count() ֆունկցիան
        brace_depth = 1 # Քանի որ սկսում ենք `{"name":"`-ով
        
        # 🚀 ՕՊՏԻՄԻԶԱՑԻԱ: Հավաքում ենք տեքստը Array-ում՝ print ֆունկցիան I/O բլոկավորումից հանելու համար
        pieces = ['{"name":"']
        
        try:
            decoder.advance_base_state(pieces[0])
        except RuntimeError:
            continue

        # Profiling փոփոխականներ
        t_llm_total = 0.0
        t_mask_total = 0.0

        for step in range(40):
            # ⏱ Չափում ենք միայն LLM-ի inference ժամանակը
            t0 = time.perf_counter()
            raw_logits = llm.get_logits_from_input_ids(input_ids)
            t1 = time.perf_counter()
            t_llm_total += (t1 - t0)
            
            # ⏱ Չափում ենք միայն մեր գրած Parser-ի ժամանակը
            t2 = time.perf_counter()
            masked_logits = decoder.mask_logits(raw_logits)
            t3 = time.perf_counter()
            t_mask_total += (t3 - t2)
            
            next_token_id = int(np.argmax(masked_logits))
            next_token_str = decoder.id_to_str.get(next_token_id, "")

            pieces.append(next_token_str)
            input_ids.append(next_token_id)

            try:
                decoder.advance_base_state(next_token_str)
            except RuntimeError:
                break

            # O(L) բարդության փակագծերի հաշվիչ՝ միայն նոր եկած թոքենի վրայով
            brace_depth += next_token_str.count('{')
            brace_depth -= next_token_str.count('}')
            
            if brace_depth <= 0:
                break

        # Վերջում միացնում ենք տողերը և տպում մեկ անգամ
        generated_text = "".join(pieces)
        print(f"\n[{i}/{len(test_cases)}] {generated_text}")
        print(f"   ⏱ LLM Time: {t_llm_total:.4f}s | Mask (Parser) Time: {t_mask_total:.4f}s")

        parsed = extract_json_object_brace_counter(generated_text)
        if parsed is not None:
            cleaned_params = clean_parsed_parameters(parsed, functions_def)
            results.append({
                "prompt": prompt_text,
                "name": parsed["name"],
                "parameters": cleaned_params,
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\n⚡ Ավարտվեց {len(test_cases)} թեստ {time.perf_counter() - start_time:.2f} վայրկյանում!")


if __name__ == "__main__":
    main()
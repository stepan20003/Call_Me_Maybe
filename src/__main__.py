"""Main entry point for function calling CLI."""

import argparse
import json
from pathlib import Path
import re
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
    """Load and validate function definition JSON schema file."""
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            print("Error: Expected a JSON array in functions definition file.", file=sys.stderr)
            sys.exit(1)

        return [FunctionDefinition.model_validate(item) for item in raw]
    except Exception as e:
        print(f"Error loading functions definition from '{path_str}': {e}", file=sys.stderr)
        sys.exit(1)


def load_test_cases(path_str: str) -> List[TestCase]:
    """Load and validate test case prompts JSON file."""
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            print("Error: Expected a JSON array in input file.", file=sys.stderr)
            sys.exit(1)

        return [TestCase.model_validate(item) for item in raw]
    except Exception as e:
        print(f"Error loading input prompts from '{path_str}': {e}", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Call Me Maybe - Function Calling CLI")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    return parser.parse_args()


def extract_json_object(text: str) -> Dict[str, Any] | None:
    """Extract and parse first complete JSON object from generated text."""
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "name" in parsed and "parameters" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def main() -> None:
    """Execute main function calling generation pipeline."""
    start_time = time.perf_counter()
    args = parse_args()

    functions_def = load_functions_definition(args.functions_definition)
    test_cases = load_test_cases(args.input)

    llm = Small_LLM_Model()

    vocab_path = llm.get_path_to_vocab_file()
    merges_path = vocab_path.replace("vocab.json", "merges.txt")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab: Dict[str, int] = json.load(f)

    tokenizer = ByteLevelBPETokenizer(
        vocab_path,
        merges_path,
        pre_tokenizer=RegexPreTokenizer(RegexPreTokenizer.QWEN_PATTERN),
    )

    decoder = JSONConstraintDecoder(vocab, functions_def, tokenizer)
    results: List[Dict[str, Any]] = []

    fns_dump = json.dumps([fn.model_dump() for fn in functions_def], indent=2)

    for i, test in enumerate(test_cases, 1):
        # Clean decoder cache for independent evaluation of each test prompt
        decoder.valid_cache.clear()

        prompt_text = test.prompt

        full_prompt = (
            f"Available functions:\n{fns_dump}\n\n"
            f"User request: {prompt_text}\n\n"
            f"Return ONLY a valid JSON object with 'name' and 'parameters':\n"
        )

        # Obtain input IDs safely using llm.encode()
        encoded = llm.encode(full_prompt)
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.squeeze(0).tolist()
        else:
            input_ids = list(encoded)

        generated_text = ""
        max_steps = 60

        for step in range(max_steps):
            raw_logits = llm.get_logits_from_input_ids(input_ids)
            masked_logits = decoder.mask_logits(raw_logits, generated_text)

            next_token_id = int(np.argmax(masked_logits))
            next_token_str = decoder.id_to_str.get(next_token_id, "")

            generated_text += next_token_str
            input_ids.append(next_token_id)

            parsed = extract_json_object(generated_text)
            if parsed is not None:
                parsed_result = FunctionCallResult(
                    prompt=prompt_text,
                    name=parsed["name"],
                    parameters=parsed["parameters"],
                )
                results.append(parsed_result.model_dump())
                print(f"[{i}/{len(test_cases)}] Generated JSON: {repr(generated_text.strip())}")
                break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - start_time
    print(f"\n⚡ Completed {len(test_cases)} prompts in {elapsed:.2f} seconds!")


if __name__ == "__main__":
    main()
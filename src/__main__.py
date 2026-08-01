"""Main entry point for function calling CLI."""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
from pydantic import ValidationError

from src.models import FunctionDefinition, TestCase, FunctionCallResult
from src.constrained import JSONConstraintDecoder
from src.tokenizer import CustomTokenizer
from llm_sdk.llm_sdk import Small_LLM_Model


def load_functions_definition(path_str: str) -> list[FunctionDefinition]:
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            print("Error: Expected a JSON array in functions definition file.",
                  file=sys.stderr)
            sys.exit(1)

        return [FunctionDefinition.model_validate(item) for item in raw]
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Error loading functions definition from '{path_str}': {e}",
              file=sys.stderr)
        sys.exit(1)


def load_test_cases(path_str: str) -> list[TestCase]:
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            print("Error: Expected a JSON array in input file.",
                  file=sys.stderr)
            sys.exit(1)

        return [TestCase.model_validate(item) for item in raw]
    except Exception as e:
        print(f"Error loading input prompts from '{path_str}': {e}",
              file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling CLI")
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calls.json")
    return parser.parse_args()


def try_parse_json(text: str) -> dict | None:
    """Try parsing JSON text directly or with appended closing braces."""
    text = text.strip()
    if not text:
        return None

    # 1. Direct check
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Early Auto-close check (կանգնեցնում է ցիկլը 8-րդ քայլում)
    for suffix in ["}", "}}", '"}}', '": ""}}']:
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue

    return None


def main() -> None:
    args = parse_args()
    functions_def = load_functions_definition(args.functions_definition)
    test_cases = load_test_cases(args.input)

    llm = Small_LLM_Model()

    vocab_path = llm.get_path_to_vocab_file()
    merges_path = vocab_path.replace("vocab.json", "merges.txt")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    tokenizer = CustomTokenizer(vocab_path, merges_path)
    decoder = JSONConstraintDecoder(vocab, functions_def, tokenizer)
    results: list[dict] = []

    for idx, test in enumerate(test_cases, 1):
        prompt_text = test.prompt
        print(f"[{idx}/{len(test_cases)}] Generating: {prompt_text[:25]}...", flush=True)

        full_prompt = f"Prompt: {prompt_text}\nJSON:"
        input_ids = llm.encode(full_prompt).tolist()
        generated_json = "{"

        # MAX STEPS = 18 (35-ի փոխարեն)
        max_steps = 18
        for step in range(max_steps):
            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]

            if hasattr(input_ids, "squeeze"):
                input_ids = input_ids.squeeze().tolist()

            raw_logits = llm.get_logits_from_input_ids(input_ids)
            masked_logits = decoder.mask_logits(raw_logits, generated_json)

            next_token_id = int(np.argmax(masked_logits))
            next_token_str = tokenizer.decode([next_token_id])

            generated_json += next_token_str
            input_ids.append(next_token_id)

            # JSON Parsing & Fast Break
            parsed = try_parse_json(generated_json)
            if parsed and isinstance(parsed, dict):
                if "name" in parsed and "parameters" in parsed and parsed["name"]:
                    parsed_result = FunctionCallResult(
                        prompt=prompt_text,
                        name=parsed["name"],
                        parameters=parsed["parameters"],
                    )
                    results.append(parsed_result.model_dump())
                    print(f"   -> Done in {step + 1} steps! ({parsed['name']})", flush=True)
                    break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Done! Results saved.", flush=True)


if __name__ == "__main__":
    main()
import argparse
import sys

def main() -> None:
    """Ծրագրի հիմնական մուտքի կետը:"""
    parser = argparse.ArgumentParser(description="Call Me Maybe - Function Calling CLI")
    
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json", help="Path to functions definition JSON")
    parser.add_argument("--input", default="data/input/function_calling_tests.json", help="Path to input prompts JSON")
    parser.add_argument("--output", default="data/output/function_calls.json", help="Path to output results JSON")
    
    args = parser.parse_args()
    
    print(f"Functions Path: {args.functions_definition}")
    print(f"Input Path: {args.input}")
    print(f"Output Path: {args.output}")
    
    # Այստեղ հետագայում կկանչենք մեր հիմնական pipeline-ը

if __name__ == "__main__":
    main()
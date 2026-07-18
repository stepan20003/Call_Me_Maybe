from llm_sdk import Small_LLM_Model
from src.tokenizer import CustomTokenizer

def test_my_tokenizer() -> None:
    # 1. Սահմանում ենք ֆայլերի ճշգրիտ ճանապարհները
    vocab_path = "/home/stepan/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json"
    merges_path = "/home/stepan/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/merges.txt"

    print("⏳ Բեռնվում են օրիգինալ SDK մոդելը և քո Թոքենայզերը...")
    model = Small_LLM_Model()
    my_tokenizer = CustomTokenizer(vocab_path=vocab_path, merges_path=merges_path)

    # 2. Թեստային տեքստեր (փորձիր պարզից մինչև բարդ նախադասություններ)
    test_prompts = [
        "Hello",
        "Hello World!",
        "Call me maybe",
        "The quick brown fox jumps over the lazy dog."
    ]

    print("\n🚀 Սկսվում է համեմատությունը...\n")
    
    all_passed = True
    for idx, text in enumerate(test_prompts, 1):
        print(f"--- Թեստ #{idx}: '{text}' ---")
        
        # Օրիգինալ SDK-ի արդյունքը
        sdk_ids = model.encode(text)
        if hasattr(sdk_ids, "tolist"):
            sdk_ids = sdk_ids.tolist()[0]
        # Քո գրած թոքենայզերի արդյունքը
        my_ids = my_tokenizer.encode(text)
        
        print(f"Original SDK IDs : {sdk_ids}")
        print(f"Your BPE IDs    : {my_ids}")
        
        if sdk_ids == my_ids:
            print("✅ ՄԱՔՈՒՐ Է (MATCH)")
        else:
            print("❌ ՍԽԱԼ ԿԱ (MISMATCH)")
            all_passed = False
            
            # Ստուգենք նաև դեկոդավորումը
            print(f"Original decoded: '{model.decode(sdk_ids)}'")
            print(f"Your decoded    : '{my_tokenizer.decode(my_ids)}'")
        print()

    if all_passed:
        print("🎉 Շնորհավորում եմ! Քո զրոյից գրած BPE Tokenizer-ը աշխատում է 100% ճշգրտությամբ:")
    else:
        print("⚠️ Որոշ թեստեր ձախողվեցին: Պետք է լավարկել BPE merge-ի տրամաբանությունը:")

if __name__ == "__main__":
    test_my_tokenizer()
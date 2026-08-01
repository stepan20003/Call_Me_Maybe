import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from src.tokenizer import CustomTokenizer

# 1. Ինիցիալիզացնում ենք մոդելն ու tokenizer-ը
model = Small_LLM_Model()
vocab_path = model.get_path_to_vocab_file()
tokenizer = CustomTokenizer(vocab_path)

# 2. Prompt-ը վերածում ենք ID-ների
prompt = "What is the sum of 2 and 3?"

input_ids = tokenizer.encode(prompt)

# 3. Ստանում ենք Logits-ները Qwen-ից
logits = model.get_logits_from_input_ids(input_ids)
logits_arr = np.array(logits)

# 4. Գտնում ենք Top 10 ամենաբարձր logit ունեցող թոքենների ID-ները
top_k = 10
top_indices = np.argsort(logits_arr)[-top_k:][::-1]

print(f"Prompt: '{prompt}'\n")
print(f"{'Rank':<5} | {'Token ID':<10} | {'Logit Score':<12} | {'Decoded Text'}")
print("-" * 50)

for rank, token_id in enumerate(top_indices, 1):
    score = logits_arr[token_id]
    # Decode ենք անում միայն տվյալ թոքենը
    token_str = tokenizer.decode([token_id])
    # Տերմինալում գեղեցիկ ցույց տալու համար
    printable_str = repr(token_str)
    print(f"{rank:<5} | {token_id:<10} | {score:<12.4f} | {printable_str}")
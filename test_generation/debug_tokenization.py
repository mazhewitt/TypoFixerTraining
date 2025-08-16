#!/usr/bin/env python3
"""Debug tokenization to understand the exact token count."""

from transformers import AutoTokenizer
import numpy as np

def main():
    tokenizer = AutoTokenizer.from_pretrained("mazhewitt/qwen-typo-fixer", trust_remote_code=True)
    prompt = "Fix: This setence has multple typos in it"
    
    print(f"Prompt: '{prompt}'")
    
    # Tokenize WITHOUT padding first
    inputs_no_pad = tokenizer(prompt, return_tensors="np", add_special_tokens=True)
    print(f"Without padding: {inputs_no_pad['input_ids'].shape}")
    print(f"Tokens: {inputs_no_pad['input_ids'][0].tolist()}")
    
    # Decode to see actual tokens
    for i, token_id in enumerate(inputs_no_pad['input_ids'][0]):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"  {i}: {token_id} -> '{token_text}'")
    
    print()
    
    # Tokenize WITH padding (max_length=64)
    inputs_pad = tokenizer(prompt, return_tensors="np", add_special_tokens=True, 
                          max_length=64, padding="max_length", truncation=True)
    print(f"With padding: {inputs_pad['input_ids'].shape}")
    print(f"Non-zero tokens: {np.count_nonzero(inputs_pad['input_ids'])}")
    
    # Find the actual length (first pad token)
    tokens = inputs_pad['input_ids'][0]
    actual_length = len(tokens)
    for i, token in enumerate(tokens):
        if token == tokenizer.pad_token_id or token == tokenizer.eos_token_id:
            actual_length = i
            break
    
    print(f"Actual length (before padding): {actual_length}")

if __name__ == "__main__":
    main()
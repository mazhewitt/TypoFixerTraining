#!/usr/bin/env python3
"""
Step 1: Generate prompt and tokenize
Generates few-shot prompt from test input and tokenizes it, saving with metadata.
"""

import json
import numpy as np
import os
from transformers import AutoTokenizer
from datetime import datetime

def create_few_shot_prompt(text_with_typos):
    """Create few-shot prompt for typo correction (from typo_fixer_complete.py)."""
    prompt = f"""Fix typos in these sentences:

Input: I beleive this is teh answer.
Output: I believe this is the answer.

Input: She recieved her degre yesterday.
Output: She received her degree yesterday.

Input: The resturant serves good food.
Output: The restaurant serves good food.

Input: {text_with_typos}
Output:"""
    return prompt

def tokenize_input(tokenizer, text, max_length=128):
    """Tokenize input text with proper padding (from typo_fixer_complete.py)."""
    inputs = tokenizer(
        text, 
        return_tensors="np", 
        max_length=max_length, 
        padding="max_length", 
        truncation=True,
        add_special_tokens=True
    )
    return inputs['input_ids'].astype(np.int32)

def main():
    """Generate prompt and tokenize test input."""
    # Test input sentence
    test_input = "This setence has multple typos in it"
    
    # Load tokenizer
    tokenizer_path = "mazhewitt/qwen-typo-fixer"
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Create few-shot prompt
    prompt = create_few_shot_prompt(test_input)
    print(f"Generated prompt ({len(prompt)} chars):")
    print(f"'{prompt}'")
    print()
    
    # Tokenize
    input_ids = tokenize_input(tokenizer, prompt, max_length=128)
    context_pos = np.count_nonzero(input_ids)  # Count non-zero tokens (actual length)
    
    print(f"Tokenized to shape: {input_ids.shape}")
    print(f"Actual tokens (non-padded): {context_pos}")
    print()
    
    # Create output data with metadata
    output_data = {
        "metadata": {
            "step": "1_tokenize",
            "input_text": test_input,
            "prompt": prompt,
            "tokenizer_path": tokenizer_path,
            "max_length": 128,
            "timestamp": datetime.now().isoformat(),
            "description": "Tokenized few-shot prompt for typo correction"
        },
        "data": {
            "input_ids": input_ids.tolist(),
            "context_pos": int(context_pos),
            "tensor_shape": list(input_ids.shape),
            "dtype": str(input_ids.dtype)
        }
    }
    
    # Save to file
    output_file = "test_generation/step_1_tokens.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved tokenized data to {output_file}")
    print(f"   Shape: {input_ids.shape}")
    print(f"   Actual length: {context_pos} tokens")
    print(f"   Data type: {input_ids.dtype}")
    
    # Show first few tokens for verification
    print(f"\nFirst 10 tokens: {input_ids[0][:10].tolist()}")
    decoded_sample = tokenizer.decode(input_ids[0][:context_pos], skip_special_tokens=False)
    print(f"Decoded preview: '{decoded_sample[:100]}...'")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test ByT5 with the CORRECT training prefix to see if that fixes the issue
"""

import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer

def test_with_correct_prefix():
    """Test ByT5 with the actual training prefix"""
    
    model_path = "./models/byt5-small-typo-fixer"
    
    print("Loading ByT5 typo fixer model...")
    tokenizer = ByT5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # TEST CASES with CORRECT prefix from training
    test_cases = [
        "I beleive this is teh correct answr.",
        "This is a sentnce with mny typos.",
        "The qick brown fox jumps ovr the lazy dog.",
    ]
    
    # Use the EXACT prefix from training script
    correct_prefix = "fix spelling errors only, don't change the meaning of the text:"
    wrong_prefix = "fix typos:"
    
    print("\n" + "="*80)
    print("TESTING WITH CORRECT vs WRONG PREFIX")
    print("="*80)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: '{test_text}' ---")
        
        for prefix_name, prefix in [("CORRECT", correct_prefix), ("WRONG", wrong_prefix)]:
            input_text = f"{prefix} {test_text}"
            
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,  # Increased from 128
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"{prefix_name:>7} prefix: '{result}'")

if __name__ == "__main__":
    test_with_correct_prefix()
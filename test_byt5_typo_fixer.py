#!/usr/bin/env python3
"""
Test script for the ByT5 small typo fixer model
"""

import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
import time

def load_model():
    """Load the ByT5 typo fixer model and tokenizer"""
    model_path = "./models/byt5-small-typo-fixer"
    
    print("Loading ByT5 typo fixer model...")
    tokenizer = ByT5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer, device

def fix_typos(text, model, tokenizer, device):
    """Fix typos in the given text"""
    # Prepare input with task prefix
    input_text = f"fix typos: {text}"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    # Decode
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def run_tests():
    """Run a series of typo correction tests"""
    
    # Load model
    model, tokenizer, device = load_model()
    
    # Test cases
    test_cases = [
        "I beleive this is teh correct answr.",
        "This is a sentnce with mny typos.",
        "The qick brown fox jumps ovr the lazy dog.",
        "Please chck your email for futher instructions.",
        "I recieved your mesage yesterday.",
        "We need to discus this matter urgently.",
        "The meetng is schedled for tomorrow.",
        "Can you plese send me the documnt?",
        "This is alredy completd.",
        "I dont understnd what you mean."
    ]
    
    print("\n" + "="*60)
    print("TESTING BYT5 SMALL TYPO FIXER")
    print("="*60)
    
    total_time = 0
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Original:  '{test_text}'")
        
        start_time = time.time()
        corrected = fix_typos(test_text, model, tokenizer, device)
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        
        print(f"Corrected: '{corrected}'")
        print(f"Time: {inference_time:.3f}s")
        
        # Simple check if correction was made
        if corrected.lower() != test_text.lower():
            print("Status: ✅ Changes made")
        else:
            print("Status: ⚠️ No changes")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tests: {len(test_cases)}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average time per correction: {total_time/len(test_cases):.3f}s")
    print(f"Model size: ByT5-small (~300M parameters)")

if __name__ == "__main__":
    run_tests()
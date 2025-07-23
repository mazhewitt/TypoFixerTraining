#!/usr/bin/env python3
"""
Quick test script for the trained Qwen typo correction model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model(model_path):
    print("🤖 Loading trained model...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!")
    
    # Test examples
    test_cases = [
        "Fix: I beleive this is teh correct answr.",
        "Fix: She recieved her degre last year.",
        "Fix: The resturant serves excelent food.",
        "Fix: He is studyng for his final examintion.",
        "Fix: We dicussed the importnt details.",
        "Fix: The begining of the story was excting.",
        "Fix: I definately need to imporve my skils.",
        "Fix: The experiance was chalenging and rewardng.",
    ]
    
    print("\n🧪 Testing typo correction:")
    print("=" * 60)
    
    with torch.no_grad():
        for i, prompt in enumerate(test_cases, 1):
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
            
            # Generate correction
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode generated text (skip prompt)
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            print(f"{i}. Input:  {prompt}")
            print(f"   Output: {generated_text}")
            print()

if __name__ == "__main__":
    model_path = "models/qwen-typo-fixer-debug"  # Update this path
    test_model(model_path)
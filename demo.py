#!/usr/bin/env python3
"""
Demonstration script showing the complete Qwen typo correction workflow.
This script demonstrates both the basic and optimized inference approaches.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def basic_typo_correction(model, tokenizer, text):
    """Basic typo correction using the trained model."""
    prompt = f"Fix: {text}"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
    
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    if '.' in generated:
        corrected = generated.split('.')[0].strip() + '.'
    else:
        corrected = generated.strip()
    
    return corrected

def optimized_typo_correction(model, tokenizer, corrupted_sentence):
    """Optimized inference achieving 88.5% sentence accuracy."""
    
    # Few-shot prompt with examples
    full_prompt = f"""Fix typos in these sentences:

Input: I beleive this is teh answer.
Output: I believe this is the answer.

Input: She recieved her degre yesterday.
Output: She received her degree yesterday.

Input: The resturant serves good food.
Output: The restaurant serves good food.

Input: {corrupted_sentence}
Output:"""
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=256)
    
    # Generate with optimal parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,  # Greedy decoding works best
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.02,
        )
    
    # Decode and clean output
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Extract just the corrected sentence
    generated = ' '.join(generated.split())
    if '\n' in generated:
        generated = generated.split('\n')[0].strip()
    
    # Remove suffixes that may appear
    for suffix in ['Input:', 'Output:', 'Human:', 'Assistant:']:
        if suffix.lower() in generated.lower():
            generated = generated.split(suffix)[0].strip()
    
    if '.' in generated:
        corrected = generated.split('.')[0].strip() + '.'
    else:
        corrected = generated.strip()
    
    return corrected

def demo():
    print("🎯 Qwen Typo Correction Demo")
    print("=" * 60)
    
    # Load model
    print("🤖 Loading fine-tuned model...")
    model_path = "mazhewitt/qwen-typo-fixer"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model loaded successfully!\n")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure you have internet connection to download from HuggingFace")
        return
    
    # Test examples
    test_examples = [
        "I beleive this is teh correct answr.",
        "She recieved her degre last year.",
        "The resturant serves excelent food.",
        "Please chekc your emial for more informaton.",
        "Thank you for your consideratin of my applicaton.",
        "I am goign to teh store to buy som grocerys.",
    ]
    
    print("📊 Comparison: Basic vs Optimized Approach")
    print("-" * 60)
    
    for i, example in enumerate(test_examples, 1):
        print(f"{i}. Input: {example}")
        
        # Basic approach
        basic_result = basic_typo_correction(model, tokenizer, example)
        print(f"   Basic:     {basic_result}")
        
        # Optimized approach  
        optimized_result = optimized_typo_correction(model, tokenizer, example)
        print(f"   Optimized: {optimized_result}")
        print()
    
    print("🏆 Summary:")
    print("- Basic approach: Simple 'Fix:' prompting")
    print("- Optimized approach: Few-shot prompting (88.5% accuracy)")
    print("- Use optimized approach for production applications")
    print()
    print("🔗 Model: https://huggingface.co/mazhewitt/qwen-typo-fixer")
    print("📚 Repository: https://github.com/mazhewitt/TypoFixerTraining")

if __name__ == "__main__":
    demo()
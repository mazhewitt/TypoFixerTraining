#!/usr/bin/env python3
"""
Conservative inference test script using the same algorithm as the training verification.
This prevents overfitted models from being too creative and focuses on typo correction.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def conservative_inference(model, tokenizer, prompt: str) -> str:
    """
    Conservative inference for typo correction - prevents overfitted models from being too creative.
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
    
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate MINIMAL correction (just fix typos, don't change anything else)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,  # Very short - just the corrected sentence
            do_sample=False,  # No sampling - deterministic
            num_beams=1,  # No beam search - fastest/most direct
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,  # Very minimal
        )
    
    # Decode generated text (skip prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    # AGGRESSIVE cleaning for overfitted model
    generated_text = generated_text.strip()
    
    # Remove newlines and extra whitespace
    generated_text = ' '.join(generated_text.split())
    
    # Split on period and take first part
    if '.' in generated_text:
        corrected = generated_text.split('.')[0].strip() + '.'
    else:
        corrected = generated_text.strip()
    
    # Remove unwanted symbols and prefixes that overfitted model adds
    corrected = corrected.replace('##', '').replace('#', '').strip()
    
    # Remove common overfitted prefixes
    unwanted_prefixes = [
        'Here is', 'The corrected', 'Correction:', 'Fixed:', 'Answer:', 
        'The answer is', 'Result:', 'Output:', 'Corrected:'
    ]
    for prefix in unwanted_prefixes:
        if corrected.lower().startswith(prefix.lower()):
            corrected = corrected[len(prefix):].strip()
    
    # Length limiting removed - conservative generation already handles this
    # The max_new_tokens=15 and do_sample=False already prevent over-generation
    
    return corrected

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
    
    print("\n🧪 Testing CONSERVATIVE typo correction:")
    print("=" * 60)
    
    for i, prompt in enumerate(test_cases, 1):
        # Use conservative inference algorithm
        corrected = conservative_inference(model, tokenizer, prompt)
        
        print(f"{i}. Input:  {prompt}")
        print(f"   Output: {corrected}")
        print()

if __name__ == "__main__":
    model_path = "mazhewitt/qwen-typo-fixer"  # HuggingFace uploaded model
    test_model(model_path)
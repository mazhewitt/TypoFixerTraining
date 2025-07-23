#!/usr/bin/env python3
"""
Compare fine-tuned model vs base model performance on typo correction.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def conservative_inference(model, tokenizer, prompt: str) -> str:
    """Conservative inference for typo correction."""
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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
    
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Clean up output
    generated_text = ' '.join(generated_text.split())
    if '.' in generated_text:
        corrected = generated_text.split('.')[0].strip() + '.'
    else:
        corrected = generated_text.strip()
    
    corrected = corrected.replace('##', '').replace('#', '').strip()
    
    # Remove unwanted prefixes
    unwanted_prefixes = [
        'Here is', 'The corrected', 'Correction:', 'Fixed:', 'Answer:', 
        'The answer is', 'Result:', 'Output:', 'Corrected:'
    ]
    for prefix in unwanted_prefixes:
        if corrected.lower().startswith(prefix.lower()):
            corrected = corrected[len(prefix):].strip()
    
    return corrected

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    normalized = ' '.join(text.strip().lower().split())
    if normalized.endswith('.'):
        normalized = normalized[:-1]
    return normalized

def test_models():
    print("🔍 Model Comparison: Fine-tuned vs Base Qwen")
    print("=" * 70)
    
    # Load fine-tuned model
    print("🤖 Loading fine-tuned model...")
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained("mazhewitt/qwen-typo-fixer", trust_remote_code=True)
    fine_tuned_model = AutoModelForCausalLM.from_pretrained("mazhewitt/qwen-typo-fixer", trust_remote_code=True)
    
    if fine_tuned_tokenizer.pad_token is None:
        fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token
    
    # Load base model
    print("🤖 Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    print("✅ Both models loaded successfully!\n")
    
    # Test cases with expected outputs
    test_cases = [
        {
            "input": "Fix: I beleive this is teh correct answr.",
            "expected": "I believe this is the correct answer."
        },
        {
            "input": "Fix: She recieved her degre last year.",
            "expected": "She received her degree last year."
        },
        {
            "input": "Fix: The resturant serves excelent food.",
            "expected": "The restaurant serves excellent food."
        },
        {
            "input": "Fix: He is studyng for his final examintion.",
            "expected": "He is studying for his final examination."
        },
        {
            "input": "Fix: We dicussed the importnt details yesterday.",
            "expected": "We discussed the important details yesterday."
        },
        {
            "input": "Fix: The begining of the story was excting.",
            "expected": "The beginning of the story was exciting."
        },
        {
            "input": "Fix: I definately need to imporve my skils.",
            "expected": "I definitely need to improve my skills."
        },
        {
            "input": "Fix: The experiance was chalenging and rewardng.",
            "expected": "The experience was challenging and rewarding."
        },
    ]
    
    fine_tuned_correct = 0
    base_correct = 0
    
    print("📊 Comparison Results:")
    print("-" * 70)
    
    for i, case in enumerate(test_cases, 1):
        prompt = case["input"]
        expected = case["expected"]
        
        # Test fine-tuned model
        ft_output = conservative_inference(fine_tuned_model, fine_tuned_tokenizer, prompt)
        ft_normalized = normalize_text(ft_output)
        expected_normalized = normalize_text(expected)
        ft_correct = ft_normalized == expected_normalized
        
        if ft_correct:
            fine_tuned_correct += 1
        
        # Test base model
        base_output = conservative_inference(base_model, base_tokenizer, prompt)
        base_normalized = normalize_text(base_output)
        base_correct_flag = base_normalized == expected_normalized
        
        if base_correct_flag:
            base_correct += 1
        
        # Display results
        ft_status = "✅" if ft_correct else "❌"
        base_status = "✅" if base_correct_flag else "❌"
        
        print(f"{i}. Input: {prompt.replace('Fix: ', '')}")
        print(f"   Expected:   {expected}")
        print(f"   Fine-tuned: {ft_output} {ft_status}")
        print(f"   Base model: {base_output} {base_status}")
        print()
    
    # Summary
    total_tests = len(test_cases)
    ft_accuracy = fine_tuned_correct / total_tests
    base_accuracy = base_correct / total_tests
    improvement = ft_accuracy - base_accuracy
    
    print("🏆 FINAL COMPARISON")
    print("=" * 70)
    print(f"Fine-tuned Model: {fine_tuned_correct}/{total_tests} = {ft_accuracy:.1%}")
    print(f"Base Model:       {base_correct}/{total_tests} = {base_accuracy:.1%}")
    print(f"Improvement:      {improvement:+.1%} ({improvement*100:+.1f} percentage points)")
    
    if improvement > 0.5:
        print("🚀 Excellent improvement from fine-tuning!")
    elif improvement > 0.2:
        print("✅ Good improvement from fine-tuning!")
    elif improvement > 0:
        print("🟡 Modest improvement from fine-tuning")
    else:
        print("❌ Fine-tuning did not improve performance")

if __name__ == "__main__":
    test_models()
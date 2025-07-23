#!/usr/bin/env python3
"""
Test few-shot prompting strategies for better typo correction performance.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def few_shot_inference(model, tokenizer, prompt: str, strategy: str = "basic") -> str:
    """Test different few-shot prompting strategies."""
    
    if strategy == "basic":
        # Current approach
        full_prompt = prompt
    
    elif strategy == "few_shot_examples":
        # Add examples in the prompt
        full_prompt = """Fix typos in these sentences:

Input: I beleive this is teh answer.
Output: I believe this is the answer.

Input: She recieved her degre yesterday.
Output: She received her degree yesterday.

Input: The resturant serves good food.
Output: The restaurant serves good food.

Input: """ + prompt.replace("Fix: ", "") + """
Output:"""

    elif strategy == "instruction_based":
        # Clear instructions
        full_prompt = f"""Task: Fix only the spelling errors in the sentence below. Do not change the meaning, grammar, or add extra words. Only correct misspelled words.

Sentence with typos: {prompt.replace('Fix: ', '')}
Corrected sentence:"""

    elif strategy == "step_by_step":
        # Step-by-step approach
        full_prompt = f"""Please correct the typos in this sentence step by step:
1. Identify misspelled words
2. Fix each spelling error
3. Return only the corrected sentence

Sentence: {prompt.replace('Fix: ', '')}
Corrected:"""

    elif strategy == "constrained":
        # Very constrained approach
        full_prompt = f"""ONLY fix spelling mistakes. Do NOT change anything else.

Original: {prompt.replace('Fix: ', '')}
Fixed:"""
    
    else:
        full_prompt = prompt
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with conservative settings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.02,
        )
    
    # Decode
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Clean up output
    generated_text = ' '.join(generated_text.split())
    
    # Extract just the corrected sentence
    if '\n' in generated_text:
        generated_text = generated_text.split('\n')[0].strip()
    
    if '.' in generated_text:
        corrected = generated_text.split('.')[0].strip() + '.'
    else:
        corrected = generated_text.strip()
    
    # Remove unwanted artifacts
    corrected = corrected.replace('##', '').replace('#', '').strip()
    
    # Remove common prefixes that models add
    unwanted_prefixes = [
        'Here is', 'The corrected', 'Correction:', 'Fixed:', 'Answer:', 
        'The answer is', 'Result:', 'Output:', 'Corrected:', 'Task:', 'Step',
        'Original:', 'Sentence:', 'Please', 'ONLY', '1.', '2.', '3.'
    ]
    for prefix in unwanted_prefixes:
        if corrected.lower().startswith(prefix.lower()):
            corrected = corrected[len(prefix):].strip()
            if corrected.startswith(':'):
                corrected = corrected[1:].strip()
    
    return corrected

def test_prompting_strategies():
    print("🧪 Testing Few-Shot Prompting Strategies")
    print("=" * 70)
    
    # Load model
    print("🤖 Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained("mazhewitt/qwen-typo-fixer", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("mazhewitt/qwen-typo-fixer", trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded!\n")
    
    # Test cases that previously failed
    challenging_cases = [
        {
            "input": "Fix: Please chekc your emial for more informaton.",
            "expected": "Please check your email for more information."
        },
        {
            "input": "Fix: Thank you for your consideratin of my applicaton.",
            "expected": "Thank you for your consideration of my application."
        },
        {
            "input": "Fix: I am goign to teh store to buy som grocerys.",
            "expected": "I am going to the store to buy some groceries."
        },
        {
            "input": "Fix: There going to there house tomorrow.",
            "expected": "They're going to their house tomorrow."
        },
        {
            "input": "Fix: Its a beautiful day, isnt it?",
            "expected": "It's a beautiful day, isn't it?"
        }
    ]
    
    strategies = [
        ("basic", "Current Approach"),
        ("few_shot_examples", "Few-Shot Examples"),
        ("instruction_based", "Clear Instructions"),
        ("step_by_step", "Step-by-Step"),
        ("constrained", "Heavily Constrained")
    ]
    
    results = {}
    
    for strategy_name, strategy_desc in strategies:
        print(f"📋 Testing Strategy: {strategy_desc}")
        print("-" * 50)
        
        correct = 0
        total = len(challenging_cases)
        
        for i, case in enumerate(challenging_cases, 1):
            prompt = case["input"]
            expected = case["expected"]
            
            # Test this strategy
            output = few_shot_inference(model, tokenizer, prompt, strategy_name)
            
            # Normalize for comparison
            output_norm = ' '.join(output.strip().lower().split())
            expected_norm = ' '.join(expected.strip().lower().split())
            
            # Remove trailing periods for comparison
            if output_norm.endswith('.'):
                output_norm = output_norm[:-1]
            if expected_norm.endswith('.'):
                expected_norm = expected_norm[:-1]
            
            is_correct = output_norm == expected_norm
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{i}. {status} Input:    {prompt.replace('Fix: ', '')}")
            print(f"   Expected: {expected}")
            print(f"   Output:   {output}")
            if not is_correct:
                print(f"   Norm Exp: '{expected_norm}'")
                print(f"   Norm Out: '{output_norm}'")
            print()
        
        accuracy = correct / total
        results[strategy_name] = {"correct": correct, "total": total, "accuracy": accuracy}
        
        print(f"📊 {strategy_desc} Results: {correct}/{total} = {accuracy:.1%}")
        print()
    
    # Summary comparison
    print("🏆 STRATEGY COMPARISON")
    print("=" * 70)
    
    best_strategy = max(results.items(), key=lambda x: x[1]["accuracy"])
    
    for strategy_name, strategy_desc in strategies:
        if strategy_name in results:
            r = results[strategy_name]
            marker = "🥇" if strategy_name == best_strategy[0] else "  "
            print(f"{marker} {strategy_desc:20s}: {r['correct']}/{r['total']} = {r['accuracy']:6.1%}")
    
    print(f"\n🎯 Best Strategy: {dict(strategies)[best_strategy[0]]} with {best_strategy[1]['accuracy']:.1%} accuracy")
    
    return results

if __name__ == "__main__":
    test_prompting_strategies()
#!/usr/bin/env python3
"""
Test different generation parameters for better typo correction.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_generation_params(model, tokenizer, prompt: str, **gen_params) -> str:
    """Test different generation parameters."""
    
    # Use few-shot prompt (best from previous test)
    full_prompt = f"""Fix typos in these sentences:

Input: I beleive this is teh answer.
Output: I believe this is the answer.

Input: She recieved her degre yesterday.
Output: She received her degree yesterday.

Input: The resturant serves good food.
Output: The restaurant serves good food.

Input: {prompt.replace('Fix: ', '')}
Output:"""
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Default parameters
    default_params = {
        'max_new_tokens': 25,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    
    # Merge with provided parameters
    generation_params = {**default_params, **gen_params}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_params)
    
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
    
    # Remove common suffixes that appear
    for suffix in ['Input:', 'Output:', 'Human:', 'Assistant:']:
        if suffix.lower() in generated_text.lower():
            generated_text = generated_text.split(suffix)[0].strip()
    
    if '.' in generated_text:
        corrected = generated_text.split('.')[0].strip() + '.'
    else:
        corrected = generated_text.strip()
    
    # Remove unwanted artifacts
    corrected = corrected.replace('##', '').replace('#', '').strip()
    
    return corrected

def test_parameter_combinations():
    print("🔧 Testing Generation Parameters")
    print("=" * 70)
    
    # Load model
    print("🤖 Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained("mazhewitt/qwen-typo-fixer", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("mazhewitt/qwen-typo-fixer", trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded!\n")
    
    # Test cases
    test_cases = [
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
        }
    ]
    
    # Different parameter combinations
    param_configs = [
        {
            "name": "Greedy (baseline)",
            "params": {
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": 1.02
            }
        },
        {
            "name": "Low temp sampling", 
            "params": {
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.8,
                "repetition_penalty": 1.05
            }
        },
        {
            "name": "Very low temp",
            "params": {
                "do_sample": True,
                "temperature": 0.1,
                "top_p": 0.5,
                "repetition_penalty": 1.1
            }
        },
        {
            "name": "Beam search",
            "params": {
                "do_sample": False,
                "num_beams": 3,
                "repetition_penalty": 1.05,
                "early_stopping": True
            }
        },
        {
            "name": "Top-k sampling",
            "params": {
                "do_sample": True,
                "top_k": 5,
                "temperature": 0.2,
                "repetition_penalty": 1.1
            }
        },
        {
            "name": "High repetition penalty",
            "params": {
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": 1.3,
                "no_repeat_ngram_size": 2
            }
        }
    ]
    
    results = {}
    
    for config in param_configs:
        config_name = config["name"]
        params = config["params"]
        
        print(f"🧪 Testing: {config_name}")
        print(f"   Parameters: {params}")
        print("-" * 50)
        
        correct = 0
        total = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            prompt = case["input"]
            expected = case["expected"]
            
            try:
                # Test this parameter set
                output = test_generation_params(model, tokenizer, prompt, **params)
                
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
                
            except Exception as e:
                print(f"{i}. ❌ ERROR: {e}")
                print()
        
        accuracy = correct / total
        results[config_name] = {"correct": correct, "total": total, "accuracy": accuracy}
        
        print(f"📊 {config_name} Results: {correct}/{total} = {accuracy:.1%}")
        print()
    
    # Summary comparison
    print("🏆 PARAMETER COMPARISON")
    print("=" * 70)
    
    best_config = max(results.items(), key=lambda x: x[1]["accuracy"])
    
    for config_name, result in results.items():
        marker = "🥇" if config_name == best_config[0] else "  "
        print(f"{marker} {config_name:20s}: {result['correct']}/{result['total']} = {result['accuracy']:6.1%}")
    
    print(f"\n🎯 Best Configuration: {best_config[0]} with {best_config[1]['accuracy']:.1%} accuracy")
    
    return results

if __name__ == "__main__":
    test_parameter_combinations()
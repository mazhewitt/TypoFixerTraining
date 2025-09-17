#!/usr/bin/env python3
"""
Test different generation parameters to make model less creative
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_conservative_params():
    # Load the best model
    model_path = "models/qwen-enhanced-typo-fixer/checkpoint-5500"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    test_cases = [
        "The meetign will start at 9 oclock tommorrow.",
        "I beleive this is teh correct answr.",
        "She recieved the packege yesteday afternoon."
    ]

    generation_configs = [
        {
            "name": "Current (too creative)",
            "params": {
                "max_new_tokens": 50,
                "do_sample": False,
                "repetition_penalty": 1.1
            }
        },
        {
            "name": "Conservative 1 - Lower temp + penalty",
            "params": {
                "max_new_tokens": 30,  # Shorter output
                "do_sample": True,
                "temperature": 0.1,    # Very low creativity
                "repetition_penalty": 1.3,  # Strong anti-repetition
                "length_penalty": 0.8  # Prefer shorter outputs
            }
        },
        {
            "name": "Conservative 2 - Greedy + constrained",
            "params": {
                "max_new_tokens": 25,  # Even shorter
                "do_sample": False,
                "repetition_penalty": 1.5,
                "no_repeat_ngram_size": 3,  # Prevent repeating 3-grams
                "early_stopping": True
            }
        },
        {
            "name": "Conservative 3 - Minimal changes",
            "params": {
                "max_new_tokens": 20,  # Very short
                "do_sample": True,
                "temperature": 0.01,   # Extremely low
                "top_p": 0.8,         # Nucleus sampling
                "repetition_penalty": 1.2
            }
        }
    ]

    for config in generation_configs:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {config['name']}")
        print(f"{'='*60}")

        for test_case in test_cases:
            prompt = f"<|im_start|>user\nCorrect the typos in this text: {test_case}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **config['params']
                )

            # Extract correction
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = outputs[0][len(prompt_tokens):]
            correction = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

            # Clean up
            if correction.startswith('\n'):
                correction = correction[1:]
            if '\nassistant\n' in correction:
                correction = correction.split('\nassistant\n')[0].strip()
            if correction.startswith('assistant\n'):
                correction = correction[10:].strip()

            print(f"Original:  '{test_case}'")
            print(f"Corrected: '{correction}'")
            print(f"{'-'*50}")

if __name__ == "__main__":
    test_conservative_params()
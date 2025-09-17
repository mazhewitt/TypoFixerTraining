#!/usr/bin/env python3
"""
Test extremely minimal approaches to get conservative corrections
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_minimal_approaches():
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

    test_case = "The meetign will start at 9 oclock tommorrow."

    approaches = [
        {
            "name": "Ultra-short generation",
            "prompt": f"<|im_start|>user\nCorrect: {test_case}<|im_end|>\n<|im_start|>assistant\n",
            "params": {
                "max_new_tokens": 15,  # Very short
                "do_sample": False,    # Greedy
                "repetition_penalty": 2.0  # Strong penalty
            }
        },
        {
            "name": "Completion style",
            "prompt": f"Correct spelling: {test_case}\nCorrected: ",
            "params": {
                "max_new_tokens": 12,
                "do_sample": False,
                "repetition_penalty": 1.5,
                "early_stopping": True
            }
        },
        {
            "name": "Direct instruction",
            "prompt": f"<|im_start|>user\n{test_case}\n\nFix spelling only:<|im_end|>\n<|im_start|>assistant\n",
            "params": {
                "max_new_tokens": 10,
                "do_sample": False,
                "repetition_penalty": 3.0
            }
        },
        {
            "name": "Minimal context",
            "prompt": f"Fix: {test_case}\n",
            "params": {
                "max_new_tokens": 8,
                "do_sample": False,
                "early_stopping": True
            }
        },
        {
            "name": "Stop at period",
            "prompt": f"<|im_start|>user\nCorrect typos: {test_case}<|im_end|>\n<|im_start|>assistant\n",
            "params": {
                "max_new_tokens": 20,
                "do_sample": False,
                "repetition_penalty": 1.8,
                "stopping_criteria": None  # We'll add custom stopping
            }
        }
    ]

    for approach in approaches:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {approach['name']}")
        print(f"{'='*60}")

        prompt = approach['prompt']
        print(f"Prompt: {repr(prompt)}")

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **approach['params']
            )

        # Extract correction based on prompt style
        if approach['name'] == "Completion style":
            # For completion style, just decode everything after input
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            correction = full_text.replace(prompt, "").strip()
        elif approach['name'] == "Minimal context":
            # For minimal context, extract after "Fix: "
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            correction = full_text.replace(prompt, "").strip()
        else:
            # For chat formats, extract response tokens
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = outputs[0][len(prompt_tokens):]
            correction = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        # Basic cleanup
        if correction.startswith('\n'):
            correction = correction[1:]

        # Stop at first period for this test
        if '.' in correction and approach['name'] == "Stop at period":
            correction = correction.split('.')[0] + '.'

        print(f"Original:  '{test_case}'")
        print(f"Corrected: '{correction}'")
        print(f"Length: {len(correction.split())} words (vs {len(test_case.split())} original)")

if __name__ == "__main__":
    test_minimal_approaches()
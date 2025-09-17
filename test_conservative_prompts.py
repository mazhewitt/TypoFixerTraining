#!/usr/bin/env python3
"""
Test different prompt styles to make model more conservative
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_conservative_prompts():
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

    prompts = [
        {
            "name": "Original",
            "prompt": f"<|im_start|>user\nCorrect the typos in this text: {test_case}<|im_end|>\n<|im_start|>assistant\n"
        },
        {
            "name": "Minimal changes only",
            "prompt": f"<|im_start|>user\nFix only the spelling errors in this text, change nothing else: {test_case}<|im_end|>\n<|im_start|>assistant\n"
        },
        {
            "name": "Conservative instruction",
            "prompt": f"<|im_start|>user\nCorrect only the typos in this text. Do not change the meaning or add information: {test_case}<|im_end|>\n<|im_start|>assistant\n"
        },
        {
            "name": "Strict instruction",
            "prompt": f"<|im_start|>user\nFix the spelling mistakes. Keep everything else exactly the same: {test_case}<|im_end|>\n<|im_start|>assistant\n"
        },
        {
            "name": "Explicit constraint",
            "prompt": f"<|im_start|>user\nCorrect spelling errors only. Do not change times, dates, or other details: {test_case}<|im_end|>\n<|im_start|>assistant\n"
        }
    ]

    for prompt_config in prompts:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {prompt_config['name']}")
        print(f"{'='*60}")

        prompt = prompt_config['prompt']
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=25,  # Conservative length
                do_sample=True,
                temperature=0.1,    # Low creativity
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
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

        print(f"Prompt: {prompt_config['prompt']}")
        print(f"Original:  '{test_case}'")
        print(f"Corrected: '{correction}'")
        print(f"{'-'*50}")

if __name__ == "__main__":
    test_conservative_prompts()
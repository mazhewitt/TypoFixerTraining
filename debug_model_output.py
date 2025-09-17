#!/usr/bin/env python3
"""
Debug what the model is actually outputting
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def debug_model():
    # Load the final model
    model_path = "models/qwen-enhanced-typo-fixer"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Test case
    test_text = "I beleive this is teh correct answr."

    print(f"🧪 Testing: '{test_text}'")
    print("=" * 60)

    # Try the prompt format
    prompt = f"<|im_start|>user\nCorrect the typos in this text: {test_text}<|im_end|>\n<|im_start|>assistant\n"

    print(f"📝 Full prompt:")
    print(repr(prompt))
    print()

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # Move to GPU
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Reduce to prevent repetition
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # Prevent repetition
        )

    # Decode full output
    full_generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"🔍 Raw model output:")
    print(repr(full_generated))
    print()

    print(f"🔍 Pretty model output:")
    print(full_generated)
    print()

    # NEW PARSING METHOD: Extract just the new tokens
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    response_tokens = outputs[0][len(prompt_tokens):]
    correction = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    print(f"🔧 NEW: Prompt tokens length: {len(prompt_tokens)}")
    print(f"🔧 NEW: Response tokens: {response_tokens}")
    print(f"🔧 NEW: Raw correction: {repr(correction)}")

    # Clean up common issues
    if correction.startswith('\n'):
        correction = correction[1:]

    # If there are multiple assistant responses, take the first one
    if '\nassistant\n' in correction:
        correction = correction.split('\nassistant\n')[0].strip()

    # Remove any remaining assistant labels
    if correction.startswith('assistant\n'):
        correction = correction[10:].strip()

    print(f"✨ Final extracted correction:")
    print(repr(correction))
    print()
    print(f"Final: '{correction}'")

if __name__ == "__main__":
    debug_model()
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
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode full output
    full_generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"🔍 Raw model output:")
    print(repr(full_generated))
    print()

    print(f"🔍 Pretty model output:")
    print(full_generated)
    print()

    # Try to extract just the assistant response
    if "<|im_start|>assistant" in full_generated:
        assistant_part = full_generated.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in assistant_part:
            correction = assistant_part.split("<|im_end|>")[0].strip()
        else:
            correction = assistant_part.strip()
    else:
        # Fallback: remove the original prompt
        correction = full_generated.replace(prompt, "").strip()

    print(f"✨ Extracted correction:")
    print(repr(correction))
    print()
    print(f"Final: '{correction}'")

if __name__ == "__main__":
    debug_model()
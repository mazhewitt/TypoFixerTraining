#!/usr/bin/env python3
"""
Evaluate all saved checkpoints and select the best one.
"""
import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def evaluate_checkpoint(checkpoint_path, test_examples):
    """Evaluate a specific checkpoint."""
    print(f"📊 Evaluating {checkpoint_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    correct = 0
    total = len(test_examples)

    for example in tqdm(test_examples, desc="Testing"):
        corrupted = example['corrupted']
        expected = example['clean']

        # Generate correction
        prompt = f"<|im_start|>user\nCorrect the typos in this text: {corrupted}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        # Move inputs to same device as model
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

        # Decode with special tokens to see structure
        generated_with_tokens = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the new tokens (after the prompt)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = outputs[0][len(prompt_tokens):]
        correction = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        # Clean up common issues
        if correction.startswith('\n'):
            correction = correction[1:]

        # If there are multiple assistant responses, take the first one
        if '\nassistant\n' in correction:
            correction = correction.split('\nassistant\n')[0].strip()

        # Remove any remaining assistant labels
        if correction.startswith('assistant\n'):
            correction = correction[10:].strip()

        # Simple exact match evaluation (case insensitive, stripped)
        if correction.lower().strip() == expected.lower().strip():
            correct += 1

    accuracy = correct / total
    return accuracy

def find_best_checkpoint():
    """Find and evaluate all checkpoints."""
    base_dir = Path("models/qwen-enhanced-typo-fixer")

    # Load test examples
    test_examples = []
    with open("data/enhanced_qwen_training.jsonl", "r") as f:
        all_examples = [json.loads(line) for line in f]
        # Use last 100 examples as test set
        for example in all_examples[-100:]:
            messages = example['messages']
            corrupted = messages[0]['content'].replace("Correct the typos in this text: ", "")
            clean = messages[1]['content']
            test_examples.append({'corrupted': corrupted, 'clean': clean})

    print(f"📊 Loaded {len(test_examples)} test examples")

    # Find all checkpoint directories
    checkpoints = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            checkpoints.append(item)

    if not checkpoints:
        print("⚠️  No checkpoints found, using final model")
        if base_dir.exists() and ((base_dir / "pytorch_model.bin").exists() or (base_dir / "model.safetensors").exists()):
            checkpoints = [base_dir]

    if not checkpoints:
        print("❌ No valid model found!")
        return None, 0

    # Evaluate each checkpoint
    results = {}
    for checkpoint in sorted(checkpoints):
        if (checkpoint / "pytorch_model.bin").exists() or (checkpoint / "model.safetensors").exists():
            try:
                accuracy = evaluate_checkpoint(checkpoint, test_examples)
                results[str(checkpoint)] = accuracy
                print(f"  {checkpoint.name}: Accuracy = {accuracy:.3f}")
            except Exception as e:
                print(f"  {checkpoint.name}: Failed to evaluate - {e}")

    if not results:
        print("❌ No checkpoints could be evaluated!")
        return None, 0

    # Find best checkpoint
    best_checkpoint = max(results.keys(), key=lambda k: results[k])
    best_accuracy = results[best_checkpoint]

    print(f"\n🏆 Best Checkpoint: {best_checkpoint}")
    print(f"📊 Best Accuracy: {best_accuracy:.3f}")

    return best_checkpoint, best_accuracy

def test_best_model():
    """Test the best model with sample examples."""
    best_checkpoint, best_accuracy = find_best_checkpoint()

    if not best_checkpoint:
        return

    # Load the best model
    print(f"\n🧪 Testing best model: {best_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(best_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        best_checkpoint,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Test cases
    test_cases = [
        "I beleive this is teh correct answr.",
        "The meetign will start at 9 oclock tommorrow.",
        "She recieved the packege yesteday afternoon.",
        "We need to discus the projcet detials today",
        "The resturant serves excellnt food evry day"
    ]

    print("\n📝 Sample Corrections:")
    print("=" * 50)

    for test_case in test_cases:
        prompt = f"<|im_start|>user\nCorrect the typos in this text: {test_case}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        # Move inputs to same device as model
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

        # Extract just the new tokens (after the prompt)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = outputs[0][len(prompt_tokens):]
        correction = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        # Clean up common issues
        if correction.startswith('\n'):
            correction = correction[1:]

        # If there are multiple assistant responses, take the first one
        if '\nassistant\n' in correction:
            correction = correction.split('\nassistant\n')[0].strip()

        # Remove any remaining assistant labels
        if correction.startswith('assistant\n'):
            correction = correction[10:].strip()

        print(f"Original:  '{test_case}'")
        print(f"Corrected: '{correction}'")
        print("-" * 50)

if __name__ == "__main__":
    test_best_model()
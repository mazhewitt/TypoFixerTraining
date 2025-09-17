#!/usr/bin/env python3
"""
Pre-tokenize dataset for faster training startup
"""
import os
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from train_enhanced_qwen import load_enhanced_dataset, preprocess_function

def pretokenize_dataset(
    model_name="Qwen/Qwen3-0.6B-Base",
    train_file="data/enhanced_qwen_training.jsonl",
    output_dir="data/tokenized_cache",
    max_length=512,
    eval_split=0.1
):
    """Pre-tokenize the dataset and save to cache."""

    print("🔧 Pre-tokenizing dataset for faster training...")
    print(f"Model: {model_name}")
    print(f"Dataset: {train_file}")
    print(f"Output: {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("📊 Loading dataset...")
    dataset = load_enhanced_dataset(train_file, eval_split)

    print(f"Train samples: {len(dataset['train']):,}")
    if 'eval' in dataset:
        print(f"Eval samples: {len(dataset['eval']):,}")

    # Tokenize dataset
    print("🔤 Tokenizing...")
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer, max_length)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing",
        num_proc=4  # Use multiple processes for speed
    )

    # Save tokenized dataset
    output_path = Path(output_dir) / "tokenized_dataset"
    print(f"💾 Saving tokenized dataset to: {output_path}")
    tokenized_dataset.save_to_disk(str(output_path))

    # Save metadata
    metadata = {
        "model_name": model_name,
        "train_file": train_file,
        "max_length": max_length,
        "eval_split": eval_split,
        "train_size": len(tokenized_dataset['train']),
        "eval_size": len(tokenized_dataset.get('eval', [])),
        "tokenizer_vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
    }

    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"📋 Saved metadata to: {metadata_path}")
    print()
    print("✅ Pre-tokenization complete!")
    print(f"🚀 Now you can train with: python train_single_gpu_cached.py")
    print(f"💡 Or use: python train_enhanced_qwen.py --config-file training_config_single_gpu_cached.json")

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset for training")
    parser.add_argument('--model-name', type=str, default="Qwen/Qwen3-0.6B-Base", help='Model name for tokenizer')
    parser.add_argument('--train-file', type=str, default="data/enhanced_qwen_training.jsonl", help='Training data file')
    parser.add_argument('--output-dir', type=str, default="data/tokenized_cache", help='Output directory for cached tokens')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--eval-split', type=float, default=0.1, help='Eval split ratio')

    args = parser.parse_args()

    pretokenize_dataset(
        model_name=args.model_name,
        train_file=args.train_file,
        output_dir=args.output_dir,
        max_length=args.max_length,
        eval_split=args.eval_split
    )

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Pre-tokenize and cache the training dataset to speed up training iterations.

This script tokenizes the dataset once and saves it to disk, eliminating the need
to re-tokenize on every training run.
"""

import os
import json
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_enhanced_dataset(file_path: str, eval_split: float = 0.1) -> DatasetDict:
    """Load the enhanced Qwen training dataset."""

    logger.info(f"Loading dataset from {file_path}...")

    # Load JSONL data
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                examples.append(data)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON on line {line_num}")
                continue

    logger.info(f"Loaded {len(examples):,} examples")

    # Convert to datasets format
    processed_examples = []
    for example in examples:
        # Extract messages
        messages = example.get('messages', [])
        if len(messages) != 2:
            continue

        user_msg = messages[0]['content']
        assistant_msg = messages[1]['content']

        # Create conversation format for Qwen
        conversation = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"

        processed_examples.append({
            'text': conversation,
            'metadata': example.get('metadata', {})
        })

    logger.info(f"Processed {len(processed_examples):,} examples")

    # Create Dataset
    dataset = Dataset.from_list(processed_examples)

    # Split into train/eval
    if eval_split > 0:
        dataset = dataset.train_test_split(test_size=eval_split, seed=42)
        dataset_dict = DatasetDict({
            'train': dataset['train'],
            'eval': dataset['test']
        })
    else:
        dataset_dict = DatasetDict({'train': dataset})

    logger.info(f"Dataset splits: train={len(dataset_dict['train']):,}, eval={len(dataset_dict.get('eval', [])):,}")

    return dataset_dict

def preprocess_function(examples: Dict, tokenizer: AutoTokenizer, max_length: int = 512) -> Dict:
    """Preprocess examples for training with proper padding and truncation."""

    # Tokenize the text
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',  # Pad to max_length for consistent tensor sizes
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=True
    )

    # For causal LM, labels are the same as input_ids
    # But we need to ignore padding tokens in loss calculation
    labels = tokenized['input_ids'].copy()

    # Replace padding token ids with -100 so they're ignored in loss calculation
    for i, input_ids in enumerate(labels):
        labels[i] = [token_id if token_id != tokenizer.pad_token_id else -100 for token_id in input_ids]

    tokenized['labels'] = labels

    return tokenized

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize and cache the training dataset")
    parser.add_argument('--model-name', type=str, default="Qwen/Qwen3-0.6B-Base", help='Model name for tokenizer')
    parser.add_argument('--train-file', type=str, default="data/enhanced_qwen_training.jsonl", help='Training data file')
    parser.add_argument('--output-dir', type=str, default="data/tokenized_cache", help='Output directory for cached tokens')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--eval-split', type=float, default=0.1, help='Evaluation split ratio')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for tokenization')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side='right'
    )

    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    logger.info("Loading and preprocessing dataset...")
    dataset = load_enhanced_dataset(args.train_file, args.eval_split)

    # Tokenize dataset
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer, args.max_length)

    logger.info("Tokenizing dataset (this will take a few minutes)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing dataset",
        num_proc=4  # Use multiple processes for faster tokenization
    )

    # Save tokenized dataset
    logger.info(f"Saving tokenized dataset to {output_path}")

    # Save using datasets library format (Arrow format - very efficient)
    tokenized_dataset.save_to_disk(str(output_path / "tokenized_dataset"))

    # Also save metadata
    metadata = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'eval_split': args.eval_split,
        'train_size': len(tokenized_dataset['train']),
        'eval_size': len(tokenized_dataset.get('eval', [])),
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'vocab_size': len(tokenizer)
    }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save tokenizer for reference
    tokenizer.save_pretrained(str(output_path / "tokenizer"))

    print(f"\n✅ Pre-tokenization complete!")
    print(f"📁 Cached data saved to: {output_path}")
    print(f"📊 Train samples: {metadata['train_size']:,}")
    print(f"📊 Eval samples: {metadata['eval_size']:,}")
    print(f"\n💡 To use cached tokens, run training with:")
    print(f"   --use-cached-tokens --cached-tokens-dir {output_path}")

    # Calculate size on disk
    total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    print(f"\n💾 Cache size: {total_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
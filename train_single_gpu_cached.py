#!/usr/bin/env python3
"""
Single GPU Training Script with Cached Tokenization for Qwen Typo Fixer
Uses pre-tokenized cached dataset for faster training startup
"""
import os
import torch
from train_enhanced_qwen import QwenTrainingConfig, train_enhanced_qwen
import json

def setup_single_gpu():
    """Setup for single GPU training."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    gpu_count = torch.cuda.device_count()
    print(f"🔥 Available GPUs: {gpu_count}")

    # Use first GPU
    gpu_name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Using GPU 0: {gpu_name} ({memory:.1f}GB)")

    # Set to use only first GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    return 1

def main():
    # Setup single GPU
    gpu_count = setup_single_gpu()

    # Load training configuration for single GPU with caching
    config_file = 'training_config_single_gpu_cached.json'
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        print("💡 Please create the config file or run without caching")
        return

    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    config = QwenTrainingConfig(**config_dict)

    # Verify cached tokens exist
    from pathlib import Path
    cached_dataset_path = Path(config.cached_tokens_dir) / "tokenized_dataset"

    if not cached_dataset_path.exists():
        print(f"❌ Cached dataset not found at: {cached_dataset_path}")
        print(f"💡 Please run first: python pretokenize_dataset.py --output-dir {config.cached_tokens_dir}")
        print("   This will pre-tokenize your dataset for faster training")
        return

    # Start training
    print("🚀 Starting Enhanced Qwen Training on Single GPU with Cached Tokens")
    print("💡 Using pre-tokenized dataset for faster startup")
    train_enhanced_qwen(config)

if __name__ == "__main__":
    main()
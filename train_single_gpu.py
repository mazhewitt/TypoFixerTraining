#!/usr/bin/env python3
"""
Single GPU Training Script for Enhanced Qwen Typo Fixer
Fallback when dual GPU training has CUDA issues
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

    # Load training configuration
    with open('training_config.json', 'r') as f:
        config_dict = json.load(f)

    # Adjust for single GPU - increase batch size to compensate
    config_dict['per_device_train_batch_size'] = 32  # Increase from 16
    config_dict['per_device_eval_batch_size'] = 32   # Increase from 16
    config_dict['gradient_accumulation_steps'] = 1   # Reduce since bigger batch
    # Total effective batch size = 32 * 1 GPU * 1 grad_accum = 32

    # Update run name
    config_dict['run_name'] = "qwen-enhanced-typo-fixer-single-gpu"

    config = QwenTrainingConfig(**config_dict)

    # Start training
    print("🚀 Starting Enhanced Qwen Training on Single GPU")
    print("💡 Using larger batch size to compensate for single GPU")
    train_enhanced_qwen(config)

if __name__ == "__main__":
    main()
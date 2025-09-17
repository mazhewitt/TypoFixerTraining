#!/usr/bin/env python3
"""
Dual RTX5090 Training Script for Enhanced Qwen Typo Fixer
"""
import os
import torch
from train_enhanced_qwen import QwenTrainingConfig, train_enhanced_qwen
import json

def setup_dual_gpu():
    """Setup for dual GPU training."""
    # Verify GPU setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    gpu_count = torch.cuda.device_count()
    print(f"🔥 Available GPUs: {gpu_count}")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({memory:.1f}GB)")

    if gpu_count < 2:
        print("⚠️  Warning: Expected 2 GPUs, found", gpu_count)

    # Set environment variables for distributed training
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use both GPUs

    return gpu_count

def main():
    # Setup GPUs
    gpu_count = setup_dual_gpu()

    # Load training configuration for dual GPU
    config_file = 'training_config_dual_gpu_test.json'
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    # Use conservative batch sizes for dual GPU with Qwen3-0.6B-Base
    config_dict['per_device_train_batch_size'] = 12  # Per GPU - conservative
    config_dict['per_device_eval_batch_size'] = 6    # Per GPU - conservative
    config_dict['gradient_accumulation_steps'] = 2   # To maintain effective batch size
    # Total effective batch size = 12 * 2 GPUs * 2 grad_accum = 48

    config = QwenTrainingConfig(**config_dict)

    # Start training
    print("🚀 Starting Enhanced Qwen Training on Dual RTX5090")
    train_enhanced_qwen(config)

if __name__ == "__main__":
    main()
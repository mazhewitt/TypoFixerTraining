#!/usr/bin/env python3
"""
Memory-optimized training script for dual RTX 5070 Ti (16GB each).
Reduced batch sizes and optimized for 16GB VRAM per GPU.
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    default_data_collator,
)
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenTypoDataset(Dataset):
    """Memory-optimized dataset for RTX 5070 Ti training."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 128):  # Reduced from 256
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"📖 Loading training data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading examples"):
                data = json.loads(line.strip())
                
                # Format as instruction-following prompt
                prompt = f"Fix: {data['corrupted']}"  # Shorter prompt
                target = data['clean']
                
                # Create full training text with clear separator
                full_text = f"{prompt}\n{target}"
                
                # Tokenize with padding for batching efficiency
                encoding = self.tokenizer(
                    full_text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Create labels (copy input_ids, mask prompt)
                labels = encoding['input_ids'].clone()
                
                # Find prompt length to mask it in loss calculation
                prompt_tokens = self.tokenizer(
                    prompt + "\n",
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'  
                )
                prompt_length = min(prompt_tokens['input_ids'].shape[-1], labels.shape[-1])
                
                # Mask prompt tokens (-100 = ignore in loss)
                labels[:, :prompt_length] = -100
                
                self.examples.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': labels.squeeze(),
                })
        
        logger.info(f"✅ Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def setup_model_tokenizer(model_name: str):
    """Setup model and tokenizer for RTX 5070 Ti with memory optimization."""
    logger.info(f"🚀 Loading model for dual RTX 5070 Ti: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use FP16 instead of BF16 for memory
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Resize embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer

def create_validation_split(dataset, ratio=0.1):
    """Create train/validation split."""
    total = len(dataset)
    val_size = int(total * ratio)
    train_size = total - val_size
    
    indices = torch.randperm(total).tolist()
    train_subset = torch.utils.data.Subset(dataset, indices[:train_size])
    val_subset = torch.utils.data.Subset(dataset, indices[train_size:])
    
    return train_subset, val_subset

def main():
    parser = argparse.ArgumentParser(description="Memory-optimized Qwen typo correction training")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B',
                       help='Base Qwen model')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Training JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--hf_repo', type=str,
                       help='HuggingFace repo to push to (username/repo)')
    
    # Memory-optimized parameters for RTX 5070 Ti
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Max sequence length (reduced for memory)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Per-device batch size (reduced for 16GB VRAM)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation (increased to maintain effective batch size)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                       help='Warmup ratio')
    parser.add_argument('--save_steps', type=int, default=50,
                       help='Save every N steps')
    parser.add_argument('--eval_steps', type=int, default=50,
                       help='Eval every N steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='Log every N steps')
    
    # Quality targets
    parser.add_argument('--target_accuracy', type=float, default=0.9,
                       help='Target accuracy (90%)')
    
    args = parser.parse_args()
    
    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Set seed
    set_seed(42)
    
    logger.info("🚀 Memory-Optimized Dual RTX 5070 Ti Training")
    logger.info(f"📁 Output: {args.output_dir}")
    logger.info(f"🎯 Target accuracy: {args.target_accuracy:.1%}")
    logger.info(f"💾 Memory optimizations enabled for 16GB VRAM per GPU")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_tokenizer(args.model_name)
    
    # Load dataset
    full_dataset = QwenTypoDataset(args.train_file, tokenizer, args.max_seq_len)
    train_dataset, eval_dataset = create_validation_split(full_dataset, 0.1)
    
    logger.info(f"📊 Train: {len(train_dataset):,} examples")
    logger.info(f"📊 Eval: {len(eval_dataset):,} examples")
    
    # Calculate training parameters
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * 2  # 2 GPUs
    total_steps = len(train_dataset) // effective_batch_size * args.num_epochs
    
    logger.info(f"📊 Effective batch size: {effective_batch_size}")
    logger.info(f"📊 Total training steps: {total_steps:,}")
    
    # Memory-optimized training arguments
    training_args = TrainingArguments(
        # Basic setup
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        
        # Memory-optimized batch sizes
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Learning parameters
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,  # Reduced to save disk space
        
        # Memory optimizations for RTX 5070 Ti
        fp16=True,  # Use FP16 for memory efficiency
        dataloader_pin_memory=False,  # Disable for memory
        dataloader_num_workers=4,  # Reduced workers
        gradient_checkpointing=True,  # Essential for memory saving
        max_grad_norm=1.0,  # Gradient clipping
        
        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        
        # Other optimizations
        remove_unused_columns=False,
        prediction_loss_only=True,  # Faster evaluation
        
        # Disable wandb by default
        report_to=[],
        run_name=f"qwen-typo-{args.num_epochs}ep-mem-opt",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    
    # Training info
    logger.info(f"\n📋 Memory-Optimized Training Plan:")
    logger.info(f"   GPUs: 2x RTX 5070 Ti (16GB each)")
    logger.info(f"   Precision: FP16 (memory optimized)")
    logger.info(f"   Sequence length: {args.max_seq_len} tokens")
    logger.info(f"   Batch size: {args.batch_size} per GPU")
    logger.info(f"   Effective batch: {effective_batch_size}")
    logger.info(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"   Total steps: {total_steps:,}")
    
    # Estimate training time
    estimated_time_min = total_steps * 0.6 / 60  # ~0.6 sec per step with smaller batches
    logger.info(f"   Estimated time: {estimated_time_min:.1f} minutes")
    
    # Clear cache before training
    torch.cuda.empty_cache()
    
    # Start training
    logger.info("\n🚀 Starting memory-optimized training...")
    start_time = time.time()
    
    try:
        # Train the model
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # Save model
        logger.info(f"💾 Saving model to {args.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training info
        training_info = {
            "model_name": args.model_name,
            "training_time_minutes": training_time / 60,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "training_examples": len(train_dataset),
            "validation_examples": len(eval_dataset),
            "epochs": args.num_epochs,
            "effective_batch_size": effective_batch_size,
            "learning_rate": args.learning_rate,
            "max_seq_len": args.max_seq_len,
            "gpu": "2x RTX 5070 Ti (memory optimized)",
            "optimizations": ["FP16", "Gradient Checkpointing", "Small Batches"],
            "total_steps": total_steps,
        }
        
        with open(f"{args.output_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        # HuggingFace deployment
        if args.hf_repo:
            logger.info(f"🤗 Model ready for upload to {args.hf_repo}")
            logger.info(f"📋 Upload command: cd {args.output_dir} && huggingface-cli upload . {args.hf_repo}")
        
        logger.info("\n🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {args.output_dir}")
        logger.info(f"⏱️ Training time: {training_time/60:.1f} minutes")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
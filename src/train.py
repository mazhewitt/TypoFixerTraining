#!/usr/bin/env python3
"""
Training script for DistilBERT typo correction using masked language modeling.
Freezes all layers except the MLM head to reduce trainable parameters from 66M to 1.5M.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TypoDataset(Dataset):
    """Dataset for corrupted/clean sentence pairs formatted for MLM training."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading data"):
                data = json.loads(line.strip())
                
                # Tokenize corrupted text
                corrupted_tokens = self.tokenizer(
                    data['corrupted'],
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Tokenize clean text for labels
                clean_tokens = self.tokenizer(
                    data['clean'],
                    truncation=True, 
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                self.examples.append({
                    'input_ids': corrupted_tokens['input_ids'].squeeze(),
                    'attention_mask': corrupted_tokens['attention_mask'].squeeze(),
                    'labels': clean_tokens['input_ids'].squeeze(),
                })
        
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def freeze_model_layers(model, freeze_embeddings: bool = True, freeze_transformer: bool = True):
    """Freeze specified model components, keeping only MLM head trainable."""
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then selectively unfreeze MLM head components
    for param in model.vocab_transform.parameters():
        param.requires_grad = True
    for param in model.vocab_layer_norm.parameters():
        param.requires_grad = True
    if hasattr(model, 'vocab_projector'):
        for param in model.vocab_projector.parameters():
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return model

class TypoTrainer(Trainer):
    """Custom trainer with typo-specific metrics."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute masked language modeling loss."""
        # Use the default MLM loss from the model
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for typo correction")
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                       help='Pretrained model name')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Training JSONL file')
    parser.add_argument('--validation_file', type=str,
                       help='Validation JSONL file (optional)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for trained model')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length (128 for ANE compatibility)')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32,
                       help='Per device training batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32,
                       help='Per device evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--save_steps', type=int, default=5000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=500,
                       help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=2000,
                       help='Evaluate every N steps (if validation file provided)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_total_limit', type=int, default=3,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--load_best_model_at_end', action='store_true',
                       help='Load best model at end of training')
    parser.add_argument('--metric_for_best_model', type=str, default='eval_loss',
                       help='Metric to use for best model selection')
    parser.add_argument('--greater_is_better', action='store_true',
                       help='Whether higher metric is better')
    parser.add_argument('--report_to', type=str, default=None,
                       help='Experiment tracking (wandb, tensorboard, etc.)')
    parser.add_argument('--run_name', type=str,
                       help='Experiment run name')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup distributed training
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    
    # Setup logging with run info
    run_name = args.run_name or f"typo-fixer-{args.num_train_epochs}ep"
    
    logger.info("🚀 Starting DistilBERT typo correction training...")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"📊 Run name: {run_name}")
    logger.info(f"🔄 Epochs: {args.num_train_epochs}")
    logger.info(f"📚 Training file: {args.train_file}")
    if args.validation_file:
        logger.info(f"✅ Validation file: {args.validation_file}")
    
    logger.info(f"Loading tokenizer and model: {args.model_name}")
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    model = DistilBertForMaskedLM.from_pretrained(args.model_name)
    
    # Freeze model layers except MLM head
    model = freeze_model_layers(model)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"🔧 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.info(f"🎯 Training efficiency: {100*trainable_params/total_params:.1f}% of parameters trainable")
    
    # Load datasets
    logger.info(f"📖 Loading training data from {args.train_file}")
    train_dataset = TypoDataset(args.train_file, tokenizer, args.max_seq_len)
    
    eval_dataset = None
    if args.validation_file:
        logger.info(f"📖 Loading validation data from {args.validation_file}")
        eval_dataset = TypoDataset(args.validation_file, tokenizer, args.max_seq_len)
    
    logger.info(f"📊 Training dataset size: {len(train_dataset):,}")
    if eval_dataset:
        logger.info(f"📊 Validation dataset size: {len(eval_dataset):,}")
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # Standard MLM masking
    )
    
    # Determine evaluation strategy
    eval_strategy = "steps" if eval_dataset else "no"
    
    # Training arguments
    training_args = TrainingArguments(
        # Output and logging
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        run_name=run_name,
        
        # Training parameters
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        
        # Evaluation
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        
        # Performance optimizations
        fp16=torch.cuda.is_available(),  # Mixed precision on GPU
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # Parallel data loading
        remove_unused_columns=False,  # Important for custom dataset
        
        # Experiment tracking
        report_to=[args.report_to] if args.report_to else [],
        
        # Distributed training
        local_rank=args.local_rank,
        
        # Prediction and evaluation
        prediction_loss_only=True,  # Only compute loss for faster evaluation
        seed=args.seed,
    )
    
    # Create trainer
    trainer = TypoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Print training plan
    total_steps = len(train_dataset) // args.per_device_train_batch_size * args.num_train_epochs
    logger.info(f"\n📋 Training Plan:")
    logger.info(f"   Total steps: {total_steps:,}")
    logger.info(f"   Steps per epoch: {len(train_dataset) // args.per_device_train_batch_size:,}")
    logger.info(f"   Logging every: {args.logging_steps} steps")
    logger.info(f"   Saving every: {args.save_steps} steps")
    if eval_dataset:
        logger.info(f"   Evaluating every: {args.eval_steps} steps")
    
    # Estimate training time
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 Training on GPU: {gpu_name}")
        if "RTX 4090" in gpu_name:
            estimated_time = total_steps * 0.05 / 60  # ~50ms per step estimate
        elif "A100" in gpu_name:
            estimated_time = total_steps * 0.03 / 60  # ~30ms per step estimate
        else:
            estimated_time = total_steps * 0.1 / 60   # Conservative estimate
        logger.info(f"⏱️ Estimated training time: {estimated_time:.1f} minutes")
    else:
        logger.info("💻 Training on CPU (will be slower)")
    
    # Start training
    logger.info("\n🚀 Starting training...")
    import time
    start_time = time.time()
    
    try:
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # Save final model
        logger.info(f"💾 Saving model to {args.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Get final metrics
        final_loss = "N/A"
        final_eval_loss = "N/A"
        
        if trainer.state.log_history:
            # Find last training loss
            for log_entry in reversed(trainer.state.log_history):
                if "train_loss" in log_entry:
                    final_loss = log_entry["train_loss"]
                    break
            
            # Find last evaluation loss
            if eval_dataset:
                for log_entry in reversed(trainer.state.log_history):
                    if "eval_loss" in log_entry:
                        final_eval_loss = log_entry["eval_loss"]
                        break
        
        # Save comprehensive training info
        training_info = {
            "model_name": args.model_name,
            "training_time_minutes": training_time / 60,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "training_examples": len(train_dataset),
            "validation_examples": len(eval_dataset) if eval_dataset else 0,
            "epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "max_seq_len": args.max_seq_len,
            "final_train_loss": final_loss,
            "final_eval_loss": final_eval_loss,
            "total_steps": total_steps,
            "run_name": run_name,
            "gpu_used": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
        
        with open(f"{args.output_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("\n🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {args.output_dir}")
        logger.info(f"📊 Final training loss: {final_loss}")
        if eval_dataset:
            logger.info(f"📊 Final validation loss: {final_eval_loss}")
        logger.info(f"⏱️ Training time: {training_time/60:.1f} minutes")
        
        # Next steps guidance
        if len(train_dataset) >= 50000:
            logger.info(f"\n📋 Next steps:")
            logger.info(f"   1. Validate accuracy with: python src/validate.py --model_dir {args.output_dir}")
            logger.info(f"   2. Upload to HF Hub: huggingface-cli upload {args.output_dir} username/model-name")
            logger.info(f"   3. Convert to ANE: python src/apple_ane_conversion.py --input_model {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
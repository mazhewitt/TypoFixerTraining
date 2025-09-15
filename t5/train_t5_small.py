#!/usr/bin/env python3
"""
T5-small training for typo correction - optimized for M4 MacBook.
Uses lessons learned from T5-tiny optimization.
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback,
)
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class T5SmallTypoDataset(Dataset):
    """Optimized T5-small dataset for typo correction."""
    
    def __init__(self, data_file: str, tokenizer, max_source_length: int = 128, max_target_length: int = 128):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.examples = []
        
        logger.info(f"📖 Loading T5-small training data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading examples"):
                data = json.loads(line.strip())
                
                # Enhanced T5 formatting for better performance
                source_text = f"fix spelling errors: {data['corrupted']}"
                target_text = data['clean']
                
                # Skip examples with no learning signal
                if source_text.replace("fix spelling errors: ", "") == target_text:
                    continue
                
                # Tokenize with longer sequences for T5-small
                source_encoding = self.tokenizer(
                    source_text,
                    truncation=True,
                    padding=False,
                    max_length=self.max_source_length,
                    return_tensors='pt'
                )
                
                target_encoding = self.tokenizer(
                    target_text,
                    truncation=True,
                    padding=False,
                    max_length=self.max_target_length,
                    return_tensors='pt'
                )
                
                # Filter reasonable lengths
                if (source_encoding['input_ids'].shape[1] > 5 and 
                    target_encoding['input_ids'].shape[1] > 2 and
                    source_encoding['input_ids'].shape[1] <= self.max_source_length and
                    target_encoding['input_ids'].shape[1] <= self.max_target_length):
                    self.examples.append({
                        'input_ids': source_encoding['input_ids'].squeeze(),
                        'attention_mask': source_encoding['attention_mask'].squeeze(),
                        'labels': target_encoding['input_ids'].squeeze(),
                    })
        
        logger.info(f"✅ Loaded {len(self.examples)} T5-small training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_t5_small():
    """Train T5-small model with optimized parameters."""
    
    # T5-small optimized parameters (5x larger model)
    args = {
        'model_name': 'google-t5/t5-small',
        'train_file': 'data/enhanced_training_balanced.jsonl',
        'output_dir': 'models/t5-small-typo-fixer',
        'max_source_length': 128,  # Increased for longer sequences
        'max_target_length': 128,
        'batch_size': 16,  # Reduced from 32 due to larger model
        'gradient_accumulation_steps': 2,  # Maintain effective batch size of 32
        'learning_rate': 1e-4,  # Lower learning rate for larger model
        'num_epochs': 4,  # Fewer epochs needed for larger model
        'warmup_ratio': 0.1,
        'save_steps': 500,
        'eval_steps': 250,
        'logging_steps': 50,
        'weight_decay': 0.01,
        'target_accuracy': 0.8,  # Higher target for T5-small
        'early_stopping_patience': 3,
    }
    
    # Convert to namespace
    class Args:
        pass
    
    args_obj = Args()
    for key, value in args.items():
        setattr(args_obj, key, value)
    
    # Set seed
    set_seed(42)
    
    logger.info("🚀 T5-small Typo Correction Training")
    logger.info(f"📁 Output: {args_obj.output_dir}")
    logger.info(f"🎯 Target accuracy: {args_obj.target_accuracy:.1%}")
    logger.info(f"💻 Optimized for M4 MacBook MPS")
    
    # Create output directory
    Path(args_obj.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup model and tokenizer
    logger.info("📥 Loading T5-small model (77M parameters)...")
    tokenizer = T5Tokenizer.from_pretrained(args_obj.model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(args_obj.model_name)
    
    # Move to MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ T5-small loaded: {total_params:,} parameters")
    logger.info(f"🔧 Device: {device}")
    logger.info(f"📊 Model size: {total_params/1e6:.1f}M params (5x larger than T5-tiny)")
    
    # Load dataset
    full_dataset = T5SmallTypoDataset(
        args_obj.train_file, 
        tokenizer, 
        args_obj.max_source_length, 
        args_obj.max_target_length
    )
    
    # Train/val split
    total = len(full_dataset)
    val_size = int(total * 0.1)
    train_size = total - val_size
    
    indices = torch.randperm(total).tolist()
    train_subset = torch.utils.data.Subset(full_dataset, indices[:train_size])
    val_subset = torch.utils.data.Subset(full_dataset, indices[train_size:])
    
    logger.info(f"📊 Train: {len(train_subset):,} examples")
    logger.info(f"📊 Eval: {len(val_subset):,} examples")
    
    # Calculate training parameters
    effective_batch_size = args_obj.batch_size * args_obj.gradient_accumulation_steps
    total_steps = len(train_subset) // effective_batch_size * args_obj.num_epochs
    
    logger.info(f"📊 Effective batch size: {effective_batch_size}")
    logger.info(f"📊 Total training steps: {total_steps:,}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=args_obj.max_target_length,
        pad_to_multiple_of=8,
    )
    
    # Training arguments optimized for T5-small
    training_args = TrainingArguments(
        # Basic setup
        output_dir=args_obj.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args_obj.num_epochs,
        
        # Batch sizes (optimized for T5-small memory requirements)
        per_device_train_batch_size=args_obj.batch_size,
        per_device_eval_batch_size=args_obj.batch_size,
        gradient_accumulation_steps=args_obj.gradient_accumulation_steps,
        
        # Learning parameters (conservative for larger model)
        learning_rate=args_obj.learning_rate,
        weight_decay=args_obj.weight_decay,
        warmup_ratio=args_obj.warmup_ratio,
        lr_scheduler_type="cosine",
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=args_obj.eval_steps,
        save_strategy="steps", 
        save_steps=args_obj.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # MPS optimizations
        fp16=False,  # Keep FP32 for MPS stability
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        
        # Logging
        logging_dir=f"{args_obj.output_dir}/logs",
        logging_steps=args_obj.logging_steps,
        logging_first_step=True,
        
        # Optimizations
        remove_unused_columns=False,
        prediction_loss_only=True,
        include_inputs_for_metrics=False,
        save_safetensors=True,
        
        # Disable wandb
        report_to=[],
        run_name=f"t5-small-typo-{args_obj.num_epochs}ep",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args_obj.early_stopping_patience)],
    )
    
    # Training info
    logger.info(f"\n📋 T5-small Training Plan:")
    logger.info(f"   Model: T5-small ({total_params:,} params)")
    logger.info(f"   Device: {device}")
    logger.info(f"   Batch size: {args_obj.batch_size} x {args_obj.gradient_accumulation_steps} = {effective_batch_size}")
    logger.info(f"   Learning rate: {args_obj.learning_rate}")
    logger.info(f"   Epochs: {args_obj.num_epochs}")
    logger.info(f"   Sequence length: {args_obj.max_source_length} → {args_obj.max_target_length}")
    logger.info(f"   Expected improvement over T5-tiny: Significant")
    
    # Start training
    logger.info("\n🚀 Starting T5-small training...")
    start_time = time.time()
    
    try:
        # Train the model
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ T5-small training completed in {training_time/60:.1f} minutes")
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(args_obj.output_dir)
        
        # Test the model
        logger.info("🧪 Testing T5-small model...")
        test_cases = [
            "I beleive this is correct",
            "The qucik brown fox jumps",
            "She recieved her degre yesterday",
            "Th eonly survivor sare alive"
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            prompt = f"fix spelling errors: {test_input}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=128, 
                    num_beams=3,  # Higher beam search for better quality
                    do_sample=False,
                    early_stopping=True
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"   {i}. '{test_input}' → '{result}'")
        
        # Save training info
        training_info = {
            "model_name": args_obj.model_name,
            "model_size": "T5-small (77M parameters)",
            "training_type": "optimized_large_model",
            "training_time_minutes": training_time / 60,
            "total_parameters": total_params,
            "training_examples": len(train_subset),
            "validation_examples": len(val_subset),
            "epochs": args_obj.num_epochs,
            "effective_batch_size": effective_batch_size,
            "learning_rate": args_obj.learning_rate,
            "max_sequence_length": args_obj.max_source_length,
            "improvements_over_tiny": [
                "5x more parameters (77M vs 15.6M)",
                "longer sequences (128 vs 64)",
                "better capacity for complex patterns",
                "expected higher accuracy"
            ],
            "total_steps": total_steps,
            "device": device,
        }
        
        with open(f"{args_obj.output_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"\n✅ T5-small training completed successfully!")
        logger.info(f"⏱️ Training time: {training_time/60:.1f} minutes")
        logger.info(f"💾 Model saved to: {args_obj.output_dir}")
        logger.info(f"🎯 Expected: Significantly better typo correction than T5-tiny")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ T5-small training failed: {e}")
        return False

if __name__ == "__main__":
    success = train_t5_small()
    if success:
        print("\n🎉 T5-small training completed! This should show much better typo correction.")
    else:
        print("\n❌ T5-small training failed. Check the logs for details.")
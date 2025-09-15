#!/usr/bin/env python3
"""
Optimized T5 training with higher GPU utilization and better learning parameters.
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

class OptimizedT5TypoDataset(Dataset):
    """Optimized T5 dataset with better preprocessing."""
    
    def __init__(self, data_file: str, tokenizer, max_source_length: int = 64, max_target_length: int = 64):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.examples = []
        
        logger.info(f"📖 Loading optimized T5 training data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading examples"):
                data = json.loads(line.strip())
                
                # More explicit T5 formatting
                source_text = f"fix spelling errors: {data['corrupted']}"
                target_text = data['clean']
                
                # Skip examples that are too similar (no learning signal)
                if source_text.replace("fix spelling errors: ", "") == target_text:
                    continue
                
                # Tokenize with proper length limits
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
                
                # Only add if both source and target are reasonable length
                if (source_encoding['input_ids'].shape[1] > 5 and 
                    target_encoding['input_ids'].shape[1] > 2):
                    self.examples.append({
                        'input_ids': source_encoding['input_ids'].squeeze(),
                        'attention_mask': source_encoding['attention_mask'].squeeze(),
                        'labels': target_encoding['input_ids'].squeeze(),
                    })
        
        logger.info(f"✅ Loaded {len(self.examples)} optimized training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def optimized_training():
    """Run optimized T5 training with better GPU utilization."""
    
    # Optimized parameters based on analysis
    args = {
        'model_name': 'google/t5-efficient-tiny',
        'train_file': 'data/enhanced_training_balanced.jsonl',
        'output_dir': 'models/t5-typo-fixer-optimized',
        'max_source_length': 64,
        'max_target_length': 64,
        'batch_size': 32,  # Doubled from 16
        'gradient_accumulation_steps': 1,  # Reduced since batch size increased
        'learning_rate': 3e-4,  # Higher for T5
        'num_epochs': 6,  # More epochs
        'warmup_ratio': 0.1,
        'save_steps': 250,
        'eval_steps': 250,
        'logging_steps': 25,
        'weight_decay': 0.01,
        'target_accuracy': 0.7,  # More realistic
        'early_stopping_patience': 3,
    }
    
    # Convert to namespace for compatibility
    class Args:
        pass
    
    args_obj = Args()
    for key, value in args.items():
        setattr(args_obj, key, value)
    
    # Set seed
    set_seed(42)
    
    logger.info("🚀 OPTIMIZED T5-efficient-tiny Training")
    logger.info(f"📁 Output: {args_obj.output_dir}")
    logger.info(f"🎯 Target accuracy: {args_obj.target_accuracy:.1%}")
    logger.info(f"💻 Optimized for maximum M4 MacBook utilization")
    
    # Create output directory
    Path(args_obj.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args_obj.model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(args_obj.model_name)
    
    # Move to MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    
    logger.info(f"✅ T5 model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"🔧 Device: {device}")
    
    # Load optimized dataset
    full_dataset = OptimizedT5TypoDataset(
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
    
    # Optimized data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=args_obj.max_target_length,
        pad_to_multiple_of=8,  # Optimize for tensor cores
    )
    
    # Optimized training arguments
    training_args = TrainingArguments(
        # Basic setup
        output_dir=args_obj.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args_obj.num_epochs,
        
        # OPTIMIZED batch sizes for better GPU utilization
        per_device_train_batch_size=args_obj.batch_size,
        per_device_eval_batch_size=args_obj.batch_size,
        gradient_accumulation_steps=args_obj.gradient_accumulation_steps,
        
        # OPTIMIZED learning parameters
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
        
        # OPTIMIZED memory settings for MPS
        fp16=False,  # Keep FP32 for MPS stability
        dataloader_pin_memory=False,  # Disable for MPS compatibility
        dataloader_num_workers=0,  # Disable multiprocessing for MPS
        
        # Logging
        logging_dir=f"{args_obj.output_dir}/logs",
        logging_steps=args_obj.logging_steps,
        logging_first_step=True,
        
        # Other optimizations
        remove_unused_columns=False,
        prediction_loss_only=True,
        include_inputs_for_metrics=False,
        save_safetensors=True,
        
        # Disable wandb
        report_to=[],
        run_name=f"t5-tiny-optimized-{args_obj.num_epochs}ep",
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
    logger.info(f"\n📋 OPTIMIZED Training Plan:")
    logger.info(f"   Device: {device}")
    logger.info(f"   Batch size: {args_obj.batch_size} (DOUBLED)")
    logger.info(f"   Learning rate: {args_obj.learning_rate} (INCREASED)")
    logger.info(f"   Epochs: {args_obj.num_epochs} (DOUBLED)")
    logger.info(f"   GPU utilization: Target 80%+")
    logger.info(f"   Memory optimization: Enabled")
    
    # Start training
    logger.info("\n🚀 Starting OPTIMIZED training...")
    start_time = time.time()
    
    try:
        # Train the model
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Optimized training completed in {training_time/60:.1f} minutes")
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(args_obj.output_dir)
        
        # Test the optimized model
        logger.info("🧪 Testing optimized model...")
        test_cases = [
            "I beleive this is correct",
            "The qucik brown fox jumps",
            "She recieved her degre yesterday"
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            prompt = f"fix spelling errors: {test_input}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=64, 
                    num_beams=2,  # Slight beam search
                    do_sample=False,
                    early_stopping=True
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"   {i}. '{test_input}' → '{result}'")
        
        # Save training info
        training_info = {
            "model_name": args_obj.model_name,
            "training_type": "optimized",
            "training_time_minutes": training_time / 60,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "training_examples": len(train_subset),
            "validation_examples": len(val_subset),
            "epochs": args_obj.num_epochs,
            "batch_size": effective_batch_size,
            "learning_rate": args_obj.learning_rate,
            "optimizations": ["doubled_batch_size", "increased_lr", "more_epochs", "better_preprocessing"],
            "total_steps": total_steps,
            "device": device,
        }
        
        with open(f"{args_obj.output_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"\n✅ Optimized training completed successfully!")
        logger.info(f"⏱️ Training time: {training_time/60:.1f} minutes")
        logger.info(f"💾 Model saved to: {args_obj.output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimized training failed: {e}")
        return False

if __name__ == "__main__":
    success = optimized_training()
    if success:
        print("\n🎉 Optimized training completed! The model should show significantly better performance.")
    else:
        print("\n❌ Optimized training failed. Check the logs for details.")
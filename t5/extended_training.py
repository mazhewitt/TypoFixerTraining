#!/usr/bin/env python3
"""
Extended training for T5-small to push beyond current 40% accuracy.
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
    """T5-small dataset for extended training."""
    
    def __init__(self, data_file: str, tokenizer, max_source_length: int = 128, max_target_length: int = 128):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.examples = []
        
        logger.info(f"📖 Loading extended training data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading examples"):
                data = json.loads(line.strip())
                
                # Enhanced T5 formatting
                source_text = f"fix spelling errors: {data['corrupted']}"
                target_text = data['clean']
                
                # Skip examples with no learning signal
                if source_text.replace("fix spelling errors: ", "") == target_text:
                    continue
                
                # Tokenize
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
        
        logger.info(f"✅ Loaded {len(self.examples)} examples for extended training")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def extended_training():
    """Extended training to push T5-small beyond 40% accuracy."""
    
    # Extended training parameters
    args = {
        'model_path': 'models/t5-small-typo-fixer',  # Resume from trained model
        'train_file': 'data/enhanced_training_balanced.jsonl',
        'output_dir': 'models/t5-small-typo-fixer-extended',
        'max_source_length': 128,
        'max_target_length': 128,
        'batch_size': 16,
        'gradient_accumulation_steps': 2,
        'learning_rate': 5e-5,  # Reduced learning rate for fine-tuning
        'num_epochs': 3,  # Additional epochs
        'warmup_ratio': 0.05,  # Less warmup for continued training
        'save_steps': 250,
        'eval_steps': 125,  # More frequent evaluation
        'logging_steps': 25,
        'weight_decay': 0.01,
        'target_accuracy': 0.6,  # Higher target
        'early_stopping_patience': 4,  # More patience
    }
    
    # Convert to namespace
    class Args:
        pass
    
    args_obj = Args()
    for key, value in args.items():
        setattr(args_obj, key, value)
    
    # Set seed
    set_seed(42)
    
    logger.info("🚀 T5-small Extended Training")
    logger.info(f"📁 Resuming from: {args_obj.model_path}")
    logger.info(f"📁 Output: {args_obj.output_dir}")
    logger.info(f"🎯 Target accuracy: {args_obj.target_accuracy:.1%}")
    logger.info(f"💡 Strategy: Extended fine-tuning to reach 60%+ accuracy")
    
    # Create output directory
    Path(args_obj.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load previously trained model
    logger.info("📥 Loading previously trained T5-small model...")
    tokenizer = T5Tokenizer.from_pretrained(args_obj.model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(args_obj.model_path)
    
    # Move to MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ T5-small resumed: {total_params:,} parameters")
    logger.info(f"🔧 Device: {device}")
    
    # Load dataset
    full_dataset = T5SmallTypoDataset(
        args_obj.train_file, 
        tokenizer, 
        args_obj.max_source_length, 
        args_obj.max_target_length
    )
    
    # Train/val split (same split as before for consistency)
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
    logger.info(f"📊 Extended training steps: {total_steps:,}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=args_obj.max_target_length,
        pad_to_multiple_of=8,
    )
    
    # Extended training arguments
    training_args = TrainingArguments(
        # Basic setup
        output_dir=args_obj.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args_obj.num_epochs,
        
        # Batch sizes
        per_device_train_batch_size=args_obj.batch_size,
        per_device_eval_batch_size=args_obj.batch_size,
        gradient_accumulation_steps=args_obj.gradient_accumulation_steps,
        
        # Learning parameters (reduced for continued training)
        learning_rate=args_obj.learning_rate,
        weight_decay=args_obj.weight_decay,
        warmup_ratio=args_obj.warmup_ratio,
        lr_scheduler_type="cosine",
        
        # Evaluation and saving (more frequent)
        eval_strategy="steps",
        eval_steps=args_obj.eval_steps,
        save_strategy="steps", 
        save_steps=args_obj.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # MPS optimizations
        fp16=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        
        # Logging (more detailed)
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
        run_name=f"t5-small-extended-{args_obj.num_epochs}ep",
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
    logger.info(f"\n📋 Extended Training Plan:")
    logger.info(f"   Starting point: 40% accuracy model")
    logger.info(f"   Device: {device}")
    logger.info(f"   Batch size: {effective_batch_size}")
    logger.info(f"   Learning rate: {args_obj.learning_rate} (reduced)")
    logger.info(f"   Additional epochs: {args_obj.num_epochs}")
    logger.info(f"   Target: 60%+ accuracy")
    logger.info(f"   Early stopping: {args_obj.early_stopping_patience} patience")
    
    # Start extended training
    logger.info("\n🚀 Starting extended training...")
    start_time = time.time()
    
    try:
        # Train the model
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Extended training completed in {training_time/60:.1f} minutes")
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(args_obj.output_dir)
        
        # Test the extended model
        logger.info("🧪 Testing extended model...")
        test_cases = [
            "I beleive this is correct",
            "The qucik brown fox jumps",
            "She recieved her degre yesterday",
            "Th eonly survivor sare alive",
            "These had to be burie din mass graves"
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            prompt = f"fix spelling errors: {test_input}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=128, 
                    num_beams=3,
                    do_sample=False,
                    early_stopping=True
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"   {i}. '{test_input}' → '{result}'")
        
        # Save extended training info
        training_info = {
            "model_name": "T5-small Extended",
            "base_model_path": args_obj.model_path,
            "training_type": "extended_fine_tuning",
            "extended_training_time_minutes": training_time / 60,
            "total_parameters": total_params,
            "training_examples": len(train_subset),
            "validation_examples": len(val_subset),
            "additional_epochs": args_obj.num_epochs,
            "effective_batch_size": effective_batch_size,
            "learning_rate": args_obj.learning_rate,
            "starting_accuracy": "40%",
            "target_accuracy": f"{args_obj.target_accuracy:.0%}",
            "improvements": [
                "reduced learning rate for stability",
                "more frequent evaluation",
                "extended early stopping patience",
                "continued from best previous checkpoint"
            ],
            "total_steps": total_steps,
            "device": device,
        }
        
        with open(f"{args_obj.output_dir}/extended_training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"\n✅ Extended training completed successfully!")
        logger.info(f"⏱️ Extended training time: {training_time/60:.1f} minutes")
        logger.info(f"💾 Model saved to: {args_obj.output_dir}")
        logger.info(f"🎯 Expected: 50-70% accuracy (significant improvement over 40%)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Extended training failed: {e}")
        return False

if __name__ == "__main__":
    success = extended_training()
    if success:
        print("\n🎉 Extended training completed! Model should now exceed 50% accuracy.")
    else:
        print("\n❌ Extended training failed. Check the logs for details.")
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
    parser = argparse.ArgumentParser(description="Train DistilBERT typo correction model")
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                       help='Base model name or path')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Path to training JSONL file')
    parser.add_argument('--validation_file', type=str, default=None,
                       help='Path to validation JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for model checkpoints')
    parser.add_argument('--max_seq_len', type=int, default=64,
                       help='Maximum sequence length')
    parser.add_argument('--per_device_batch_size', type=int, default=32,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps')
    parser.add_argument('--save_steps', type=int, default=5000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=5000,
                       help='Evaluation every N steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Logging every N steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup distributed training
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    model = DistilBertForMaskedLM.from_pretrained(args.model_name)
    
    # Freeze layers except MLM head
    model = freeze_model_layers(model)
    
    # Load datasets
    train_dataset = TypoDataset(args.train_file, tokenizer, args.max_seq_len)
    
    eval_dataset = None
    if args.validation_file:
        eval_dataset = TypoDataset(args.validation_file, tokenizer, args.max_seq_len)
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # Mask 15% of tokens during training
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        prediction_loss_only=False,
        dataloader_num_workers=4,
        local_rank=args.local_rank,
        report_to=None,  # Disable wandb/tensorboard
        seed=args.seed,
    )
    
    # Initialize trainer
    trainer = TypoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training info
    info = {
        "model_name": args.model_name,
        "max_seq_len": args.max_seq_len,
        "num_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "train_samples": len(train_dataset),
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Model tweaks script for DistilBERT typo correction optimization.
Implements selective layer freezing and hyperparameter adjustments to reach 90% accuracy.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    DistilBertForMaskedLM, 
    DistilBertTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTweaker:
    def __init__(self, model_dir: str):
        """Initialize with existing trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {model_dir}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForMaskedLM.from_pretrained(model_dir)
        self.model.to(self.device)
        
        logger.info(f"Model loaded on {self.device}")
        
    def freeze_embeddings(self):
        """Freeze embedding layers to stabilize learned representations."""
        logger.info("🧊 Freezing embedding layers...")
        
        # Freeze word embeddings
        for param in self.model.distilbert.embeddings.parameters():
            param.requires_grad = False
            
        # Count frozen parameters
        frozen_params = sum(1 for p in self.model.distilbert.embeddings.parameters() if not p.requires_grad)
        total_params = sum(1 for p in self.model.parameters())
        
        logger.info(f"   Frozen {frozen_params} embedding parameters")
        logger.info(f"   Trainable parameters: {total_params - frozen_params}/{total_params}")
        
    def freeze_lower_layers(self, num_layers_to_freeze: int = 2):
        """Freeze lower transformer layers to focus on higher-level patterns."""
        logger.info(f"🧊 Freezing bottom {num_layers_to_freeze} transformer layers...")
        
        frozen_count = 0
        for i in range(min(num_layers_to_freeze, len(self.model.distilbert.transformer.layer))):
            for param in self.model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False
                frozen_count += 1
                
        logger.info(f"   Frozen {frozen_count} parameters in {num_layers_to_freeze} layers")
        
    def apply_gradient_scaling(self, head_lr_multiplier: float = 2.0):
        """Apply different learning rates to different parts of the model."""
        logger.info(f"📈 Setting up gradient scaling (head LR × {head_lr_multiplier})...")
        
        # Create parameter groups with different learning rates
        head_params = []
        transformer_params = []
        embedding_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'vocab_transform' in name or 'vocab_layer_norm' in name or 'vocab_projector' in name:
                    head_params.append(param)
                elif 'embeddings' in name:
                    embedding_params.append(param)
                else:
                    transformer_params.append(param)
        
        self.param_groups = [
            {'params': head_params, 'lr_multiplier': head_lr_multiplier, 'name': 'head'},
            {'params': transformer_params, 'lr_multiplier': 1.0, 'name': 'transformer'},
            {'params': embedding_params, 'lr_multiplier': 0.5, 'name': 'embeddings'}
        ]
        
        logger.info(f"   Head parameters: {len(head_params)}")
        logger.info(f"   Transformer parameters: {len(transformer_params)}")
        logger.info(f"   Embedding parameters: {len(embedding_params)}")
        
    def get_optimized_training_args(self, 
                                  output_dir: str,
                                  base_lr: float = 1e-5,
                                  num_epochs: int = 3,
                                  batch_size: int = 32) -> TrainingArguments:
        """Get optimized training arguments for fine-tuning."""
        
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            
            # Optimized learning schedule
            learning_rate=base_lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            
            # Regularization
            weight_decay=0.05,  # Increased for better generalization
            max_grad_norm=0.5,  # Gradient clipping
            
            # Optimization
            adam_beta1=0.9,
            adam_beta2=0.98,  # Slightly higher beta2 for stability
            adam_epsilon=1e-6,
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            
            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            logging_strategy="steps",
            
            # Performance
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            fp16=torch.cuda.is_available(),  # Mixed precision if available
            
            # Early stopping patience
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Prevent overfitting
            report_to=None  # Disable wandb/tensorboard for cleaner output
        )

def load_training_data(file_path: str, max_samples: Optional[int] = None) -> Dataset:
    """Load training data from JSONL file."""
    logger.info(f"Loading training data from {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            example = json.loads(line.strip())
            data.append({
                'text': example['corrupted'],
                'labels': example['clean']
            })
    
    logger.info(f"Loaded {len(data)} training examples")
    return Dataset.from_list(data)

def create_mlm_dataset(dataset: Dataset, tokenizer: DistilBertTokenizer, max_length: int = 128) -> Dataset:
    """Convert dataset to MLM format."""
    
    def tokenize_function(examples):
        # Tokenize corrupted text (input)
        corrupted_tokens = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Tokenize clean text (labels)
        clean_tokens = tokenizer(
            examples['labels'],
            truncation=True,
            padding='max_length', 
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': corrupted_tokens['input_ids'].squeeze(),
            'attention_mask': corrupted_tokens['attention_mask'].squeeze(),
            'labels': clean_tokens['input_ids'].squeeze()
        }
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description="Apply model tweaks for DistilBERT typo correction")
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing base trained model')
    parser.add_argument('--train_file', type=str, required=True,
                       help='JSONL training file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for tweaked model')
    
    # Freezing options
    parser.add_argument('--freeze_embeddings', action='store_true',
                       help='Freeze embedding layers')
    parser.add_argument('--freeze_layers', type=int, default=0,
                       help='Number of bottom transformer layers to freeze (0-6)')
    
    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Base learning rate')
    parser.add_argument('--head_lr_multiplier', type=float, default=2.0,
                       help='Learning rate multiplier for classification head')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--max_samples', type=int,
                       help='Maximum training samples (for testing)')
    parser.add_argument('--max_length', type=int, default=96,
                       help='Maximum sequence length')
    
    # Optimization presets
    parser.add_argument('--preset', type=str, choices=['conservative', 'aggressive', 'balanced'],
                       default='balanced', help='Optimization preset')
    
    args = parser.parse_args()
    
    logger.info("🔧 Starting model tweaks for improved accuracy...")
    
    # Apply presets
    if args.preset == 'conservative':
        args.freeze_embeddings = True
        args.freeze_layers = 1
        args.learning_rate = 5e-6
        args.head_lr_multiplier = 1.5
        args.num_epochs = 2
        logger.info("📊 Using CONSERVATIVE preset: minimal changes, stable training")
        
    elif args.preset == 'aggressive':
        args.freeze_embeddings = False
        args.freeze_layers = 0
        args.learning_rate = 2e-5
        args.head_lr_multiplier = 3.0
        args.num_epochs = 5
        logger.info("📊 Using AGGRESSIVE preset: full fine-tuning, higher learning rates")
        
    else:  # balanced
        args.freeze_embeddings = True
        args.freeze_layers = 2
        args.learning_rate = 1e-5
        args.head_lr_multiplier = 2.0
        args.num_epochs = 3
        logger.info("📊 Using BALANCED preset: selective freezing, moderate learning rates")
    
    # Initialize tweaker
    tweaker = ModelTweaker(args.model_dir)
    
    # Apply freezing strategies
    if args.freeze_embeddings:
        tweaker.freeze_embeddings()
        
    if args.freeze_layers > 0:
        tweaker.freeze_lower_layers(args.freeze_layers)
    
    # Setup gradient scaling
    tweaker.apply_gradient_scaling(args.head_lr_multiplier)
    
    # Load and prepare data
    dataset = load_training_data(args.train_file, args.max_samples)
    tokenized_dataset = create_mlm_dataset(dataset, tweaker.tokenizer, args.max_length)
    
    # Split into train/eval (90/10)
    train_size = int(0.9 * len(tokenized_dataset))
    eval_size = len(tokenized_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        tokenized_dataset, [train_size, eval_size]
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Get optimized training arguments
    training_args = tweaker.get_optimized_training_args(
        output_dir=args.output_dir,
        base_lr=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tweaker.tokenizer,
        mlm=False,  # We're doing direct supervision, not random masking
        return_tensors="pt"
    )
    
    # Custom optimizer with parameter groups
    def optimizer_factory(model):
        """Create optimizer with different learning rates for different parts."""
        param_groups = []
        for group in tweaker.param_groups:
            if group['params']:  # Only add non-empty groups
                param_groups.append({
                    'params': group['params'],
                    'lr': args.learning_rate * group['lr_multiplier']
                })
        
        if not param_groups:
            # Fallback to all parameters
            param_groups = [{'params': model.parameters(), 'lr': args.learning_rate}]
            
        return torch.optim.AdamW(
            param_groups,
            lr=args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.05
        )
    
    # Initialize trainer
    trainer = Trainer(
        model=tweaker.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer_factory(tweaker.model), None)  # (optimizer, scheduler)
    )
    
    # Train model
    logger.info("🚀 Starting fine-tuning with model tweaks...")
    
    # Log training configuration
    logger.info("🔧 Training Configuration:")
    logger.info(f"   Base learning rate: {args.learning_rate}")
    logger.info(f"   Head LR multiplier: {args.head_lr_multiplier}")
    logger.info(f"   Epochs: {args.num_epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Embeddings frozen: {args.freeze_embeddings}")
    logger.info(f"   Layers frozen: {args.freeze_layers}")
    
    # Train
    train_result = trainer.train()
    
    # Save the tweaked model
    trainer.save_model(args.output_dir)
    tweaker.tokenizer.save_pretrained(args.output_dir)
    
    # Save training configuration
    config = {
        'base_model': args.model_dir,
        'preset': args.preset,
        'freeze_embeddings': args.freeze_embeddings,
        'freeze_layers': args.freeze_layers,
        'learning_rate': args.learning_rate,
        'head_lr_multiplier': args.head_lr_multiplier,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'training_loss': train_result.training_loss,
        'train_samples': len(train_dataset)
    }
    
    with open(os.path.join(args.output_dir, 'tweaks_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("✅ Model tweaks completed!")
    logger.info(f"📁 Tweaked model saved to: {args.output_dir}")
    logger.info(f"📊 Final training loss: {train_result.training_loss:.4f}")
    
    # Recommendations for next steps
    logger.info("\n💡 Next steps:")
    logger.info("   1. Validate with: python src/validate_mlm.py --model_dir " + args.output_dir)
    logger.info("   2. Run diagnostics: python src/diagnostic_analysis.py --model_dir " + args.output_dir)
    logger.info("   3. If accuracy < 85%, try 'aggressive' preset")
    logger.info("   4. If accuracy > 90%, ready for ANE conversion!")

if __name__ == "__main__":
    main()
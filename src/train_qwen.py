#!/usr/bin/env python3
"""
Training script for Qwen 0.6B typo correction using text-to-text fine-tuning.
Optimized for >90% sentence accuracy and ANE deployment compatibility.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenTypoDataset(Dataset):
    """Dataset for text-to-text typo correction training with Qwen."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"📖 Loading data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading examples"):
                data = json.loads(line.strip())
                
                # Format as text-to-text with natural prompt
                prompt = f"Correct the typos: {data['corrupted']}"
                target = data['clean']
                
                # Create training text: prompt + target
                full_text = f"{prompt}\n{target}"
                
                # Tokenize with proper padding
                encoding = self.tokenizer(
                    full_text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Create labels (copy of input_ids)
                labels = encoding['input_ids'].clone()
                
                # Find where the target starts (after the newline)
                prompt_tokens = self.tokenizer(
                    prompt + "\n",
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'  
                )
                prompt_length = min(prompt_tokens['input_ids'].shape[-1], labels.shape[-1])
                
                # Mask the prompt tokens in labels (-100 = ignore in loss)
                labels[:, :prompt_length] = -100
                
                self.examples.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': labels.squeeze(),
                    'prompt': prompt,
                    'target': target,
                    'corrupted': data['corrupted'],
                    'clean': data['clean'],
                    'complexity': data.get('complexity', 'unknown'),
                    'source': data.get('source', 'unknown')
                })
        
        logger.info(f"✅ Loaded {len(self.examples)} training examples")
        
        # Show data distribution
        complexity_counts = {}
        source_counts = {}
        for ex in self.examples:
            complexity_counts[ex['complexity']] = complexity_counts.get(ex['complexity'], 0) + 1
            source_counts[ex['source']] = source_counts.get(ex['source'], 0) + 1
        
        logger.info(f"📊 Data distribution by complexity: {complexity_counts}")
        logger.info(f"📊 Data distribution by source: {source_counts}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.examples[idx]['input_ids'],
            'attention_mask': self.examples[idx]['attention_mask'],
            'labels': self.examples[idx]['labels']
        }

class QwenTypoTrainer(Trainer):
    """Custom trainer with typo-specific metrics and evaluation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_examples = []
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute causal language modeling loss."""
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with accuracy metrics."""
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add sentence-level accuracy evaluation on a sample
        if eval_dataset and len(eval_dataset) > 0:
            accuracy = self._compute_sentence_accuracy(eval_dataset)
            eval_results[f"{metric_key_prefix}_sentence_accuracy"] = accuracy
            logger.info(f"📊 Sentence accuracy: {accuracy:.1%}")
        
        return eval_results
    
    def _compute_sentence_accuracy(self, dataset, num_samples=100):
        """Compute sentence-level accuracy on a sample of examples."""
        self.model.eval()
        
        # Sample examples for evaluation
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        correct = 0
        total = 0
        
        with torch.no_grad():
            for idx in indices:
                example = dataset.examples[idx]  # Access original example
                prompt = example['prompt']
                expected = example['target']
                
                # Generate prediction
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256
                ).to(self.model.device)
                
                # Generate with controlled parameters
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,  # Use greedy decoding for consistency
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode prediction (skip the prompt)
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[-1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Check if prediction matches expected (case insensitive, whitespace normalized)
                pred_normalized = ' '.join(generated_text.lower().split())
                expected_normalized = ' '.join(expected.lower().split())
                
                if pred_normalized == expected_normalized:
                    correct += 1
                elif total < 5:  # Show first few examples for debugging
                    logger.info(f"❌ Mismatch #{total+1}:")
                    logger.info(f"   Prompt: {prompt}")
                    logger.info(f"   Expected: {expected}")
                    logger.info(f"   Generated: {generated_text}")
                
                total += 1
        
        self.model.train()
        return correct / total if total > 0 else 0.0

def setup_model_and_tokenizer(model_name: str, max_length: int = 256):
    """Setup Qwen model and tokenizer with proper configuration."""
    logger.info(f"🤖 Loading Qwen model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        pad_token='<|endoftext|>',
        eos_token='<|endoftext|>',
        model_max_length=max_length
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Resize embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer

def create_validation_split(train_dataset, validation_ratio=0.1):
    """Create validation split from training data."""
    total_size = len(train_dataset)
    val_size = int(total_size * validation_ratio)
    train_size = total_size - val_size
    
    # Random split
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    # Copy examples for validation access
    val_subset.examples = [train_dataset.examples[i] for i in val_indices]
    
    logger.info(f"📊 Created train/val split: {len(train_subset):,} / {len(val_subset):,}")
    
    return train_subset, val_subset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 0.6B for typo correction")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B',
                       help='Qwen model name or path')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Training JSONL file')
    parser.add_argument('--validation_file', type=str,
                       help='Validation JSONL file (optional - will split from train)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for trained model')
    parser.add_argument('--max_seq_len', type=int, default=256,
                       help='Maximum sequence length (256 for ANE optimization)')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                       help='Per device training batch size (smaller for memory)')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4,
                       help='Per device evaluation batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps (effective batch size = batch_size * accum_steps)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate (higher for fine-tuning)')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                       help='Warmup ratio (3% of training)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_total_limit', type=int, default=3,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--target_accuracy', type=float, default=0.9,
                       help='Target sentence accuracy (90%)')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Early stopping patience (evaluations)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    logger.info("🚀 Starting Qwen typo correction fine-tuning...")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"🎯 Target accuracy: {args.target_accuracy:.1%}")
    logger.info(f"🔄 Epochs: {args.num_train_epochs}")
    logger.info(f"📚 Training file: {args.train_file}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.max_seq_len)
    
    # Load training dataset
    train_dataset = QwenTypoDataset(args.train_file, tokenizer, args.max_seq_len)
    
    # Create or load validation dataset
    if args.validation_file:
        logger.info(f"📖 Loading validation data from {args.validation_file}")
        eval_dataset = QwenTypoDataset(args.validation_file, tokenizer, args.max_seq_len)
    else:
        logger.info("📊 Creating validation split from training data")
        train_dataset, eval_dataset = create_validation_split(train_dataset, 0.1)
    
    logger.info(f"📊 Training dataset size: {len(train_dataset):,}")
    logger.info(f"📊 Validation dataset size: {len(eval_dataset):,}")
    
    # Calculate effective batch size
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    logger.info(f"📊 Effective batch size: {effective_batch_size}")
    
    # Training arguments optimized for accuracy
    training_args = TrainingArguments(
        # Output and logging
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        run_name=f"qwen-typo-fixer-{args.num_train_epochs}ep",
        
        # Training parameters
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_sentence_accuracy",
        greater_is_better=True,
        
        # Performance optimizations
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Save memory
        
        # Other settings
        seed=args.seed,
        report_to=[],  # Disable wandb by default
        prediction_loss_only=False,  # We want all metrics
    )
    
    # Simple data collator since we pre-pad in dataset
    from transformers import default_data_collator
    data_collator = default_data_collator
    
    # Create trainer
    trainer = QwenTypoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Calculate training info
    total_steps = len(train_dataset) // effective_batch_size * args.num_train_epochs
    
    logger.info(f"\n📋 Training Plan:")
    logger.info(f"   Total steps: {total_steps:,}")
    logger.info(f"   Steps per epoch: {len(train_dataset) // effective_batch_size:,}")
    logger.info(f"   Logging every: {args.logging_steps} steps")
    logger.info(f"   Saving every: {args.save_steps} steps")
    logger.info(f"   Evaluating every: {args.eval_steps} steps")
    
    # Check device
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 Training on GPU: {gpu_name}")
    else:
        logger.info("💻 Training on CPU (will be slower)")
    
    # Start training
    logger.info("\n🚀 Starting training...")
    start_time = time.time()
    
    try:
        # Train the model
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # Final evaluation
        logger.info("📊 Running final evaluation...")
        final_metrics = trainer.evaluate()
        final_accuracy = final_metrics.get('eval_sentence_accuracy', 0.0)
        
        logger.info(f"🎯 Final sentence accuracy: {final_accuracy:.1%}")
        
        if final_accuracy >= args.target_accuracy:
            logger.info(f"🎉 Target accuracy of {args.target_accuracy:.1%} achieved!")
        else:
            logger.warning(f"⚠️ Target accuracy of {args.target_accuracy:.1%} not reached. Consider training longer or adjusting parameters.")
        
        # Save final model
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
            "epochs": args.num_train_epochs,
            "effective_batch_size": effective_batch_size,
            "learning_rate": args.learning_rate,
            "max_seq_len": args.max_seq_len,
            "final_sentence_accuracy": final_accuracy,
            "target_accuracy": args.target_accuracy,
            "target_achieved": final_accuracy >= args.target_accuracy,
            "total_steps": total_steps,
            "gpu_used": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
        
        with open(f"{args.output_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("\n🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {args.output_dir}")
        logger.info(f"📊 Final sentence accuracy: {final_accuracy:.1%}")
        logger.info(f"⏱️ Training time: {training_time/60:.1f} minutes")
        
        # Next steps
        logger.info(f"\n📋 Next steps:")
        logger.info(f"   1. Test the model: python src/test_qwen.py --model_dir {args.output_dir}")
        logger.info(f"   2. Convert to ANE: ./anemll/utils/convert_model.sh --model {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
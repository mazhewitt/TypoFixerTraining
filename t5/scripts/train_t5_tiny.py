#!/usr/bin/env python3
"""
T5-efficient-tiny training script for typo correction.
Optimized for M4 MacBook training with minimal memory usage.
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

class T5TypoDataset(Dataset):
    """T5 seq2seq dataset for typo correction."""
    
    def __init__(self, data_file: str, tokenizer, max_source_length: int = 64, max_target_length: int = 64):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.examples = []
        
        logger.info(f"📖 Loading T5 training data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading examples"):
                data = json.loads(line.strip())
                
                # T5 format: "correct typos: [corrupted text]" -> "[clean text]"
                source_text = f"correct typos: {data['corrupted']}"
                target_text = data['clean']
                
                # Tokenize source
                source_encoding = self.tokenizer(
                    source_text,
                    truncation=True,
                    padding=False,
                    max_length=self.max_source_length,
                    return_tensors='pt'
                )
                
                # Tokenize target
                target_encoding = self.tokenizer(
                    target_text,
                    truncation=True,
                    padding=False,
                    max_length=self.max_target_length,
                    return_tensors='pt'
                )
                
                self.examples.append({
                    'input_ids': source_encoding['input_ids'].squeeze(),
                    'attention_mask': source_encoding['attention_mask'].squeeze(),
                    'labels': target_encoding['input_ids'].squeeze(),
                })
        
        logger.info(f"✅ Loaded {len(self.examples)} T5 training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def test_t5_model_accuracy(model, tokenizer, eval_dataset, num_samples=20):
    """Test T5 model's accuracy on typo correction."""
    model.eval()
    
    # Get sample from dataset
    total_examples = len(eval_dataset)
    if hasattr(eval_dataset, 'dataset'):  # Subset
        original_examples = eval_dataset.dataset.examples
        sample_indices = random.sample(eval_dataset.indices, min(num_samples, len(eval_dataset.indices)))
        test_examples = [original_examples[i] for i in sample_indices]
    else:  # Direct dataset
        sample_indices = random.sample(range(total_examples), min(num_samples, total_examples))
        test_examples = [eval_dataset.examples[i] for i in sample_indices]
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, example_data in enumerate(test_examples):
            try:
                # Get input text
                input_ids = example_data['input_ids'].unsqueeze(0)
                attention_mask = example_data['attention_mask'].unsqueeze(0)
                
                # Move to device
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
                
                # Generate correction
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=64,
                    num_beams=1,  # Greedy decoding for speed
                    do_sample=False,
                    early_stopping=True,
                )
                
                # Decode prediction
                predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                # Decode expected target
                expected_text = tokenizer.decode(example_data['labels'], skip_special_tokens=True).strip()
                
                # Normalize for comparison
                pred_normalized = ' '.join(predicted_text.lower().replace('.', '').split())
                expected_normalized = ' '.join(expected_text.lower().replace('.', '').split())
                
                if pred_normalized == expected_normalized:
                    correct += 1
                elif i < 3:  # Show first few examples
                    # Get source text for context
                    source_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
                    logger.info(f"❌ Example {i+1}:")
                    logger.info(f"   Input: '{source_text}'")
                    logger.info(f"   Generated: '{predicted_text}'")
                    logger.info(f"   Expected: '{expected_text}'")
                
                total += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Error testing example {i+1}: {e}")
                continue
    
    model.train()
    return correct / total if total > 0 else 0.0

def setup_t5_model_tokenizer(model_name: str):
    """Setup T5 model and tokenizer for M4 MacBook."""
    logger.info(f"🚀 Loading T5 model for M4 MacBook: {model_name}")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Load model with memory optimizations for MacBook
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use FP32 for MPS stability
    )
    
    # Move to MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        logger.info("🍎 Using Apple MPS acceleration")
    else:
        device = torch.device("cpu")
        logger.info("💻 Using CPU")
    
    logger.info(f"✅ T5 model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer

def create_validation_split(dataset, ratio=0.1):
    """Create train/validation split."""
    total = len(dataset)
    val_size = int(total * ratio)
    train_size = total - val_size
    
    indices = torch.randperm(total).tolist()
    train_subset = torch.utils.data.Subset(dataset, indices[:train_size])
    val_subset = torch.utils.data.Subset(dataset, indices[train_size:])
    
    # Copy examples for evaluation access
    val_subset.examples = [dataset.examples[i] for i in indices[train_size:]]
    
    return train_subset, val_subset

def main():
    parser = argparse.ArgumentParser(description="T5-efficient-tiny typo correction training for M4 MacBook")
    parser.add_argument('--model_name', type=str, default='google/t5-efficient-tiny',
                       help='T5 model name')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Training JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--hf_repo', type=str,
                       help='HuggingFace repo to push to (username/repo)')
    
    # M4 MacBook optimized parameters
    parser.add_argument('--max_source_length', type=int, default=64,
                       help='Max source sequence length')
    parser.add_argument('--max_target_length', type=int, default=64,
                       help='Max target sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Per-device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (T5 typically needs higher LR)')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Eval every N steps')
    parser.add_argument('--logging_steps', type=int, default=50,
                       help='Log every N steps')
    
    # Quality targets
    parser.add_argument('--target_accuracy', type=float, default=0.85,
                       help='Target accuracy (85% for T5-tiny)')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(42)
    
    logger.info("🚀 T5-efficient-tiny Typo Correction Training (M4 MacBook)")
    logger.info(f"📁 Output: {args.output_dir}")
    logger.info(f"🎯 Target accuracy: {args.target_accuracy:.1%}")
    logger.info(f"💻 Optimized for Apple M4 MacBook")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_t5_model_tokenizer(args.model_name)
    
    # Load dataset
    full_dataset = T5TypoDataset(
        args.train_file, 
        tokenizer, 
        args.max_source_length, 
        args.max_target_length
    )
    train_dataset, eval_dataset = create_validation_split(full_dataset, 0.1)
    
    logger.info(f"📊 Train: {len(train_dataset):,} examples")
    logger.info(f"📊 Eval: {len(eval_dataset):,} examples")
    
    # Calculate training parameters
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    total_steps = len(train_dataset) // effective_batch_size * args.num_epochs
    
    logger.info(f"📊 Effective batch size: {effective_batch_size}")
    logger.info(f"📊 Total training steps: {total_steps:,}")
    
    # Data collator for T5
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=args.max_target_length,
    )
    
    # Training arguments for M4 MacBook
    training_args = TrainingArguments(
        # Basic setup
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        
        # Batch sizes
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Learning parameters
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps", 
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Memory optimizations for MacBook
        fp16=False,  # Disable FP16 for MPS stability
        dataloader_pin_memory=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues on macOS
        
        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        
        # Other optimizations
        remove_unused_columns=False,
        prediction_loss_only=True,
        include_inputs_for_metrics=False,
        save_safetensors=True,
        
        # Disable wandb by default
        report_to=[],
        run_name=f"t5-tiny-typo-{args.num_epochs}ep-m4",
        
        # Use CPU for eval if MPS has issues
        use_cpu=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    
    # Training info
    logger.info(f"\n📋 T5 Training Plan:")
    logger.info(f"   Device: {'Apple MPS' if torch.backends.mps.is_available() else 'CPU'}")
    logger.info(f"   Model: {args.model_name}")
    logger.info(f"   Source length: {args.max_source_length} tokens")
    logger.info(f"   Target length: {args.max_target_length} tokens")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Effective batch: {effective_batch_size}")
    logger.info(f"   Learning rate: {args.learning_rate}")
    logger.info(f"   Weight decay: {args.weight_decay}")
    logger.info(f"   Total steps: {total_steps:,}")
    
    # Estimate training time
    estimated_time_min = total_steps * 1.0 / 60  # ~1 sec per step on M4
    logger.info(f"   Estimated time: {estimated_time_min:.1f} minutes")
    
    # Start training
    logger.info("\n🚀 Starting T5 training...")
    start_time = time.time()
    
    try:
        # Train the model
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # Final evaluation
        final_metrics = trainer.evaluate()
        
        # Test accuracy
        logger.info("🧪 Testing T5 model accuracy...")
        final_accuracy = test_t5_model_accuracy(model, tokenizer, eval_dataset)
        
        logger.info(f"🎯 Final accuracy: {final_accuracy:.1%}")
        
        # Save model
        logger.info(f"💾 Saving T5 model to {args.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training info
        training_info = {
            "model_name": args.model_name,
            "model_type": "T5ForConditionalGeneration",
            "training_time_minutes": training_time / 60,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "training_examples": len(train_dataset),
            "validation_examples": len(eval_dataset),
            "epochs": args.num_epochs,
            "batch_size": effective_batch_size,
            "learning_rate": args.learning_rate,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
            "final_accuracy": final_accuracy,
            "target_accuracy": args.target_accuracy,
            "target_achieved": final_accuracy >= args.target_accuracy,
            "device": "Apple M4 MacBook" if torch.backends.mps.is_available() else "CPU",
            "optimizations": ["MPS", "Seq2Seq", "Early Stopping"],
            "total_steps": total_steps,
        }
        
        with open(f"{args.output_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        # HuggingFace deployment
        if args.hf_repo:
            logger.info(f"🤗 Preparing HuggingFace deployment to {args.hf_repo}")
            
            # Create model card
            model_card = f"""---
language: en
tags:
- text2text-generation
- typo-correction
- t5
- fine-tuned
license: apache-2.0
datasets:
- custom-typo-dataset
base_model: {args.model_name}
---

# T5-Efficient-Tiny Typo Correction Model

Fine-tuned T5-efficient-tiny model for automatic typo correction, achieving {final_accuracy:.1%} sentence accuracy.

## Model Details

- **Base Model**: {args.model_name}
- **Architecture**: Text-to-text (seq2seq)
- **Fine-tuning Data**: {len(train_dataset):,} examples
- **Training Device**: Apple M4 MacBook
- **Training Time**: {training_time/60:.1f} minutes
- **Final Accuracy**: {final_accuracy:.1%}

## Usage

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("{args.hf_repo}")
model = T5ForConditionalGeneration.from_pretrained("{args.hf_repo}")

# Correct typos
input_text = "correct typos: I beleive this is teh correct answr."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=64)
correction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(correction)  # "I believe this is the correct answer."
```

## Performance

- **Sentence Accuracy**: {final_accuracy:.1%}
- **Model Size**: {sum(p.numel() for p in model.parameters()):,} parameters
- **Training Examples**: {len(train_dataset):,}
- **Validation Examples**: {len(eval_dataset):,}
"""
            
            with open(f"{args.output_dir}/README.md", 'w') as f:
                f.write(model_card)
            
            logger.info(f"📝 Model card saved to {args.output_dir}/README.md")
        
        # Success summary
        logger.info(f"\n🎉 T5 training completed successfully!")
        logger.info(f"📊 Final accuracy: {final_accuracy:.1%}")
        logger.info(f"🎯 Target {'✅ ACHIEVED' if final_accuracy >= args.target_accuracy else '❌ NOT MET'}")
        logger.info(f"⏱️ Training time: {training_time/60:.1f} minutes")
        logger.info(f"💾 Model saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
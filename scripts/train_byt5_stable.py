#!/usr/bin/env python3
"""
Stable ByT5 training script with proper gradient handling and loss monitoring
"""

import argparse
import os
import json
import time
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

def guess_columns(example: Dict[str, Any], source_col: Optional[str], target_col: Optional[str]):
    if source_col and target_col:
        return source_col, target_col
    cols = list(example.keys())
    candidates = [
        ("corrupted", "clean"),
        ("input", "target"),
        ("source", "target"),
    ]
    for s, t in candidates:
        if s in cols and t in cols:
            return s, t
    return cols[0], cols[1] if len(cols) >= 2 else cols[0]

def main():
    ap = argparse.ArgumentParser(description="Stable ByT5 training")
    ap.add_argument("--model-name", default="google/byt5-small")
    ap.add_argument("--train-file", default="data/enhanced_training_balanced.jsonl")
    ap.add_argument("--output-dir", default="models/byt5-stable-typo-fixer")
    ap.add_argument("--prefix", default="fix typos:")
    ap.add_argument("--max-length", type=int, default=128)  # Shorter for stability
    ap.add_argument("--learning-rate", type=float, default=1e-5)  # Lower LR
    ap.add_argument("--num-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    
    args = ap.parse_args()

    # Force single GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 STABLE ByT5 Training")
    print(f"📊 Model: {args.model_name}")
    print(f"🎯 Learning Rate: {args.learning_rate} (conservative)")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📏 Max length: {args.max_length}")
    print()

    # Load tokenizer and model
    print("🔧 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.config.use_cache = False

    # Load dataset - smaller subset for debugging
    print("📚 Loading dataset...")
    raw = load_dataset('json', data_files={'train': args.train_file})
    
    # Use even smaller subset to debug the loss issue
    dataset = raw["train"].select(range(min(5000, len(raw["train"]))))
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_data = split["train"]
    eval_data = split["test"]

    # Infer columns
    sample = train_data[0]
    source_col, target_col = guess_columns(sample, None, None)
    print(f"📋 Columns: {source_col} → {target_col}")
    print(f"📊 Train: {len(train_data)}, Eval: {len(eval_data)}")
    
    # Check a few examples
    print("\n📝 Sample data:")
    for i, ex in enumerate(train_data.select(range(3))):
        print(f"  {i+1}. '{ex[source_col]}' → '{ex[target_col]}'")
    print()

    # Preprocess with careful handling
    def preprocess(batch):
        sources = []
        targets = []
        
        for src, tgt in zip(batch[source_col], batch[target_col]):
            # Clean and validate inputs
            if not src or not tgt or len(src.strip()) == 0 or len(tgt.strip()) == 0:
                continue
                
            source_text = f"{args.prefix} {src}".strip()
            target_text = tgt.strip()
            
            # Skip if too similar (might cause loss=0)
            if source_text.replace(args.prefix, "").strip() == target_text:
                continue
                
            sources.append(source_text)
            targets.append(target_text)
        
        if not sources:  # Skip empty batches
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        model_inputs = tokenizer(
            sources, 
            max_length=args.max_length, 
            truncation=True, 
            padding=False
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=args.max_length, 
                truncation=True, 
                padding=False
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("🔄 Processing data...")
    train_ds = train_data.map(
        preprocess, 
        batched=True, 
        remove_columns=train_data.column_names,
        batch_size=100
    )
    eval_ds = eval_data.map(
        preprocess, 
        batched=True, 
        remove_columns=eval_data.column_names,
        batch_size=100
    )

    # Filter out empty examples
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 0)
    eval_ds = eval_ds.filter(lambda x: len(x["input_ids"]) > 0)
    
    print(f"📊 After filtering - Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # Conservative training arguments to prevent collapse
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,  # Very conservative
        num_train_epochs=args.num_epochs,
        
        # Gradient handling
        max_grad_norm=1.0,  # Clip gradients
        gradient_accumulation_steps=2,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,  # Frequent evaluation
        save_strategy="steps", 
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Generation
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=2,
        
        # System - conservative settings
        logging_steps=10,  # Frequent logging
        fp16=False,  # Disable FP16 to avoid numerical issues
        dataloader_num_workers=0,
        report_to=["none"],
        seed=42,
        remove_unused_columns=False,
        
        # Stability
        dataloader_drop_last=True,
        ignore_data_skip=True,
    )

    # Custom callback to monitor for collapse
    class LossMonitorCallback:
        def __init__(self):
            self.loss_history = []
            
        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss = logs["loss"]
                self.loss_history.append(loss)
                
                # Check for collapse (loss too low or NaN gradients)
                if loss < 0.001 or (logs.get("grad_norm") is not None and 
                                   (torch.isnan(torch.tensor(logs["grad_norm"])) or 
                                    logs["grad_norm"] == 0)):
                    print(f"\n⚠️  WARNING: Potential model collapse detected!")
                    print(f"   Loss: {loss:.6f}")
                    print(f"   Grad norm: {logs.get('grad_norm', 'N/A')}")
                    print(f"   This suggests the model may be collapsing.")
                    
                # Check for instability
                if len(self.loss_history) >= 5:
                    recent_losses = self.loss_history[-5:]
                    if all(l == 0.0 for l in recent_losses[-3:]):
                        print(f"\n🛑 STOPPING: Loss has been 0.0 for multiple steps!")
                        control.should_training_stop = True

    # Trainer with monitoring
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LossMonitorCallback()],
    )

    print("🚀 Starting stable training...")
    print("🔍 Monitoring for loss collapse...")
    start_time = time.time()
    
    try:
        trainer.train()
        print(f"✅ Training completed in {(time.time() - start_time)/60:.1f} minutes")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"💾 Model saved to: {args.output_dir}")

    # Comprehensive test
    print("\n🧪 Comprehensive test:")
    model.eval()
    
    test_cases = [
        "I beleive this is teh correct answr.",
        "The qick brown fox jumps ovr the lazy dog.",
        "Please chck your email for futher instructions.",
    ]
    
    for test_input in test_cases:
        input_text = f"{args.prefix} {test_input}"
        inputs = tokenizer(input_text, return_tensors='pt', max_length=args.max_length, truncation=True)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=args.max_length, 
                num_beams=2,
                do_sample=False,
                early_stopping=True
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  '{test_input}' → '{result}'")

    print(f"\n🎉 Stable training complete!")

if __name__ == "__main__":
    main()
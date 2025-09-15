#!/usr/bin/env python3
"""
Minimal ByT5 training script - no complex callbacks, just works
"""

import argparse
import os
import sys
import json
import time
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
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
        ("noisy", "clean"),
        ("text", "label"),
        ("src", "tgt"),
    ]
    for s, t in candidates:
        if s in cols and t in cols:
            return s, t
    text_cols = [c for c in cols if isinstance(example[c], str)]
    if len(text_cols) >= 2:
        return text_cols[0], text_cols[1]
    raise ValueError(f"Could not infer source/target columns from columns: {cols}")

def main():
    ap = argparse.ArgumentParser(description="Minimal ByT5 typo correction training")
    ap.add_argument("--model-name", default="google/byt5-small", help="Base model")
    ap.add_argument("--train-file", default="data/enhanced_training_balanced.jsonl", help="Training file")
    ap.add_argument("--output-dir", default="models/byt5-small-typo-fixer-v3", help="Output directory")
    ap.add_argument("--prefix", default="fix typos:", help="Instruction prefix")
    ap.add_argument("--max-source-len", type=int, default=256, help="Max source length")
    ap.add_argument("--max-target-len", type=int, default=256, help="Max target length")
    ap.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    ap.add_argument("--num-epochs", type=int, default=3, help="Training epochs")
    ap.add_argument("--per-device-train-batch-size", type=int, default=8, help="Batch size per device")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation")
    
    args = ap.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"🚀 Found {num_gpus} GPUs - using DataParallel")
    else:
        print(f"💻 Using single GPU/CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 MINIMAL ByT5 Typo Fixer Training")
    print(f"📊 Configuration:")
    print(f"   Model: {args.model_name}")
    print(f"   Prefix: '{args.prefix}'")
    print(f"   Max Length: {args.max_source_len}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch Size: {args.per_device_train_batch_size}")
    print()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.config.use_cache = False

    # Handle multi-GPU
    if num_gpus > 1:
        model = nn.DataParallel(model)

    # Load dataset
    raw = load_dataset('json', data_files={'train': args.train_file})
    
    # Auto-split train/validation
    split = raw["train"].train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    eval_data = split["test"]

    # Infer columns
    sample = train_data[0]
    source_col, target_col = guess_columns(sample, None, None)
    print(f"📋 Using columns: source='{source_col}', target='{target_col}'")
    print(f"📊 Training samples: {len(train_data)}")
    print(f"📊 Validation samples: {len(eval_data)}")

    # Preprocess function
    def preprocess(batch):
        sources = [f"{args.prefix} {s}".strip() for s in batch[source_col]]
        model_inputs = tokenizer(
            sources, 
            max_length=args.max_source_len, 
            truncation=True, 
            padding=False
        )
        labels = tokenizer(
            text_target=batch[target_col], 
            max_length=args.max_target_len, 
            truncation=True, 
            padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Process datasets
    train_ds = train_data.map(preprocess, batched=True, remove_columns=train_data.column_names)
    eval_ds = eval_data.map(preprocess, batched=True, remove_columns=eval_data.column_names)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    # Training arguments - simple and stable
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=16,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps", 
        save_steps=1000,
        save_total_limit=2,
        
        # Generation
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=4,
        
        # System
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True,
        dataloader_num_workers=2,
        report_to=["none"],
        seed=42,
        remove_unused_columns=False,  # Fix for column mismatch
    )

    # Trainer - minimal setup
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Starting training...")
    start_time = time.time()
    
    trainer.train()
    
    training_time = time.time() - start_time
    print(f"⏱️ Training completed in {training_time/60:.1f} minutes")

    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ Model saved to: {args.output_dir}")
    
    # Quick test
    print("\n🧪 Quick test:")
    test_cases = [
        "I beleive this is teh correct answr.",
        "The qick brown fox jumps ovr the lazy dog.",
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_to_test = model.module if hasattr(model, 'module') else model
    model_to_test.eval()
    
    for test_input in test_cases:
        input_text = f"{args.prefix} {test_input}"
        inputs = tokenizer(input_text, return_tensors='pt', max_length=args.max_source_len, truncation=True)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model_to_test.generate(**inputs, max_length=args.max_target_len, num_beams=4)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  '{test_input}' → '{result}'")

    print(f"\n🎉 Training complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Single GPU ByT5 training script - simpler and more stable
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
    ap = argparse.ArgumentParser(description="Single GPU ByT5 training")
    ap.add_argument("--model-name", default="google/byt5-small")
    ap.add_argument("--train-file", default="data/enhanced_training_balanced.jsonl")
    ap.add_argument("--output-dir", default="models/byt5-small-typo-fixer-v3")
    ap.add_argument("--prefix", default="fix typos:")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--num-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    
    args = ap.parse_args()

    # Force single GPU to avoid DataParallel issues
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"🚀 Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 SINGLE GPU ByT5 Training")
    print(f"📊 Model: {args.model_name}")
    print(f"🎯 Epochs: {args.num_epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print()

    # Load tokenizer and model
    print("🔧 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.config.use_cache = False

    # Load dataset
    print("📚 Loading dataset...")
    raw = load_dataset('json', data_files={'train': args.train_file})
    
    # Take smaller subset for faster testing
    dataset = raw["train"].select(range(min(20000, len(raw["train"]))))
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    eval_data = split["test"]

    # Infer columns
    sample = train_data[0]
    source_col, target_col = guess_columns(sample, None, None)
    print(f"📋 Columns: {source_col} → {target_col}")
    print(f"📊 Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Preprocess
    def preprocess(batch):
        sources = [f"{args.prefix} {s}".strip() for s in batch[source_col]]
        model_inputs = tokenizer(
            sources, 
            max_length=args.max_length, 
            truncation=True, 
            padding=False
        )
        labels = tokenizer(
            text_target=batch[target_col], 
            max_length=args.max_length, 
            truncation=True, 
            padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("🔄 Processing data...")
    train_ds = train_data.map(preprocess, batched=True, remove_columns=train_data.column_names)
    eval_ds = eval_data.map(preprocess, batched=True, remove_columns=eval_data.column_names)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Simple training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps", 
        save_steps=500,
        save_total_limit=2,
        
        # Generation
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=2,  # Smaller for speed
        
        # System
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        report_to=["none"],
        seed=42,
        remove_unused_columns=False,
    )

    # Trainer
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

    # Quick test
    print("\n🧪 Quick test:")
    model.eval()
    
    test_cases = [
        "I beleive this is teh correct answr.",
        "The qick brown fox jumps ovr the lazy dog.",
    ]
    
    for test_input in test_cases:
        input_text = f"{args.prefix} {test_input}"
        inputs = tokenizer(input_text, return_tensors='pt', max_length=args.max_length, truncation=True)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=args.max_length, num_beams=2)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  '{test_input}' → '{result}'")

    print(f"\n🎉 Training complete!")
    
    # Create simple test results
    results = {
        "training_time_minutes": (time.time() - start_time) / 60,
        "training_samples": len(train_ds),
        "model_path": args.output_dir,
        "test_examples": [(test_input, result) for test_input in test_cases]
    }
    
    with open(f"{args.output_dir}/simple_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
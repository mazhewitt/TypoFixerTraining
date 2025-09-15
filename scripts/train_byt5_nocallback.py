#!/usr/bin/env python3
"""
ByT5 training script without custom callbacks - just stable training
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
    ap = argparse.ArgumentParser(description="ByT5 training without callbacks")
    ap.add_argument("--model-name", default="google/byt5-small")
    ap.add_argument("--train-file", default="data/enhanced_training_balanced.jsonl")
    ap.add_argument("--output-dir", default="models/byt5-nocallback-typo-fixer")
    ap.add_argument("--prefix", default="fix typos:")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=3e-5)  # Slightly higher but safe
    ap.add_argument("--num-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dataset-size", type=int, default=10000)  # Subset for testing
    
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 ByT5 Training (No Callbacks)")
    print(f"📊 Model: {args.model_name}")
    print(f"🎯 Learning Rate: {args.learning_rate}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📏 Max length: {args.max_length}")
    print(f"📊 Dataset size: {args.dataset_size}")
    print()

    # Load tokenizer and model
    print("🔧 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.config.use_cache = False

    # Load dataset
    print("📚 Loading dataset...")
    raw = load_dataset('json', data_files={'train': args.train_file})
    
    # Use subset
    dataset = raw["train"].select(range(min(args.dataset_size, len(raw["train"]))))
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    eval_data = split["test"]

    # Infer columns
    sample = train_data[0]
    source_col, target_col = guess_columns(sample, None, None)
    print(f"📋 Columns: {source_col} → {target_col}")
    print(f"📊 Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Preprocess - simple and robust
    def preprocess(batch):
        sources = [f"{args.prefix} {s}".strip() for s in batch[source_col]]
        targets = [t.strip() for t in batch[target_col]]
        
        model_inputs = tokenizer(
            sources, 
            max_length=args.max_length, 
            truncation=True, 
            padding=False
        )
        
        # Use text_target parameter to avoid deprecation warning
        labels = tokenizer(
            text_target=targets, 
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
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        label_pad_token_id=-100
    )

    # Training arguments - stable and simple
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        
        # Stability
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        
        # Evaluation
        eval_strategy="epoch",  # Evaluate at end of each epoch only
        save_strategy="epoch",  # Save only at end of each epoch
        save_total_limit=1,     # Keep only the latest save
        
        # Generation
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=2,
        
        # System
        logging_steps=50,
        fp16=False,  # Avoid numerical issues
        dataloader_num_workers=0,
        report_to=["none"],
        seed=42,
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    # Simple trainer - no custom callbacks
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Starting training...")
    print("💡 Watch the loss - it should start high and decrease gradually")
    print("⚠️  If loss goes to 0.0 quickly, stop training (Ctrl+C)")
    
    start_time = time.time()
    
    try:
        trainer.train()
        print(f"✅ Training completed in {(time.time() - start_time)/60:.1f} minutes")
    except KeyboardInterrupt:
        print("\n🛑 Training stopped by user")
        return
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"💾 Model saved to: {args.output_dir}")

    # Test the model
    print("\n🧪 Testing model:")
    model.eval()
    
    test_cases = [
        "I beleive this is teh correct answr.",
        "The qick brown fox jumps ovr the lazy dog.",
        "Please chck your email for futher instructions.",
    ]
    
    results = []
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
        results.append((test_input, result))
        print(f"  '{test_input}' → '{result}'")

    # Save test results
    test_results = {
        "training_time_minutes": (time.time() - start_time) / 60,
        "training_samples": len(train_ds),
        "model_path": args.output_dir,
        "test_results": results,
        "training_args": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "epochs": args.num_epochs
        }
    }
    
    with open(f"{args.output_dir}/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n🎉 Training complete!")
    print(f"📁 Model: {args.output_dir}")
    print(f"📊 Results: {args.output_dir}/test_results.json")

if __name__ == "__main__":
    main()
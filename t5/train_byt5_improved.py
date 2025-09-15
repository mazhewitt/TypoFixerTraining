#!/usr/bin/env python3
"""
IMPROVED ByT5-small typo fixer training script
Optimized for dual RTX 5090 CUDA setup with lessons learned from failed model.

Key improvements:
1. Consistent prompt format between training and inference
2. Better hyperparameters for ByT5 character-level training  
3. Longer sequences to avoid truncation
4. More epochs for proper character-level learning
5. CUDA optimizations for dual RTX 5090s
6. Built-in testing during training

Usage:
  python3 t5/train_byt5_improved.py \
    --train-file data/enhanced_training_full.jsonl \
    --output-dir models/byt5-small-typo-fixer-v2 \
    --num-epochs 5
"""

import argparse
import os
import sys
import json
import time
from typing import List, Dict, Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

try:
    import sacrebleu  # type: ignore
except Exception:
    sacrebleu = None


def setup_distributed():
    """Setup for multi-GPU training if available"""
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"🚀 Found {num_gpus} GPUs available")
        # Only initialize distributed if not already done and env vars are set
        if not dist.is_available() or not dist.is_initialized():
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                print("🔧 Initializing distributed training...")
                dist.init_process_group(backend='nccl')
            else:
                print("⚠️  Distributed env vars not set - using DataParallel instead")
                return num_gpus
    else:
        print(f"💻 Using single GPU training")
    return num_gpus


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


def run_inference_test(model, tokenizer, device, prefix):
    """Run quick inference test during training"""
    test_cases = [
        "I beleive this is teh correct answr.",
        "The qick brown fox jumps ovr the lazy dog.",
        "Please chck your email for futher instructions.",
    ]
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for test_text in test_cases:
            input_text = f"{prefix} {test_text}".strip()
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(device)
            
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append((test_text, result))
    
    model.train()
    return results


def build_compute_metrics(tokenizer, prefix):
    def postprocess_text(preds: List[str], labels: List[str]):
        preds = [p.strip() for p in preds]
        labels = [l.strip() for l in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in labels so we can decode
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        metrics = {}
        if sacrebleu is not None:
            try:
                metrics["chrf"] = round(sacrebleu.corpus_chrf(decoded_preds, [decoded_labels]).score, 2)
                metrics["bleu"] = round(sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score, 2)
            except Exception:
                pass
        
        exact = sum(p == l for p, l in zip(decoded_preds, decoded_labels)) / max(1, len(decoded_labels))
        metrics["exact_match"] = round(exact * 100, 2)
        metrics["pred_len"] = float(sum(len(p) for p in decoded_preds) / max(1, len(decoded_preds)))
        
        # Quick inference test
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Note: We'll get the model from the trainer in a callback instead
        
        return metrics

    return compute_metrics


class InferenceTestCallback:
    """Custom callback to run inference tests during training"""
    
    def __init__(self, tokenizer, prefix):
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.test_results = []
    
    def on_init_end(self, args, state, control, **kwargs):
        # Required method for callback
        return control
    
    def on_train_begin(self, args, state, control, **kwargs):
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        return control
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        return control
    
    def on_step_begin(self, args, state, control, **kwargs):
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        return control
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is not None:
            device = next(model.parameters()).device
            results = run_inference_test(model, self.tokenizer, device, self.prefix)
            
            print("\n" + "="*60)
            print(f"INFERENCE TEST - Epoch {state.epoch:.1f}")
            print("="*60)
            for orig, corrected in results:
                print(f"'{orig}' → '{corrected}'")
            print("="*60)
            
            self.test_results.append({
                'epoch': state.epoch,
                'step': state.global_step,
                'results': results
            })
        return control


def main():
    ap = argparse.ArgumentParser(description="IMPROVED ByT5-small typo correction training")
    ap.add_argument("--model-name", default="google/byt5-small", help="Base model")
    ap.add_argument("--train-file", default="data/enhanced_training_balanced.jsonl", help="Training file (json/jsonl/csv)")
    ap.add_argument("--eval-file", default=None, help="Eval file (json/jsonl/csv)")
    ap.add_argument("--output-dir", default="models/byt5-small-typo-fixer-v2", help="Output directory")
    ap.add_argument("--source-col", default=None, help="Source column name")
    ap.add_argument("--target-col", default=None, help="Target column name")
    
    # IMPROVED: Simple, consistent prefix
    ap.add_argument("--prefix", default="fix typos:", help="Instruction prefix - KEEP SIMPLE!")
    
    # IMPROVED: Longer sequences to avoid truncation
    ap.add_argument("--max-source-len", type=int, default=512, help="Max source length")
    ap.add_argument("--max-target-len", type=int, default=512, help="Max target length")
    
    # IMPROVED: Better learning parameters for ByT5
    ap.add_argument("--learning-rate", type=float, default=5e-5, help="Lower LR for ByT5")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    ap.add_argument("--num-epochs", type=int, default=5, help="More epochs for char-level")
    ap.add_argument("--warmup-ratio", type=float, default=0.1, help="Longer warmup")
    
    # IMPROVED: CUDA-optimized batch sizes for dual RTX 5090
    ap.add_argument("--per-device-train-batch-size", type=int, default=8, help="Per device batch size")
    ap.add_argument("--per-device-eval-batch-size", type=int, default=16, help="Eval batch size")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Grad accumulation")
    
    ap.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    ap.add_argument("--eval-steps", type=int, default=500, help="Eval frequency")
    ap.add_argument("--save-steps", type=int, default=500, help="Save frequency")
    ap.add_argument("--logging-steps", type=int, default=50, help="Logging frequency")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="Save memory")
    ap.add_argument("--push-to-hub", action="store_true", help="Upload to HuggingFace")
    ap.add_argument("--hub-model-id", default=None, help="HuggingFace model ID")
    
    args = ap.parse_args()

    # Setup distributed training for dual GPUs
    num_gpus = setup_distributed()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 IMPROVED ByT5 Typo Fixer Training")
    print(f"📊 Configuration:")
    print(f"   Model: {args.model_name}")
    print(f"   Prefix: '{args.prefix}'")
    print(f"   Sequence Length: {args.max_source_len}/{args.max_target_len}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   GPUs: {num_gpus}")
    print(f"   Effective Batch Size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * num_gpus}")
    print()

    # Load tokenizer and model - ByT5 needs use_fast=False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Disable cache during training
    model.config.use_cache = False
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Handle multi-GPU setup
    if num_gpus > 1 and not dist.is_initialized():
        print("🔧 Using DataParallel for multi-GPU training")
        model = nn.DataParallel(model)

    # Load datasets
    files = {"train": args.train_file}
    if args.eval_file:
        files["validation"] = args.eval_file

    ext = os.path.splitext(args.train_file)[1].lower().lstrip(".")
    if ext == "jsonl":
        ext = "json"
    raw = load_dataset(ext, data_files=files)

    # Auto-split if no eval file
    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=0.1, seed=args.seed)
        raw = {"train": split["train"], "validation": split["test"]}
    else:
        raw = {"train": raw["train"], "validation": raw["validation"]}

    # Infer columns
    sample = raw["train"][0]
    source_col, target_col = guess_columns(sample, args.source_col, args.target_col)
    print(f"📋 Using columns: source='{source_col}', target='{target_col}'")
    print(f"📊 Training samples: {len(raw['train'])}")
    print(f"📊 Validation samples: {len(raw['validation'])}")
    print()

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
    train_ds = raw["train"].map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    eval_ds = raw["validation"].map(preprocess, batched=True, remove_columns=raw["validation"].column_names)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model, 
        pad_to_multiple_of=8
    )

    # Training arguments - CUDA optimized
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps", 
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        
        # Generation settings
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=4,
        
        # System settings
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available(),  # Use FP16 on CUDA
        dataloader_drop_last=True,
        group_by_length=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        
        # Distributed training
        ddp_find_unused_parameters=False,
        
        # Other
        report_to=["none"],
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # Custom callback for inference testing
    inference_callback = InferenceTestCallback(tokenizer, args.prefix)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer, args.prefix),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            inference_callback,
        ],
    )

    # Initial inference test
    print("🧪 INITIAL MODEL TEST (before training):")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_results = run_inference_test(model, tokenizer, device, args.prefix)
    for orig, corrected in initial_results:
        print(f"  '{orig}' → '{corrected}'")
    print()

    # Train!
    print("🚀 Starting training...")
    start_time = time.time()
    
    trainer.train()
    
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time/60:.1f} minutes")

    # Final evaluation
    print("\n🔍 Final evaluation...")
    try:
        metrics = trainer.evaluate()
        print(f"📊 Final metrics: {metrics}")
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")

    # Final inference test
    print("\n🧪 FINAL MODEL TEST (after training):")
    final_results = run_inference_test(model, tokenizer, device, args.prefix)
    for orig, corrected in final_results:
        print(f"  '{orig}' → '{corrected}'")

    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training summary
    summary = {
        "model_name": args.model_name,
        "prefix": args.prefix,
        "max_lengths": [args.max_source_len, args.max_target_len],
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "training_time_minutes": training_time / 60,
        "num_gpus": num_gpus,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * num_gpus,
        "initial_results": initial_results,
        "final_results": final_results,
        "inference_tests": inference_callback.test_results,
    }
    
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {args.output_dir}")
    print(f"📋 Training summary saved to: {args.output_dir}/training_summary.json")
    
    if args.push_to_hub:
        print(f"🚀 Model pushed to HuggingFace Hub: {args.hub_model_id}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Train a ByT5-small typo fixer using Hugging Face Transformers on the local dataset.

Examples:
  # Auto-split (90/10) from a single JSONL file with fields: corrupted, clean
  python3 t5/train_byt5.py \
    --train-file data/enhanced_training_full.jsonl \
    --output-dir models/byt5-small-typo-fixer \
    --num-epochs 1

  # With explicit eval file and custom columns
  python3 t5/train_byt5.py \
    --train-file data/train.jsonl --eval-file data/valid.jsonl \
    --source-col corrupted --target-col clean \
    --output-dir models/byt5-small-typo-fixer
"""

import argparse
import os
import sys
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    import sacrebleu  # type: ignore
except Exception:
    sacrebleu = None


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


def build_compute_metrics(tokenizer):
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
        return metrics

    return compute_metrics


def main():
    ap = argparse.ArgumentParser(description="Train ByT5-small for typo correction")
    ap.add_argument("--model-name", default="google/byt5-small", help="Base model to fine-tune")
    ap.add_argument("--train-file", required=True, help="Path to train file (json/jsonl/csv)")
    ap.add_argument("--eval-file", default=None, help="Path to eval file (json/jsonl/csv)")
    ap.add_argument("--output-dir", default="models/byt5-small-typo-fixer", help="Output directory")
    ap.add_argument("--source-col", default=None, help="Source/input column name")
    ap.add_argument("--target-col", default=None, help="Target/label column name")
    ap.add_argument("--prefix", default="fix spelling errors only, don't change the meaning of the text:", help="Instruction prefix")
    ap.add_argument("--max-source-len", type=int, default=256, help="Max source tokens (bytes for ByT5)")
    ap.add_argument("--max-target-len", type=int, default=128, help="Max target tokens")
    ap.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    ap.add_argument("--num-epochs", type=int, default=1, help="Training epochs")
    ap.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    ap.add_argument("--per-device-train-batch-size", type=int, default=32)
    ap.add_argument("--per-device-eval-batch-size", type=int, default=32)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    # macOS + MPS often has multiprocessing sharing issues; default workers=0 there
    default_workers = 0 if sys.platform == "darwin" else min(4, (os.cpu_count() or 2))
    ap.add_argument("--num-workers", type=int, default=default_workers, help="Dataloader workers")
    ap.add_argument("--eval-steps", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=None)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="Enable FP16 (CUDA only)")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--push-to-hub", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer/model (ByT5 uses byte-level tokenizer; use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    # Disable cache during training to reduce overhead and improve throughput
    try:
        model.config.use_cache = False
    except Exception:
        pass

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load datasets
    files = {"train": args.train_file}
    if args.eval_file:
        files["validation"] = args.eval_file

    ext = os.path.splitext(args.train_file)[1].lower().lstrip(".")
    if ext in {"jsonl"}:
        ext = "json"
    raw = load_dataset(ext, data_files=files)

    # If no explicit eval file, split train 90/10
    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=0.1, seed=args.seed)
        raw = {"train": split["train"], "validation": split["test"]}
    else:
        raw = {"train": raw["train"], "validation": raw["validation"]}

    # Infer columns
    sample = raw["train"][0]
    source_col, target_col = guess_columns(sample, args.source_col, args.target_col)
    print(f"Using columns -> source: '{source_col}'  target: '{target_col}'")

    # Preprocess
    def preprocess(batch):
        sources = [f"{args.prefix} {s}".strip() for s in batch[source_col]]
        model_inputs = tokenizer(sources, max_length=args.max_source_len, truncation=True, padding=False)
        # Use text_target to avoid deprecated context manager and speed things up
        labels = tokenizer(text_target=batch[target_col], max_length=args.max_target_len, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = raw["train"].map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    eval_ds = raw["validation"].map(preprocess, batched=True, remove_columns=raw["validation"].column_names)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=4,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=["none"],
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        group_by_length=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
    dataloader_drop_last=True,
    )

    # Ensure a safe multiprocessing start method on macOS
    try:
        import torch.multiprocessing as mp
        if sys.platform == "darwin":
            mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    if torch.backends.mps.is_available() and args.fp16:
        print("Warning: FP16 is not supported on MPS; running in FP32.")

    trainer.train()
    # Explicit evaluation at end (compatible across versions)
    try:
        metrics = trainer.evaluate()
        print(f"\n📊 Eval metrics: {metrics}")
    except Exception as _:
        pass
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Training complete. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

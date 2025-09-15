#!/usr/bin/env python3
"""
ByT5-small training on CUDA (single or multi-GPU) with optional Hugging Face upload.

Single GPU example (auto-split + push to HF):
    HF_TOKEN=xxxxxxxx \
    python3 scripts/training/train_byt5_rtx5090.py \
        --train-file data/enhanced_training_full.jsonl \
        --output-dir models/byt5-small-typo-fixer \
        --hub-repo-id mazhewitt/byt5-small-typo-fixer \
        --push-to-hub --hub-private --num-epochs 2 \
        --per-device-train-batch-size 64 --per-device-eval-batch-size 64 \
        --gradient-accumulation-steps 2 --eval-steps 1000 --save-steps 1000

Dual GPU (DDP) example with torchrun:
    HF_TOKEN=xxxxxxxx \
    torchrun --nproc_per_node=2 scripts/training/train_byt5_rtx5090.py \
        --train-file data/enhanced_training_full.jsonl \
        --output-dir models/byt5-small-typo-fixer \
        --num-epochs 2 \
        --per-device-train-batch-size 64 --per-device-eval-batch-size 64 \
        --gradient-accumulation-steps 2 --eval-steps 1000 --save-steps 1000 \
        --push-to-hub --hub-repo-id mazhewitt/byt5-small-typo-fixer --hub-private
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

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

try:
    from huggingface_hub import HfFolder, create_repo
except Exception:
    HfFolder = None
    create_repo = None


def guess_columns(example: Dict[str, Any], source_col: Optional[str], target_col: Optional[str]):
    if source_col and target_col:
        return source_col, target_col
    cols = list(example.keys())
    for s, t in [("corrupted", "clean"), ("input", "target"), ("source", "target"), ("noisy", "clean"), ("text", "label"), ("src", "tgt")]:
        if s in cols and t in cols:
            return s, t
    # fallback to first two string columns
    text_cols = [c for c in cols if isinstance(example[c], str)]
    if len(text_cols) >= 2:
        return text_cols[0], text_cols[1]
    raise ValueError(f"Could not infer source/target columns from columns: {cols}")


def build_compute_metrics(tokenizer):
    def postprocess_text(preds: List[str], labels: List[str]):
        return [p.strip() for p in preds], [l.strip() for l in labels]

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        metrics: Dict[str, Any] = {}
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
    ap = argparse.ArgumentParser(description="Train ByT5-small on CUDA and optionally push to Hugging Face")
    ap.add_argument("--model-name", default="google/byt5-small")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--output-dir", default="models/byt5-small-typo-fixer")
    ap.add_argument("--source-col", default=None)
    ap.add_argument("--target-col", default=None)
    ap.add_argument("--prefix", default="fix spelling errors only, don't change the meaning of the text:")
    ap.add_argument("--max-source-len", type=int, default=256)
    ap.add_argument("--max-target-len", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-epochs", type=int, default=2)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--per-device-train-batch-size", type=int, default=64)
    ap.add_argument("--per-device-eval-batch-size", type=int, default=64)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=2)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-repo-id", default=None, help="e.g., mazhewitt/byt5-small-typo-fixer")
    ap.add_argument("--hub-private", action="store_true")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in output_dir if present")
    args = ap.parse_args()

    # CUDA-friendly settings
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    # Disable cache for training throughput
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # Load data
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    ext = os.path.splitext(args.train_file)[1].lower().lstrip(".")
    if ext == "jsonl":
        ext = "json"
    raw = load_dataset(ext, data_files=data_files)

    # Auto-split if needed
    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=0.1, seed=args.seed)
        raw = {"train": split["train"], "validation": split["test"]}
    else:
        raw = {"train": raw["train"], "validation": raw["validation"]}

    # Determine columns
    sample = raw["train"][0]
    source_col, target_col = guess_columns(sample, args.source_col, args.target_col)
    print(f"Using columns -> source: '{source_col}'  target: '{target_col}'")

    # Preprocess
    def preprocess(batch):
        sources = [f"{args.prefix} {s}".strip() for s in batch[source_col]]
        model_inputs = tokenizer(sources, max_length=args.max_source_len, truncation=True, padding=False)
        labels = tokenizer(text_target=batch[target_col], max_length=args.max_target_len, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = raw["train"].map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    eval_ds = raw["validation"].map(preprocess, batched=True, remove_columns=raw["validation"].column_names)

    # Collator and workers
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
    num_workers = max(4, min(16, (os.cpu_count() or 8)))

    # Precision selection: prefer bf16 on RTX 50xx if available
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    # Training args
    # Build TrainingArguments with a fallback for older/newer Transformers that lack some kwargs
    # Also ensure max_steps is always an int (older versions do raw comparisons)
    max_steps_value = args.max_steps if (args.max_steps is not None and args.max_steps > 0) else -1
    training_args = None
    try:
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
            save_total_limit=args.save_total_limit,
            predict_with_generate=True,
            generation_max_length=args.max_target_len,
            generation_num_beams=4,
            fp16=fp16,
            bf16=bf16,
            optim="adamw_torch_fused",
            report_to=["none"],
            seed=args.seed,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_repo_id,
            group_by_length=True,
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
            dataloader_drop_last=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            max_steps=max_steps_value,
            load_best_model_at_end=True,
            metric_for_best_model="chrf" if sacrebleu is not None else "exact_match",
            greater_is_better=True,
            ddp_find_unused_parameters=False,
            save_safetensors=True,
        )
    except TypeError:
        # Drop potentially unsupported arguments and retry
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
            save_total_limit=args.save_total_limit,
            predict_with_generate=True,
            generation_max_length=args.max_target_len,
            generation_num_beams=4,
            fp16=fp16,
            bf16=bf16,
            report_to=["none"],
            seed=args.seed,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_repo_id,
            group_by_length=True,
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
            dataloader_drop_last=True,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            max_steps=max_steps_value,
            ddp_find_unused_parameters=False,
        )

    # HF Hub prep
    if args.push_to_hub and args.hub_repo_id:
        token = args.hf_token
        if token and HfFolder is not None:
            try:
                HfFolder.save_token(token)
            except Exception:
                pass
        if create_repo is not None:
            try:
                create_repo(args.hub_repo_id, private=args.hub_private, exist_ok=True, token=token)
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

    # Train, evaluate, save
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    try:
        metrics = trainer.evaluate()
        print(f"\n📊 Eval metrics: {metrics}")
    except Exception:
        pass
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Push to hub at end
    if args.push_to_hub and args.hub_repo_id:
        try:
            trainer.push_to_hub(commit_message="ByT5-small fine-tuned for typo correction")
        except Exception as e:
            print(f"⚠️  Push to hub failed: {e}")

    print(f"✅ Training complete. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

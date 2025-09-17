#!/usr/bin/env python3
"""
Enhanced Qwen Training Script

Fine-tunes Qwen 0.6B for typo correction using the enhanced dataset with T5 improvements.
Includes advanced data generation, punctuation balancing, and multi-domain training.
"""

import os
import json
import torch
import torch.distributed as dist
import logging

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
# import wandb  # Disabled

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QwenTrainingConfig:
    """Configuration for Qwen training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-0.6B-Base"  # Base model (not instruction-tuned) for conservative behavior
    model_max_length: int = 512

    # Training data
    train_file: str = "data/enhanced_qwen_training.jsonl"
    eval_split: float = 0.1  # 10% for evaluation

    # Cached tokens configuration
    use_cached_tokens: bool = False
    cached_tokens_dir: str = "data/tokenized_cache"

    # Training hyperparameters
    output_dir: str = "models/qwen-base-typo-fixer"
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Evaluation and logging
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Optimization
    fp16: bool = True
    dataloader_num_workers: int = 4
    group_by_length: bool = True

    # Wandb logging
    report_to: str = "wandb"
    run_name: str = "qwen-base-typo-fixer-4epochs"

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    # Additional config fields that might be in JSON
    dataloader_pin_memory: bool = False
    eval_accumulation_steps: int = 1

def load_enhanced_dataset(file_path: str, eval_split: float = 0.1) -> DatasetDict:
    """Load the enhanced Qwen training dataset."""

    logger.info(f"Loading dataset from {file_path}...")

    # Load JSONL data
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                examples.append(data)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON on line {line_num}")
                continue

    logger.info(f"Loaded {len(examples):,} examples")

    # Convert to datasets format
    processed_examples = []
    for example in examples:
        # Extract messages
        messages = example.get('messages', [])
        if len(messages) != 2:
            continue

        user_msg = messages[0]['content']
        assistant_msg = messages[1]['content']

        # Create conversation format for Qwen
        conversation = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"

        processed_examples.append({
            'text': conversation,
            'metadata': example.get('metadata', {})
        })

    logger.info(f"Processed {len(processed_examples):,} examples")

    # Create Dataset
    dataset = Dataset.from_list(processed_examples)

    # Split into train/eval
    if eval_split > 0:
        dataset = dataset.train_test_split(test_size=eval_split, seed=42)
        dataset_dict = DatasetDict({
            'train': dataset['train'],
            'eval': dataset['test']
        })
    else:
        dataset_dict = DatasetDict({'train': dataset})

    logger.info(f"Dataset splits: train={len(dataset_dict['train']):,}, eval={len(dataset_dict.get('eval', [])):,}")

    return dataset_dict

def preprocess_function(examples: Dict, tokenizer: AutoTokenizer, max_length: int = 512) -> Dict:
    """Preprocess examples for training with proper padding and truncation."""

    # Tokenize the text
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',  # Pad to max_length for consistent tensor sizes
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=True
    )

    # For causal LM, labels are the same as input_ids
    # But we need to ignore padding tokens in loss calculation
    labels = tokenized['input_ids'].copy()

    # Replace padding token ids with -100 so they're ignored in loss calculation
    for i, input_ids in enumerate(labels):
        labels[i] = [token_id if token_id != tokenizer.pad_token_id else -100 for token_id in input_ids]

    tokenized['labels'] = labels

    return tokenized

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions, labels = eval_pred

    # For now, just compute perplexity
    # Could add more sophisticated metrics later
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)

    # Compute loss (negative log likelihood)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)

    # Reshape predictions and labels for loss computation
    shift_predictions = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten for loss computation
    shift_predictions = shift_predictions.view(-1, shift_predictions.size(-1))
    shift_labels = shift_labels.view(-1)

    loss = loss_fn(shift_predictions, shift_labels)
    perplexity = torch.exp(loss).item()

    return {
        'perplexity': perplexity,
        'eval_loss': loss.item()
    }

def analyze_dataset_metadata(dataset: Dataset) -> Dict:
    """Analyze metadata from the enhanced dataset."""

    logger.info("Analyzing dataset metadata...")

    metadata_stats = {
        'domain_distribution': {},
        'complexity_distribution': {},
        'error_type_distribution': {},
        'source_distribution': {},
        'difficulty_stats': {'scores': []},
        'error_count_stats': {'counts': []},
    }

    for example in dataset:
        metadata = example.get('metadata', {})

        # Domain distribution
        domain = metadata.get('domain', 'unknown')
        metadata_stats['domain_distribution'][domain] = metadata_stats['domain_distribution'].get(domain, 0) + 1

        # Complexity distribution
        complexity = metadata.get('complexity', 'unknown')
        metadata_stats['complexity_distribution'][complexity] = metadata_stats['complexity_distribution'].get(complexity, 0) + 1

        # Error types (can be multiple)
        error_types = metadata.get('error_types', [])
        for error_type in error_types:
            metadata_stats['error_type_distribution'][error_type] = metadata_stats['error_type_distribution'].get(error_type, 0) + 1

        # Source distribution
        source = metadata.get('source', 'unknown')
        metadata_stats['source_distribution'][source] = metadata_stats['source_distribution'].get(source, 0) + 1

        # Numerical stats
        if 'difficulty_score' in metadata:
            metadata_stats['difficulty_stats']['scores'].append(metadata['difficulty_score'])

        if 'num_errors' in metadata:
            metadata_stats['error_count_stats']['counts'].append(metadata['num_errors'])

    # Calculate summary statistics
    if metadata_stats['difficulty_stats']['scores']:
        scores = metadata_stats['difficulty_stats']['scores']
        metadata_stats['difficulty_stats']['mean'] = sum(scores) / len(scores)
        metadata_stats['difficulty_stats']['min'] = min(scores)
        metadata_stats['difficulty_stats']['max'] = max(scores)

    if metadata_stats['error_count_stats']['counts']:
        counts = metadata_stats['error_count_stats']['counts']
        metadata_stats['error_count_stats']['mean'] = sum(counts) / len(counts)
        metadata_stats['error_count_stats']['min'] = min(counts)
        metadata_stats['error_count_stats']['max'] = max(counts)

    return metadata_stats

def print_dataset_analysis(stats: Dict):
    """Print analysis of the dataset."""

    print("\n📊 ENHANCED DATASET ANALYSIS")
    print("=" * 50)

    print(f"\n🏷️ Domain Distribution:")
    for domain, count in sorted(stats['domain_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain:15}: {count:6,}")

    print(f"\n📈 Complexity Distribution:")
    for complexity, count in sorted(stats['complexity_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {complexity:15}: {count:6,}")

    print(f"\n🐛 Error Type Distribution:")
    for error_type, count in sorted(stats['error_type_distribution'].items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"  {error_type:15}: {count:6,}")

    if 'mean' in stats['difficulty_stats']:
        print(f"\n💪 Difficulty Statistics:")
        print(f"  Mean: {stats['difficulty_stats']['mean']:.1f}")
        print(f"  Range: {stats['difficulty_stats']['min']:.1f} - {stats['difficulty_stats']['max']:.1f}")

    if 'mean' in stats['error_count_stats']:
        print(f"\n🔢 Error Count Statistics:")
        print(f"  Mean errors per example: {stats['error_count_stats']['mean']:.1f}")
        print(f"  Range: {stats['error_count_stats']['min']} - {stats['error_count_stats']['max']}")

def train_enhanced_qwen(config: QwenTrainingConfig):
    """Train Qwen with the enhanced dataset."""

    # Initialize distributed training if using torchrun
    is_distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', -1))

    if is_distributed:
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        # Only print on main process
        if local_rank != 0:
            logger.setLevel(logging.WARNING)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = -1

    # Print training info only on main process
    if local_rank <= 0:
        print("🚀 Starting Enhanced Qwen Training")
        print("=" * 50)
        print(f"Model: {config.model_name}")
        print(f"Dataset: {config.train_file}")
        print(f"Output: {config.output_dir}")
        if is_distributed:
            print(f"Distributed Training: {os.environ.get('WORLD_SIZE', 1)} GPUs")
        print()

    # Initialize wandb if configured
    # if config.report_to == "wandb":
    #     wandb.init(
    #         project="qwen-typo-fixer",
    #         name=config.run_name,
    #         config=config.__dict__
    #     )

    # Load tokenizer and model
    logger.info(f"Loading model and tokenizer: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side='right'
    )

    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model - no device_map for distributed training
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # Use FP32 for stability (RTX5090 has plenty of VRAM)
        trust_remote_code=True,
        device_map=None  # Never use device_map with distributed training
    )

    # Don't move model to device yet - let Trainer/Accelerate handle it for DDP
    # This prevents CUDA illegal memory access errors during DDP initialization

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    # Load and preprocess dataset
    if config.use_cached_tokens:
        # Load pre-tokenized dataset from cache
        logger.info(f"Loading cached tokenized dataset from {config.cached_tokens_dir}")
        from datasets import load_from_disk

        cached_dataset_path = Path(config.cached_tokens_dir) / "tokenized_dataset"
        if not cached_dataset_path.exists():
            raise FileNotFoundError(
                f"Cached dataset not found at {cached_dataset_path}. "
                f"Please run: python pretokenize_dataset.py --output-dir {config.cached_tokens_dir}"
            )

        tokenized_dataset = load_from_disk(str(cached_dataset_path))

        # Load metadata for stats
        metadata_path = Path(config.cached_tokens_dir) / "metadata.json"
        if metadata_path.exists() and local_rank <= 0:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"\n📊 Using cached dataset:")
            print(f"  Train samples: {metadata.get('train_size', 'Unknown'):,}")
            print(f"  Eval samples: {metadata.get('eval_size', 'Unknown'):,}")
            print(f"  Max length: {metadata.get('max_length', 'Unknown')}")
            print(f"  Model: {metadata.get('model_name', 'Unknown')}\n")

        train_stats = {}  # No need to analyze, already done during pre-tokenization
    else:
        # Original tokenization flow
        logger.info("Loading and preprocessing dataset...")
        dataset = load_enhanced_dataset(config.train_file, config.eval_split)

        # Analyze dataset - only on main process
        if local_rank <= 0:
            train_stats = analyze_dataset_metadata(dataset['train'])
            print_dataset_analysis(train_stats)
        else:
            train_stats = {}

        # Tokenize dataset
        def tokenize_function(examples):
            return preprocess_function(examples, tokenizer, config.model_max_length)

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing dataset"
        )

    # Data collator with proper padding handling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8 if config.fp16 else None,
        return_tensors="pt"
    )

    # Training arguments - with distributed training support
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        group_by_length=config.group_by_length,
        report_to=config.report_to,
        run_name=config.run_name,
        seed=42,
        dataloader_pin_memory=getattr(config, 'dataloader_pin_memory', False),
        eval_accumulation_steps=getattr(config, 'eval_accumulation_steps', 1),
        # Distributed training settings
        local_rank=local_rank,
        ddp_find_unused_parameters=True,  # Changed to True for better compatibility
        ddp_backend='nccl' if is_distributed else None,
        remove_unused_columns=False,  # Prevent column removal issues
        # Additional settings for stability
        gradient_checkpointing=False,
        fsdp="" if not is_distributed else "",  # Disable FSDP
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset.get('eval'),
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of deprecated tokenizer
        compute_metrics=None,  # Disable compute_metrics to avoid OOM during eval
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold
            )
        ] if 'eval' in tokenized_dataset else None,
    )

    # Check for existing checkpoint
    checkpoint = None
    if config.resume_from_checkpoint:
        checkpoint = config.resume_from_checkpoint
    elif os.path.isdir(config.output_dir):
        checkpoint = get_last_checkpoint(config.output_dir)
        if checkpoint:
            logger.info(f"Found checkpoint: {checkpoint}")

    # Train the model
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save the final model - only on main process
    if local_rank <= 0:
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)

        # Save training statistics
        stats_file = Path(config.output_dir) / "training_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                'config': config.__dict__,
                'dataset_stats': train_stats,
                'final_metrics': trainer.state.log_history[-1] if trainer.state.log_history else {}
            }, f, indent=2)

        print(f"\n✅ Training complete!")
        print(f"📁 Model saved to: {config.output_dir}")
        print(f"📊 Stats saved to: {stats_file}")

    # if config.report_to == "wandb":
    #     wandb.finish()

    # Clean up distributed training
    if is_distributed:
        dist.destroy_process_group()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Qwen with enhanced dataset")
    parser.add_argument('--config-file', type=str, help='JSON config file')
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--train-file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--resume-from-checkpoint', type=str, help='Resume training from checkpoint')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--use-cached-tokens', action='store_true', help='Use pre-tokenized cached dataset')
    parser.add_argument('--cached-tokens-dir', type=str, default='data/tokenized_cache', help='Directory with cached tokens')

    args = parser.parse_args()

    # Load config
    if args.config_file and os.path.exists(args.config_file):
        print(f"📄 Loading config from: {args.config_file}")
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = QwenTrainingConfig(**config_dict)
    else:
        print("⚠️  Using default config")
        config = QwenTrainingConfig()

    # Override with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.train_file:
        config.train_file = args.train_file
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_train_epochs = args.num_epochs
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    if args.no_wandb:
        config.report_to = "none"
    if args.use_cached_tokens:
        config.use_cached_tokens = True
    if args.cached_tokens_dir:
        config.cached_tokens_dir = args.cached_tokens_dir

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Start training
    train_enhanced_qwen(config)

if __name__ == "__main__":
    main()
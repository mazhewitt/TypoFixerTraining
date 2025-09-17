# Token Caching System for Fast Training Startup

## Overview
The training pipeline now includes an automatic token caching system that dramatically reduces training startup time from ~2+ minutes to just seconds!

## How It Works

### Automatic Cache Management
The `run_dual_gpu_training.sh` script now:

1. **Checks for existing cache** - Looks for pre-tokenized data in `data/tokenized_cache/`
2. **Validates cache** - Ensures cache matches the current model
3. **Generates if needed** - Automatically creates cache if missing or invalid
4. **Uses cache** - Loads pre-tokenized data for instant training startup

### Benefits
- ⚡ **Instant startup**: Skip 2+ minutes of tokenization on every training run
- 💾 **Efficient storage**: Only 0.52 GB for entire tokenized dataset
- 🔄 **Automatic**: Script handles everything - no manual steps needed
- ✅ **Validated**: Ensures cache matches your model configuration

## Usage

### Simple Usage
Just run the training script as usual:
```bash
./run_dual_gpu_training.sh
```

The script will:
- Use cache if available (instant startup!)
- Generate cache if needed (first run only)
- Validate cache matches your model

### With Custom Config
```bash
./run_dual_gpu_training.sh your_config.json
```

### Manual Cache Management

#### Pre-generate Cache
```bash
python pretokenize_dataset.py \
    --model-name "Qwen/Qwen3-0.6B-Base" \
    --train-file "data/enhanced_qwen_training.jsonl" \
    --output-dir "data/tokenized_cache"
```

#### Force Regenerate Cache
```bash
rm -rf data/tokenized_cache
./run_dual_gpu_training.sh
```

#### Use Cache in Python Script
```bash
# With config file that has use_cached_tokens: true
torchrun --nproc_per_node=2 train_enhanced_qwen.py \
    --config-file training_config_dual_gpu_cached.json

# Or with command line args
torchrun --nproc_per_node=2 train_enhanced_qwen.py \
    --use-cached-tokens \
    --cached-tokens-dir data/tokenized_cache
```

## Cache Structure
```
data/tokenized_cache/
├── tokenized_dataset/     # Arrow format tokenized data
│   ├── train/
│   └── eval/
├── metadata.json          # Cache metadata (model, sizes, etc.)
└── tokenizer/            # Reference tokenizer config
```

## Performance Impact

| Operation | Without Cache | With Cache | Speedup |
|-----------|--------------|------------|---------|
| Dataset Loading | ~2-3 minutes | <1 second | ~200x |
| Total Startup | ~3-4 minutes | ~30 seconds | ~7x |
| Disk Usage | 0 GB | 0.52 GB | - |

## Troubleshooting

### Cache not being used?
Check that:
1. Cache exists: `ls -la data/tokenized_cache/`
2. Model matches: Check `model_name` in `data/tokenized_cache/metadata.json`
3. Config is correct: Ensure `use_cached_tokens: true` in config

### Want fresh tokenization?
Simply delete the cache:
```bash
rm -rf data/tokenized_cache
```

### Different model?
The cache is model-specific. When you change models, the script will automatically detect this and regenerate the cache.

## Technical Details

The caching system:
- Uses HuggingFace datasets' Arrow format for efficient storage
- Includes all tokenization, padding, and label preparation
- Preserves exact tokenization parameters (max_length, padding strategy, etc.)
- Supports multi-process tokenization for fast cache generation
- Works seamlessly with distributed training (DDP)
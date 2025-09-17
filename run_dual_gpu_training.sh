#!/bin/bash

# Run dual GPU training with torchrun
# This script automatically handles dataset caching and launches training on 2 GPUs

echo "🚀 Starting dual GPU training with torchrun"
echo "=================================================="

# Configuration
CACHE_DIR="data/tokenized_cache"
CACHE_METADATA="$CACHE_DIR/metadata.json"
CONFIG_FILE="${1:-training_config_dual_gpu_test.json}"
MODEL_NAME="Qwen/Qwen3-0.6B-Base"
TRAIN_FILE="data/enhanced_qwen_training.jsonl"

# Check if cached tokens exist
if [ -f "$CACHE_METADATA" ]; then
    echo "✅ Found cached tokenized dataset at: $CACHE_DIR"

    # Verify cache is for correct model
    CACHED_MODEL=$(python3 -c "import json; print(json.load(open('$CACHE_METADATA'))['model_name'])" 2>/dev/null)

    if [ "$CACHED_MODEL" = "$MODEL_NAME" ]; then
        echo "✅ Cache is valid for model: $MODEL_NAME"

        # Display cache info
        echo ""
        echo "📊 Cache Statistics:"
        python3 -c "
import json
with open('$CACHE_METADATA') as f:
    meta = json.load(f)
    print(f'  Train samples: {meta[\"train_size\"]:,}')
    print(f'  Eval samples: {meta[\"eval_size\"]:,}')
    print(f'  Max length: {meta[\"max_length\"]}')
"
        USE_CACHE=true
    else
        echo "⚠️  Cache is for different model ($CACHED_MODEL), regenerating..."
        USE_CACHE=false
    fi
else
    echo "ℹ️  No cached dataset found, will generate cache first..."
    USE_CACHE=false
fi

# Generate cache if needed
if [ "$USE_CACHE" = false ]; then
    echo ""
    echo "🔄 Pre-tokenizing dataset (this will take a few minutes)..."
    echo "--------------------------------------------------"

    python3 pretokenize_dataset.py \
        --model-name "$MODEL_NAME" \
        --train-file "$TRAIN_FILE" \
        --output-dir "$CACHE_DIR" \
        --max-length 512 \
        --eval-split 0.1 \
        --batch-size 1000

    if [ $? -eq 0 ]; then
        echo "✅ Successfully cached tokenized dataset!"
        USE_CACHE=true
    else
        echo "❌ Failed to create cache, will tokenize during training"
        USE_CACHE=false
    fi
fi

echo ""
echo "🚀 Starting distributed training on 2 GPUs"
echo "--------------------------------------------------"

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1

# Run with torchrun, using cache if available
if [ "$USE_CACHE" = true ]; then
    echo "📦 Using cached tokens for faster startup..."

    # Check if config file has use_cached_tokens
    if grep -q "use_cached_tokens" "$CONFIG_FILE"; then
        # Config already has cache settings, use as is
        torchrun --nproc_per_node=2 \
            --master_port=29500 \
            train_enhanced_qwen.py \
            --config-file "$CONFIG_FILE"
    else
        # Add cache arguments via command line
        torchrun --nproc_per_node=2 \
            --master_port=29500 \
            train_enhanced_qwen.py \
            --config-file "$CONFIG_FILE" \
            --use-cached-tokens \
            --cached-tokens-dir "$CACHE_DIR"
    fi
else
    echo "⏳ Will tokenize during training (slower startup)..."
    torchrun --nproc_per_node=2 \
        --master_port=29500 \
        train_enhanced_qwen.py \
        --config-file "$CONFIG_FILE"
fi

echo ""
echo "✅ Training script completed!"
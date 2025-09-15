#!/bin/bash

# Improved ByT5 training script for dual RTX 5090 CUDA server
# Optimized for high-performance training with proper monitoring

set -e

echo "🚀 IMPROVED BYT5 TYPO FIXER TRAINING"
echo "💻 Optimized for dual RTX 5090 CUDA setup"
echo "==============================================="

# Check CUDA setup
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ CUDA/nvidia-smi not found"
    exit 1
fi

echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Check Python and dependencies
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

echo "🐍 Python version: $(python3 --version)"
echo "🔧 PyTorch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "🔧 GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Configuration
MODEL_NAME="google/byt5-small"
TRAIN_FILE="data/enhanced_training_full.jsonl"
OUTPUT_DIR="models/byt5-small-typo-fixer-v2"
HUB_MODEL_ID="mazhewitt/byt5-small-typo-fixer-v2"

# Training parameters optimized for dual RTX 5090
PER_DEVICE_BATCH_SIZE=8        # Conservative for 24GB VRAM
GRADIENT_ACCUMULATION=4        # Effective batch size = 8 * 4 * 2 = 64
LEARNING_RATE=5e-5             # Lower for ByT5
NUM_EPOCHS=5                   # More epochs for character-level
MAX_LENGTH=512                 # Longer sequences
WARMUP_RATIO=0.1               # Longer warmup

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# Generate timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/byt5_training_${TIMESTAMP}.log"

echo "📋 Training Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Data: $TRAIN_FILE"
echo "   Output: $OUTPUT_DIR"
echo "   Hub ID: $HUB_MODEL_ID"
echo "   Batch Size: $PER_DEVICE_BATCH_SIZE per GPU"
echo "   Effective Batch Size: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION * 2))"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Epochs: $NUM_EPOCHS"
echo "   Max Length: $MAX_LENGTH"
echo "   Log File: $LOG_FILE"
echo ""

# Check if training data exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Training file not found: $TRAIN_FILE"
    echo "🔧 Please ensure training data is available"
    exit 1
fi

echo "📊 Training data info:"
echo "   Lines: $(wc -l < $TRAIN_FILE)"
echo "   Size: $(ls -lh $TRAIN_FILE | awk '{print $5}')"
echo ""

# Set environment variables for optimal CUDA performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1
export NCCL_DEBUG=INFO

echo "🚀 Starting training..."
echo "⏰ Start time: $(date)"
echo ""

# Run training with logging
python3 t5/train_byt5_improved.py \
    --model-name "$MODEL_NAME" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --prefix "fix typos:" \
    --max-source-len $MAX_LENGTH \
    --max-target-len $MAX_LENGTH \
    --learning-rate $LEARNING_RATE \
    --weight-decay 0.01 \
    --num-epochs $NUM_EPOCHS \
    --warmup-ratio $WARMUP_RATIO \
    --per-device-train-batch-size $PER_DEVICE_BATCH_SIZE \
    --per-device-eval-batch-size 16 \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --num-workers 4 \
    --eval-steps 500 \
    --save-steps 500 \
    --logging-steps 50 \
    --gradient-checkpointing \
    --push-to-hub \
    --hub-model-id "$HUB_MODEL_ID" \
    2>&1 | tee "$LOG_FILE"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "⏰ End time: $(date)"

# Check if training was successful
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Run comprehensive test
    echo ""
    echo "🧪 Running post-training evaluation..."
    python3 scripts/test_byt5_on_server.py \
        --model-path "$OUTPUT_DIR" \
        --prefix "fix typos:" \
        --output-file "$OUTPUT_DIR/test_results.json"
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "🎉 Model passed quality tests!"
        echo "🚀 Model successfully uploaded to: https://huggingface.co/$HUB_MODEL_ID"
    else
        echo "⚠️  Model needs improvement (check test results)"
    fi
    
    # Show training summary if it exists
    if [ -f "$OUTPUT_DIR/training_summary.json" ]; then
        echo ""
        echo "📊 Training Summary:"
        python3 -c "
import json
with open('$OUTPUT_DIR/training_summary.json') as f:
    data = json.load(f)
    print(f\"   Training time: {data.get('training_time_minutes', 0):.1f} minutes\")
    print(f\"   Effective batch size: {data.get('effective_batch_size', 'unknown')}\")
    print(f\"   GPUs used: {data.get('num_gpus', 'unknown')}\")
    if 'final_results' in data:
        print(f\"   Final test samples:\")
        for orig, corr in data['final_results'][:3]:
            print(f\"     '{orig}' → '{corr}'\")
"
    fi
    
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "📋 Check the log file: $LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi

echo ""
echo "📁 Files created:"
echo "   Model: $OUTPUT_DIR/"
echo "   Log: $LOG_FILE"
echo "   Test Results: $OUTPUT_DIR/test_results.json"
echo "   Training Summary: $OUTPUT_DIR/training_summary.json"
echo ""
echo "🎯 Next steps:"
echo "   1. Review test results in $OUTPUT_DIR/test_results.json"
echo "   2. Download and test locally: huggingface-cli download $HUB_MODEL_ID"
echo "   3. Compare with previous Qwen model performance"
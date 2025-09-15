#!/bin/bash

# SIMPLIFIED BYT5 TRAINING SCRIPT
# Just focus on getting a working model without complex pipeline

set -e

echo "🚀 SIMPLIFIED BYT5 TYPO FIXER TRAINING"
echo "💻 Dual RTX 5090 optimized"
echo "=" * 50

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ CUDA required"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits
echo ""

# Configuration
NUM_EXAMPLES=100000
OUTPUT_DIR="models/byt5-small-typo-fixer-v3"
HUB_MODEL_ID="mazhewitt/byt5-small-typo-fixer-v3"

echo "📋 Configuration:"
echo "   Examples: $NUM_EXAMPLES"
echo "   Model: $HUB_MODEL_ID"
echo ""

# STEP 1: Generate clean training data
echo "🔧 Generating training data..."
python3 scripts/generate_100k_dataset_fixed.py \
    --num-examples $NUM_EXAMPLES \
    --corruption-rate 0.15

if [ ! -f "data/enhanced_training_full.jsonl" ]; then
    echo "❌ Data generation failed"
    exit 1
fi

FULL_COUNT=$(wc -l < data/enhanced_training_full.jsonl)
echo "✅ Generated $FULL_COUNT examples"

# STEP 2: Create balanced dataset
echo "🔧 Creating balanced dataset..."
python3 t5/create_balanced_dataset.py

if [ ! -f "data/enhanced_training_balanced.jsonl" ]; then
    echo "❌ Balanced dataset creation failed"
    exit 1
fi

BALANCED_COUNT=$(wc -l < data/enhanced_training_balanced.jsonl)
echo "✅ Balanced dataset: $BALANCED_COUNT examples"

# STEP 3: Train model (simple PyTorch training)
echo "🚀 Training ByT5 model..."
echo "⏰ Start: $(date)"

mkdir -p "$OUTPUT_DIR"

python3 scripts/train_byt5_simple.py \
    --train-file "data/enhanced_training_balanced.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "google/byt5-small" \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --epochs 3 \
    --max-length 256 \
    --prefix "fix typos:" \
    # --hub-model-id "$HUB_MODEL_ID"

TRAINING_EXIT=$?

echo "⏰ End: $(date)"

if [ $TRAINING_EXIT -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Test the model
    echo "🧪 Testing model..."
    python3 scripts/test_byt5_on_server.py \
        --model-path "$OUTPUT_DIR" \
        --prefix "fix typos:" \
        --output-file "$OUTPUT_DIR/test_results.json"
    
    echo ""
    echo "🎉 SUCCESS!"
    echo "📁 Model: $OUTPUT_DIR"
    echo "🚀 HuggingFace: https://huggingface.co/$HUB_MODEL_ID"
    echo ""
    echo "🧪 Quick test results:"
    if [ -f "$OUTPUT_DIR/test_results.json" ]; then
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/test_results.json') as f:
        data = json.load(f)
    print(f'   Word accuracy: {data.get(\"avg_word_accuracy\", 0)*100:.1f}%')
    print(f'   Sentence accuracy: {data.get(\"avg_sentence_accuracy\", 0)*100:.1f}%')
    print(f'   Performance: {data.get(\"performance_level\", \"unknown\")}')
except:
    print('   Test results not available')
"
    fi
else
    echo "❌ Training failed"
    exit 1
fi
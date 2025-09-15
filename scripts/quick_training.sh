#!/bin/bash

# Quick single-GPU training to get something working
set -e

echo "🚀 QUICK BYT5 TRAINING (Single GPU)"
echo "⚡ Faster training for testing"
echo "=" * 40

# Check dataset
if [ ! -f "data/enhanced_training_balanced.jsonl" ]; then
    echo "❌ Dataset not found"
    exit 1
fi

EXAMPLES=$(wc -l < data/enhanced_training_balanced.jsonl)
echo "✅ Dataset: $EXAMPLES examples"

OUTPUT_DIR="models/byt5-quick-test"
echo "📁 Output: $OUTPUT_DIR"
echo "⏰ Start: $(date)"

# Kill any existing training processes to free up GPU memory
pkill -f "python.*train" || true
sleep 2

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Quick training with single GPU
python3 scripts/train_byt5_single_gpu.py \
    --train-file "data/enhanced_training_balanced.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "google/byt5-small" \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --num-epochs 2 \
    --max-length 128 \
    --prefix "fix typos:"

TRAINING_EXIT=$?
echo "⏰ End: $(date)"

if [ $TRAINING_EXIT -eq 0 ]; then
    echo "✅ Quick training successful!"
    
    # Show results
    if [ -f "$OUTPUT_DIR/simple_results.json" ]; then
        echo "📊 Results:"
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/simple_results.json') as f:
        data = json.load(f)
    print(f'   Training time: {data.get(\"training_time_minutes\", 0):.1f} minutes')
    print(f'   Training samples: {data.get(\"training_samples\", 0):,}')
    print(f'   Model saved: {data.get(\"model_path\", \"unknown\")}')
except:
    print('   Results not available')
"
    fi
    
    echo ""
    echo "🎯 Next: Test the model quality:"
    echo "   python3 scripts/test_byt5_on_server.py --model-path $OUTPUT_DIR"
    
else
    echo "❌ Training failed"
    exit 1
fi
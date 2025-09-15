#!/bin/bash

# Stable training to fix loss collapse issue
set -e

echo "🛡️  STABLE BYT5 TRAINING"
echo "🔍 Fixed loss collapse and gradient issues"
echo "=" * 40

# Kill any existing processes
pkill -f "python.*train" || true
sleep 2

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Check dataset
if [ ! -f "data/enhanced_training_balanced.jsonl" ]; then
    echo "❌ Dataset not found"
    exit 1
fi

OUTPUT_DIR="models/byt5-stable-typo-fixer"
echo "📁 Output: $OUTPUT_DIR"
echo "⏰ Start: $(date)"

# Stable training with conservative settings
python3 scripts/train_byt5_stable.py \
    --train-file "data/enhanced_training_balanced.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "google/byt5-small" \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --num-epochs 2 \
    --max-length 128 \
    --prefix "fix typos:"

TRAINING_EXIT=$?
echo "⏰ End: $(date)"

if [ $TRAINING_EXIT -eq 0 ]; then
    echo "✅ Stable training successful!"
    
    # Test the model
    echo "🧪 Testing model quality..."
    python3 scripts/test_byt5_on_server.py \
        --model-path "$OUTPUT_DIR" \
        --prefix "fix typos:" \
        --output-file "$OUTPUT_DIR/test_results.json"
    
    echo ""
    echo "🎉 SUCCESS! Model should now have proper loss behavior."
    
else
    echo "❌ Training failed"
    exit 1
fi
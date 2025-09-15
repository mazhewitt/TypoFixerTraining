#!/bin/bash

# Quick restart training without HuggingFace upload
# The dataset is already created, so just train the model

set -e

echo "🔄 RESTARTING BYT5 TRAINING (No Hub Upload)"
echo "📊 Using existing balanced dataset"
echo "=" * 50

# Check if dataset exists
if [ ! -f "data/enhanced_training_balanced.jsonl" ]; then
    echo "❌ Balanced dataset not found. Run data generation first:"
    echo "   python3 scripts/create_full_dataset_and_train.py"
    exit 1
fi

BALANCED_COUNT=$(wc -l < data/enhanced_training_balanced.jsonl)
echo "✅ Found dataset: $BALANCED_COUNT examples"

# Configuration
OUTPUT_DIR="models/byt5-small-typo-fixer-v3"
HUB_MODEL_ID="mazhewitt/byt5-small-typo-fixer-v3"

echo "🚀 Training ByT5 model (local only)..."
echo "⏰ Start: $(date)"

mkdir -p "$OUTPUT_DIR"

# Train with minimal script (no complex callbacks)
python3 scripts/train_byt5_minimal.py \
    --train-file "data/enhanced_training_balanced.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --prefix "fix typos:" \
    --max-source-len 256 \
    --max-target-len 256 \
    --learning-rate 5e-5 \
    --num-epochs 3 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 4

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
    echo ""
    
    # Show test results
    if [ -f "$OUTPUT_DIR/test_results.json" ]; then
        echo "🧪 Test Results:"
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/test_results.json') as f:
        data = json.load(f)
    print(f'   Word accuracy: {data.get(\"avg_word_accuracy\", 0)*100:.1f}%')
    print(f'   Sentence accuracy: {data.get(\"avg_sentence_accuracy\", 0)*100:.1f}%')
    print(f'   Performance: {data.get(\"performance_level\", \"unknown\")}')
    print(f'   Perfect sentences: {data.get(\"perfect_sentences\", 0)}/{data.get(\"test_cases\", 0)}')
except:
    print('   Test results not available')
"
    fi
    
    echo ""
    echo "📤 To upload to HuggingFace later:"
    echo "   python3 scripts/upload_to_hub.py --model-path $OUTPUT_DIR --hub-model-id $HUB_MODEL_ID"
    
else
    echo "❌ Training failed"
    exit 1
fi
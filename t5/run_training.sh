#!/bin/bash

# T5-efficient-tiny typo correction training script for M4 MacBook
# Optimized for minimal memory usage and fast training

set -e

echo "🚀 Starting T5-efficient-tiny typo correction training..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed."
    exit 1
fi

# Check if balanced training data exists
if [ ! -f "data/enhanced_training_balanced.jsonl" ]; then
    echo "⚠️  Balanced training data not found: data/enhanced_training_balanced.jsonl"
    echo "🔧 Creating balanced dataset from enhanced_training_full.jsonl..."
    python3 create_balanced_dataset.py
    
    if [ ! -f "data/enhanced_training_balanced.jsonl" ]; then
        echo "❌ Failed to create balanced dataset"
        exit 1
    fi
fi

# Create output directory
OUTPUT_DIR="models/t5-typo-fixer"
mkdir -p "$OUTPUT_DIR"

echo "📊 Training configuration:"
echo "   Model: google/t5-efficient-tiny"
echo "   Data: data/enhanced_training_balanced.jsonl (50/50 punctuation split)"
echo "   Output: $OUTPUT_DIR"
echo "   Device: Apple M4 MacBook (MPS)"
echo ""

# Run training
python3 scripts/train_t5_tiny.py \
    --model_name "google/t5-efficient-tiny" \
    --train_file "data/enhanced_training_balanced.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --max_source_length 64 \
    --max_target_length 64 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --warmup_ratio 0.1 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 50 \
    --target_accuracy 0.85 \
    --early_stopping_patience 3 \
    --weight_decay 0.01

echo ""
echo "✅ Training completed!"
echo "📁 Model saved to: $OUTPUT_DIR"
echo ""
echo "🧪 To test the model:"
echo "   cd $OUTPUT_DIR"
echo "   python3 -c \"from transformers import T5Tokenizer, T5ForConditionalGeneration; model = T5ForConditionalGeneration.from_pretrained('.'); tokenizer = T5Tokenizer.from_pretrained('.'); inputs = tokenizer('correct typos: I beleive this is teh answr.', return_tensors='pt'); outputs = model.generate(**inputs, max_length=64); print(tokenizer.decode(outputs[0], skip_special_tokens=True))\""
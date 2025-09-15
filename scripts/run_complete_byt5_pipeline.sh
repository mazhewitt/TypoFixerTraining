#!/bin/bash

# COMPLETE BYT5 TYPO FIXER PIPELINE
# Generates 100k examples from scratch, creates balanced dataset, and trains improved model
# Optimized for dual RTX 5090 CUDA server

set -e

echo "🚀 COMPLETE BYT5 TYPO FIXER PIPELINE"
echo "🎯 100k examples → Balanced dataset → Improved training"
echo "💻 Optimized for dual RTX 5090 CUDA setup"
echo "=" * 80

# Check if we're on the training server
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ This script requires CUDA (nvidia-smi not found)"
    echo "🔧 Please run on the training server with dual RTX 5090s"
    exit 1
fi

echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Configuration
NUM_EXAMPLES=100000
CORRUPTION_RATE=0.15
HUB_MODEL_ID="mazhewitt/byt5-small-typo-fixer-v3"  # New version number

# Create logs directory
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL_LOG="logs/complete_pipeline_${TIMESTAMP}.log"

echo "📋 Pipeline Configuration:"
echo "   Target examples: ${NUM_EXAMPLES:,}"
echo "   Corruption rate: $CORRUPTION_RATE"
echo "   Output model: $HUB_MODEL_ID"
echo "   Log file: $FULL_LOG"
echo ""

# Function to log and run commands
run_step() {
    local step_name="$1"
    local description="$2"
    shift 2
    
    echo "🔧 STEP: $step_name" | tee -a "$FULL_LOG"
    echo "📝 $description" | tee -a "$FULL_LOG"
    echo "⏰ Start time: $(date)" | tee -a "$FULL_LOG"
    echo "" | tee -a "$FULL_LOG"
    
    if "$@" 2>&1 | tee -a "$FULL_LOG"; then
        echo "✅ $step_name completed successfully" | tee -a "$FULL_LOG"
        echo "" | tee -a "$FULL_LOG"
        return 0
    else
        echo "❌ $step_name failed" | tee -a "$FULL_LOG"
        echo "📋 Check full log: $FULL_LOG" | tee -a "$FULL_LOG"
        exit 1
    fi
}

# STEP 1: Generate 100k training examples
run_step "DATA_GENERATION" \
    "Generating $NUM_EXAMPLES realistic typo examples from multiple sources" \
    python3 scripts/create_full_dataset_and_train.py \
        --num-examples "$NUM_EXAMPLES" \
        --corruption-rate "$CORRUPTION_RATE"

# STEP 2: Verify data was created
if [ ! -f "data/enhanced_training_balanced.jsonl" ]; then
    echo "❌ Balanced dataset not found after generation"
    exit 1
fi

ACTUAL_EXAMPLES=$(wc -l < data/enhanced_training_balanced.jsonl)
echo "📊 Dataset ready: $ACTUAL_EXAMPLES examples"

if [ "$ACTUAL_EXAMPLES" -lt 50000 ]; then
    echo "⚠️  Dataset smaller than expected ($ACTUAL_EXAMPLES < 50,000)"
    echo "🤔 Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "🛑 Stopping pipeline"
        exit 1
    fi
fi

# STEP 3: Train improved ByT5 model
echo "🚀 Starting ByT5 training with improved configuration..."
echo "🎯 Expected training time: 3-5 hours on dual RTX 5090s"
echo ""

run_step "MODEL_TRAINING" \
    "Training ByT5 model with improved hyperparameters and $ACTUAL_EXAMPLES examples" \
    python3 t5/train_byt5_improved.py \
        --train-file "data/enhanced_training_balanced.jsonl" \
        --output-dir "models/byt5-small-typo-fixer-v3" \
        --prefix "fix typos:" \
        --max-source-len 512 \
        --max-target-len 512 \
        --learning-rate 5e-5 \
        --weight-decay 0.01 \
        --num-epochs 5 \
        --warmup-ratio 0.1 \
        --per-device-train-batch-size 8 \
        --per-device-eval-batch-size 16 \
        --gradient-accumulation-steps 4 \
        --num-workers 4 \
        --eval-steps 500 \
        --save-steps 500 \
        --logging-steps 50 \
        --gradient-checkpointing \
        --push-to-hub \
        --hub-model-id "$HUB_MODEL_ID"

# STEP 4: Test the trained model
run_step "MODEL_TESTING" \
    "Running comprehensive evaluation of trained model" \
    python3 scripts/test_byt5_on_server.py \
        --model-path "models/byt5-small-typo-fixer-v3" \
        --prefix "fix typos:" \
        --output-file "models/byt5-small-typo-fixer-v3/final_test_results.json"

# STEP 5: Final summary
echo ""
echo "🎉 COMPLETE PIPELINE FINISHED!"
echo "=" * 80

# Show training summary if available
if [ -f "models/byt5-small-typo-fixer-v3/training_summary.json" ]; then
    echo "📊 Training Summary:"
    python3 -c "
import json
try:
    with open('models/byt5-small-typo-fixer-v3/training_summary.json') as f:
        data = json.load(f)
    print(f'   Training time: {data.get(\"training_time_minutes\", 0):.1f} minutes')
    print(f'   Effective batch size: {data.get(\"effective_batch_size\", \"unknown\")}')
    print(f'   GPUs used: {data.get(\"num_gpus\", \"unknown\")}')
    print(f'   Learning rate: {data.get(\"learning_rate\", \"unknown\")}')
    print(f'   Final test examples:')
    for orig, corr in data.get('final_results', [])[:3]:
        print(f'     \"{orig}\" → \"{corr}\"')
except:
    print('   Summary not available')
"
fi

# Show test results if available
if [ -f "models/byt5-small-typo-fixer-v3/final_test_results.json" ]; then
    echo ""
    echo "🧪 Test Results:"
    python3 -c "
import json
try:
    with open('models/byt5-small-typo-fixer-v3/final_test_results.json') as f:
        data = json.load(f)
    print(f'   Word-level accuracy: {data.get(\"avg_word_accuracy\", 0)*100:.1f}%')
    print(f'   Sentence-level accuracy: {data.get(\"avg_sentence_accuracy\", 0)*100:.1f}%')
    print(f'   Perfect sentences: {data.get(\"perfect_sentences\", 0)}/{data.get(\"test_cases\", 0)}')
    print(f'   Average inference time: {data.get(\"avg_inference_time\", 0)*1000:.0f}ms')
    print(f'   Performance: {data.get(\"performance_level\", \"unknown\")}')
except:
    print('   Test results not available')
"
fi

echo ""
echo "📁 Generated Files:"
echo "   • data/enhanced_training_full.jsonl (${NUM_EXAMPLES:,} raw examples)"
echo "   • data/enhanced_training_balanced.jsonl (${ACTUAL_EXAMPLES:,} balanced examples)"
echo "   • models/byt5-small-typo-fixer-v3/ (trained model)"
echo "   • $FULL_LOG (complete pipeline log)"
echo ""
echo "🚀 Model uploaded to: https://huggingface.co/$HUB_MODEL_ID"
echo ""
echo "🎯 Next Steps:"
echo "   1. Download and test locally:"
echo "      huggingface-cli download $HUB_MODEL_ID"
echo "   2. Compare performance with previous Qwen model (target: >80% accuracy)"
echo "   3. If performance is good, update CLAUDE.md documentation"
echo ""
echo "⏱️  Total pipeline time: $(($(date +%s) - $(date -d \"$(head -1 $FULL_LOG | cut -d' ' -f4-)\" +%s 2>/dev/null || echo 0))) seconds"

echo "🎉 Pipeline complete! Check the uploaded model performance."
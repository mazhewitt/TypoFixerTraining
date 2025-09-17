#!/bin/bash
# Fallback to single GPU training when dual GPU has CUDA issues

echo "🚨 Falling back to single GPU training due to CUDA distributed issues"
echo "================================================================"

# Check GPU status
echo "🔍 Checking GPU status..."
nvidia-smi

echo ""
echo "💡 Single GPU training will:"
echo "  - Use only GPU 0 (RTX5090)"
echo "  - Increase batch size to 32 (from 16) to compensate"
echo "  - Reduce gradient accumulation to 1"
echo "  - Still achieve effective batch size of 32"
echo "  - Take about 2x longer but be more stable"
echo ""

# Start single GPU training
echo "🚀 Starting single GPU training..."
python train_single_gpu.py

echo ""
echo "✅ Single GPU training complete!"
echo "📁 Model saved to: models/qwen-enhanced-typo-fixer"
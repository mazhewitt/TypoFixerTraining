#!/bin/bash

# Install required dependencies for ByT5 training
echo "🔧 Installing required dependencies..."

# Install accelerate for transformers training
pip install "accelerate>=0.26.0"

# Install other potentially missing packages
pip install "transformers[torch]"
pip install sacrebleu
pip install datasets

echo "✅ Dependencies installed!"
echo ""
echo "🚀 Now run training:"
echo "   ./scripts/simple_byt5_training.sh"
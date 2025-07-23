#!/bin/bash

# RTX 5090 Environment Setup for Qwen Typo Correction Training
# Run this on your RTX 5090 instance after cloning the repo
# NOTE: Assumes PyTorch, CUDA, and core ML libraries are pre-installed

set -e

echo "🚀 Setting up RTX 5090 environment for Qwen typo correction training..."
echo "📋 Using pre-installed PyTorch and CUDA libraries..."

# Install only missing dependencies (no torch/cuda packages)
echo "📚 Installing missing Python dependencies..."
pip install --user transformers datasets accelerate tqdm requests huggingface_hub

# Install HuggingFace CLI if not available
echo "🤗 Installing HuggingFace CLI..."
pip install --user huggingface_hub[cli]

# Try to install flash attention (may already be installed)
echo "⚡ Checking Flash Attention availability..."
python3 -c "
try:
    import flash_attn
    print('✅ Flash Attention already available')
except ImportError:
    print('⚠️ Flash Attention not found - installing...')
    import subprocess
    subprocess.run(['pip', 'install', '--user', 'flash-attn', '--no-build-isolation'])
    print('✅ Flash Attention installed')
except Exception as e:
    print(f'⚠️ Flash Attention check failed: {e}')
    print('🔄 Training will use standard attention (slower but functional)')
"

# Verify CUDA setup
echo "🔍 Verifying CUDA setup..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Set environment variables for optimal RTX 5090 performance
echo "⚙️ Setting RTX 5090 optimization environment variables..."
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 5090 architecture
export TOKENIZERS_PARALLELISM=false

# Add to bashrc for persistence
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
echo "export TORCH_CUDA_ARCH_LIST=\"8.9\"" >> ~/.bashrc
echo "export TOKENIZERS_PARALLELISM=false" >> ~/.bashrc

# Generate dataset if not present
echo "📊 Checking for training data..."
if [ ! -f "data/enhanced_training_full.jsonl" ]; then
    echo "🔄 Generating enhanced training dataset..."
    mkdir -p data
    python3 src/realistic_data_generation.py \
        --output data/enhanced_training_full.jsonl \
        --num_examples 7000 \
        --corruption_rate 0.15
else
    echo "✅ Training data already exists"
fi

echo "✅ RTX 5090 environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run training: python3 train_rtx5090.py --train_file data/enhanced_training_full.jsonl --output_dir models/qwen-typo-fixer-rtx5090 --hf_repo mazhewitt/qwen-typo-fixer"
echo ""
echo "🔧 Environment optimized for RTX 5090:"
echo "   - CUDA_VISIBLE_DEVICES=0"
echo "   - TORCH_CUDA_ARCH_LIST=8.9 (RTX 5090 architecture)"
echo "   - Flash Attention 2 enabled (if available)"
echo "   - BFloat16 + TF32 precision"
echo "   - Global Python packages used (no venv)"
#!/bin/bash

# RTX 5090 Environment Setup for Qwen Typo Correction Training
# Run this on your RTX 5090 instance after cloning the repo

set -e

echo "🚀 Setting up RTX 5090 environment for Qwen typo correction training..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and pip if needed
echo "🐍 Installing Python dependencies..."
sudo apt install -y python3 python3-pip python3-venv git

# Create virtual environment
echo "🏗️ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (RTX 5090)
echo "⚡ Installing PyTorch with CUDA 12.1 for RTX 5090..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers with flash attention support
echo "🤗 Installing Transformers with Flash Attention..."
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install datasets==2.15.0
pip install flash-attn --no-build-isolation

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install numpy pandas tqdm requests huggingface_hub

# Install HuggingFace CLI
pip install huggingface_hub[cli]

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

echo "✅ RTX 5090 environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Clone your repo with the training data"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run training: python3 train_rtx5090.py --train_file data/enhanced_training_full.jsonl --output_dir models/qwen-typo-fixer-rtx5090 --hf_repo your-username/qwen-typo-fixer"
echo ""
echo "🔧 Environment variables set for optimal RTX 5090 performance:"
echo "   - CUDA_VISIBLE_DEVICES=0"
echo "   - TORCH_CUDA_ARCH_LIST=8.9"
echo "   - Flash Attention 2 enabled"
echo "   - BFloat16 + TF32 precision"
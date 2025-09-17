#!/bin/bash
# Safe setup script for preconfigured RTX5090 machines
# Only installs minimal dependencies without touching GPU stack

echo "🔧 Safe setup for Enhanced Qwen Typo Fixer Training"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "generate_enhanced_qwen_dataset.py" ]; then
    echo "❌ Error: Run this script from the TypoFixerTraining directory"
    exit 1
fi

# Check Python and torch availability
echo "🐍 Checking Python environment..."
python3 --version
if ! python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} available'); print(f'✅ CUDA available: {torch.cuda.is_available()}')"; then
    echo "❌ Error: PyTorch not available or CUDA not working"
    echo "💡 This script assumes PyTorch/CUDA is already installed"
    exit 1
fi

# Create virtual environment but inherit system packages
echo "📦 Creating virtual environment (inheriting system packages)..."
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install ONLY the essential packages that are likely missing
echo "📥 Installing minimal additional packages..."

# Install transformers and friends (most likely to be missing)
pip install --no-deps transformers>=4.30.0
pip install --no-deps datasets>=2.10.0
pip install --no-deps huggingface_hub>=0.34.0
pip install --no-deps tokenizers>=0.13.0

# Install pure Python utilities (safe)
pip install tqdm>=4.60.0

# Check if accelerate is available, install if not
if ! python3 -c "import accelerate" 2>/dev/null; then
    echo "📥 Installing accelerate..."
    pip install accelerate>=0.20.0
else
    echo "✅ accelerate already available"
fi

# Optional: Install wandb for experiment tracking (user choice)
read -p "🤔 Install wandb for experiment tracking? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install wandb>=0.15.0
    echo "✅ wandb installed"
else
    echo "⏭️  Skipping wandb"
fi

echo ""
echo "✅ Safe setup complete!"
echo ""
echo "🧪 Testing environment..."
python3 -c "
import torch
import transformers
import datasets
print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✅ Transformers {transformers.__version__}')
print(f'✅ Datasets {datasets.__version__}')
try:
    import accelerate
    print(f'✅ Accelerate {accelerate.__version__}')
except:
    print('⚠️  Accelerate not available (may need manual install)')
try:
    import wandb
    print(f'✅ Wandb {wandb.__version__}')
except:
    print('ℹ️  Wandb not installed (optional)')
"

echo ""
echo "🚀 Ready to train! Next steps:"
echo "  1. Generate dataset: python generate_enhanced_qwen_dataset.py --target-size 50000"
echo "  2. Train model: python train_enhanced_qwen.py"
echo ""
echo "💡 If you encounter import errors, the system packages should handle it."
echo "🔥 For dual GPU: python -m torch.distributed.launch --nproc_per_node=2 train_dual_gpu.py"
#!/bin/bash
# Quick fix for the huggingface-hub version issue

echo "🔧 Fixing huggingface-hub dependency..."

# Upgrade huggingface-hub to meet transformers requirement
pip install "huggingface_hub>=0.34.0"

echo "✅ Dependencies fixed!"

# Test the fix
echo "🧪 Testing imports..."
python3 -c "
import torch
import transformers
import datasets
import huggingface_hub
print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✅ Transformers {transformers.__version__}')
print(f'✅ Datasets {datasets.__version__}')
print(f'✅ HuggingFace Hub {huggingface_hub.__version__}')
"

echo ""
echo "🚀 Ready to train! Use:"
echo "  torchrun --nproc_per_node=2 train_dual_gpu.py"
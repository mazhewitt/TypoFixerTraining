#!/bin/bash
# Fix distributed training issue with device_map='auto'

echo "🔧 Fixing distributed training device_map issue..."

# Create a backup
cp train_enhanced_qwen.py train_enhanced_qwen.py.backup

# Apply the fix using sed
cat > fix_device_map.py << 'EOF'
import sys

# Read the file
with open('train_enhanced_qwen.py', 'r') as f:
    content = f.read()

# Replace the problematic model loading section
old_code = '''    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )'''

new_code = '''    # Check if we're in distributed mode
    is_distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        # Don't use device_map in distributed training
        device_map=None if is_distributed else ("auto" if torch.cuda.is_available() else None)
    )'''

# Replace the code
content = content.replace(old_code, new_code)

# Write back
with open('train_enhanced_qwen.py', 'w') as f:
    f.write(content)

print("✅ Fixed device_map for distributed training")
EOF

python3 fix_device_map.py
rm fix_device_map.py

echo "✅ Distributed training fix applied!"

# Verify the fix
echo "🧪 Checking the fix..."
grep -A 5 -B 2 "is_distributed" train_enhanced_qwen.py

echo ""
echo "🚀 Ready to train with dual GPU! Use:"
echo "  torchrun --nproc_per_node=2 train_dual_gpu.py"
#!/bin/bash
# Quick fix for the TrainingArguments parameter name issue

echo "🔧 Fixing TrainingArguments parameter..."

# Fix the evaluation_strategy parameter name
sed -i 's/evaluation_strategy=/eval_strategy=/g' train_enhanced_qwen.py

echo "✅ TrainingArguments fixed!"

# Verify the fix
echo "🧪 Checking the fix..."
grep -n "eval_strategy" train_enhanced_qwen.py

echo ""
echo "🚀 Ready to train! Use:"
echo "  torchrun --nproc_per_node=2 train_dual_gpu.py"
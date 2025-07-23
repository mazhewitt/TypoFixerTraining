# RTX 5070 Ti Deployment Guide (Anti-Overfitting)

## 🛡️ UPDATED: Large Dataset Training to Prevent Overfitting

**ISSUE IDENTIFIED**: Previous 7K dataset caused severe overfitting (loss → 0 in 4 minutes)
**SOLUTION**: Scale to 50K+ examples for proper generalization

## Quick Setup on Your Dual RTX 5070 Ti Instance

### 1. Clone and Setup Environment
```bash
# Clone the TypoFixerTraining repo on qwen-approach branch
  git clone -b qwen-approach https://github.com/mazhewitt/TypoFixerTraining.git
  cd TypoFixerTraining

# Run basic environment setup
./setup_rtx5090.sh

# Generate LARGE dataset separately (more reliable, prevents threading crashes)
python3 generate_large_dataset.py
```

### 2. Verify Dual RTX 5070 Ti Setup
```bash
# Check both GPUs
nvidia-smi

# Verify PyTorch multi-GPU
python3 -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)') for i in range(torch.cuda.device_count())]"
```

### 3. Start Anti-Overfitting Training 
```bash
# UPDATED: Large dataset + anti-overfitting measures
python3 train_rtx5090.py \
    --train_file data/enhanced_training_large.jsonl \
    --output_dir models/qwen-typo-fixer-v2 \
    --hf_repo mazhewitt/qwen-typo-fixer-v2 \
    --num_epochs 5 \
    --early_stopping_patience 3 \
    --max_weight_decay 0.1 \
    --eval_steps 50
```

### 4. Monitor Training Progress
🎯 **ANTI-OVERFITTING INDICATORS TO WATCH:**
```
✅ GOOD SIGNS:
- Training loss: 2.0 → 0.3 (gradual decrease)
- Validation loss: 2.1 → 0.4 (follows training)
- Gap stays small: <0.2 difference
- Early stopping activates before epoch 5

❌ OVERFITTING SIGNS:
- Training loss → 0.0 (too perfect)
- Validation loss increases while training decreases
- Large gap: >0.5 difference
- Training completes in <10 minutes
```

**Expected training time**: **30-45 minutes** (10x longer than before - this is GOOD!)

### 5. Upload to HuggingFace
```bash
# Login to HuggingFace
huggingface-cli login

# Upload improved model (automatic after training completes)
cd models/qwen-typo-fixer-v2
huggingface-cli upload . mazhewitt/qwen-typo-fixer-v2
```

## 🛡️ Anti-Overfitting Optimizations Applied

### Dataset Scaling (CRITICAL)
- ✅ **50K Examples**: 7x larger dataset prevents memorization
- ✅ **Multi-Source**: Norvig, Holbrook, Wikipedia, keyboard errors
- ✅ **Diverse Patterns**: Real-world typo distributions
- ✅ **Generation Time**: 10-15 minutes (worth the wait!)

### Training Optimizations
- ✅ **Early Stopping**: Stops when validation plateaus
- ✅ **Weight Decay**: 0.1 regularization prevents overfitting
- ✅ **Cosine LR Decay**: Gradual learning rate reduction
- ✅ **BFloat16**: Stable mixed precision
- ✅ **Multiple Checkpoints**: Keeps 3 best models

### Performance Expectations (UPDATED)
- **Dataset Size**: 50,000 examples (vs 7K before)
- **Training Speed**: ~0.8 seconds per step (more data = slower)
- **Total Time**: 30-45 minutes for 5 epochs
- **Memory Usage**: ~14GB VRAM per GPU (well within 16GB limit)
- **Target**: 90%+ accuracy WITH generalization

## Training Data Summary (SCALED UP)

Your LARGE enhanced dataset includes:
- **50,000 examples** from multiple high-quality sources (7x increase!)
- **Norvig's 20k misspellings** from Google spellcheck logs
- **Academic datasets** (Holbrook/Birkbeck)
- **Wikipedia typo corrections** from revision history
- **Enhanced keyboard errors** with realistic QWERTY patterns
- **Natural sentences** from WikiText (not artificial concatenations)

## Model Performance Target (REALISTIC)

- **Previous Model**: 100% accuracy on training data (OVERFITTED - bad!)
- **New Target**: 90% sentence accuracy on UNSEEN data (GENERALIZED - good!)
- **Expected with 50K dataset**: 88-93% with proper generalization
- **Key Success Metric**: Performance on validation set, not training set

## After Training

1. **Model Size**: ~1.2GB (Qwen 0.6B fine-tuned)
2. **HuggingFace Upload**: Automatic with model card
3. **ANE Conversion**: Use anemll pipeline for Apple deployment
4. **Validation**: Built-in accuracy evaluation

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 16 --gradient_accumulation_steps 3
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi -l 1
```

### Flash Attention Issues
```bash
# Reinstall if needed
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

## Next Steps After Training

1. **Test the model** with validation examples
2. **Upload to HuggingFace** for community use
3. **Convert to ANE** for Apple deployment
4. **Deploy in production** applications

Your RTX 5090 setup is optimized for maximum performance and should achieve the 90% accuracy target efficiently!
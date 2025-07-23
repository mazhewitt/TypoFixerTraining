# RTX 5090 Deployment Guide

## Quick Setup on Your RTX 5090 Instance

### 1. Clone and Setup Environment
```bash
# Clone the TypoFixerTraining repo on qwen-approach branch
git clone -b qwen-approach https://github.com/mazhewitt/TypoFixerTraining.git
cd TypoFixerTraining

# Run RTX 5090 setup (uses global Python packages)
./setup_rtx5090.sh
```

### 2. Verify RTX 5090 Setup
```bash
# Check GPU
nvidia-smi

# Verify PyTorch CUDA
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```

### 3. Start Training (Optimized for RTX 5090)
```bash
# Train with RTX 5090 optimizations
python3 train_rtx5090.py \
    --train_file data/enhanced_training_full.jsonl \
    --output_dir models/qwen-typo-fixer-mem-opt \
    --hf_repo mazhewitt/qwen-typo-fixer \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_seq_len 128 \
    --num_epochs 3
```

### 4. Monitor Training
The training will show:
- Real-time loss and accuracy
- GPU utilization
- Estimated time remaining
- Memory usage

Expected training time: **~15-20 minutes** on RTX 5090

### 5. Upload to HuggingFace
```bash
# Login to HuggingFace
huggingface-cli login

# Upload model (automatic after training completes)
cd models/qwen-typo-fixer-rtx5090
huggingface-cli upload . mazhewitt/qwen-typo-fixer
```

## RTX 5090 Optimizations Applied

### Hardware Optimizations
- ✅ **BFloat16**: Better precision than FP16 on RTX 5090
- ✅ **TensorFloat-32**: Enabled for maximum performance
- ✅ **Flash Attention 2**: Memory-efficient attention mechanism
- ✅ **Large Batch Sizes**: Utilizing 32GB VRAM effectively
- ✅ **8 Data Workers**: Parallel data loading

### Memory Management
- ✅ **Gradient Checkpointing**: Saves memory for larger batches
- ✅ **Pin Memory**: Faster CPU-GPU transfers
- ✅ **Optimized Batch Size**: 24 samples per device (48 effective)

### Performance Expectations
- **Training Speed**: ~0.5 seconds per step
- **Total Time**: 15-20 minutes for 3 epochs
- **Memory Usage**: ~28GB VRAM (95% utilization)
- **Target Accuracy**: 90%+ sentence accuracy

## Training Data Summary

Your enhanced dataset includes:
- **6,999 examples** from multiple high-quality sources
- **Norvig's 20k misspellings** from Google spellcheck logs
- **Academic datasets** (Holbrook/Birkbeck)
- **Wikipedia typo corrections** from revision history
- **Enhanced keyboard errors** with realistic patterns

## Model Performance Target

- **Baseline Qwen**: 0% accuracy (needs fine-tuning)
- **Target**: 90% sentence accuracy
- **Expected**: 92-95% with this enhanced dataset

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
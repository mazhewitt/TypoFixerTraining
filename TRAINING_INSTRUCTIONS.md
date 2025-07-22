# Production Training Instructions

Complete step-by-step guide for training your DistilBERT typo correction model from proof-of-concept to production deployment.

## Overview

You'll go from our current 614-example proof-of-concept to a production model trained on 100K+ examples, then deploy it with Apple Neural Engine acceleration.

**Current Status**: ✅ Proof-of-concept with 1.3x ANE speedup  
**Goal**: 🎯 Production model with 92%+ accuracy on 100K examples

---

## Option A: GCP Vertex AI (Recommended - Jupyter Interface)

### Step 1: Setup Vertex AI Notebook

1. **Go to Google Cloud Console**
   - Navigate to **Vertex AI** → **Workbench**
   - Click **"New Notebook"**

2. **Configure Instance**
   - **Environment**: PyTorch 1.13 (with CUDA)
   - **Machine type**: n1-standard-4 (4 vCPUs, 15GB RAM)
   - **GPU**: NVIDIA Tesla V100 (or A100 if available)
   - **Boot disk**: 100GB

3. **Launch Notebook**
   - Click **"Create"** (takes 3-5 minutes)
   - Click **"Open JupyterLab"** when ready
   - **Verify GPU is enabled** in your runtime settings

### Step 2: Setup Training Environment

**Create new notebook and run these cells:**

```python
# Cell 1: Clone repository
!git clone https://github.com/YOUR_USERNAME/train-typo-fixer.git
%cd train-typo-fixer
```

```python
# Cell 2: Install dependencies (optimized for Colab)
!pip install -r requirements-colab.txt
# This skips PyTorch reinstall to avoid NVIDIA library updates
```

```python
# Cell 3: Verify GPU is enabled
# IF YOU SEE "CUDA available: False" - YOU NEED TO ENABLE GPU!
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    !nvidia-smi
else:
    print("❌ NO GPU DETECTED!")
    print("🔧 FIX: Runtime → Change runtime type → Hardware accelerator → GPU")
    print("Then Runtime → Restart runtime and re-run this cell")
```

### Step 3: Generate Training Data

```python
# Cell 4: Generate 100K training examples (30-45 minutes)
!python src/data_generation.py \
  --num_examples 100000 \
  --output data/processed/train_100k.jsonl \
  --dataset wikitext \
  --max_length 128
```

```python
# Cell 5: Generate validation data (5 minutes)  
!python src/data_generation.py \
  --num_examples 10000 \
  --output data/processed/validation_10k.jsonl \
  --dataset wikitext \
  --max_length 128
```

### Step 4: Train Production Model

```python
# Cell 6: Start training (1.5-2 hours)
!python src/train.py \
  --train_file data/processed/train_100k.jsonl \
  --validation_file data/processed/validation_10k.jsonl \
  --output_dir models/production_typo_fixer \
  --num_train_epochs 3 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 2e-5 \
  --max_seq_len 128 \
  --save_steps 5000 \
  --eval_steps 2000 \
  --logging_steps 500 \
  --load_best_model_at_end \
  --run_name "production-100k-3ep"
```

### Step 5: Validate Model

```python
# Cell 7: Test model performance
!python src/validate.py \
  --model_dir models/production_typo_fixer \
  --target_accuracy 0.92 \
  --benchmark_speed \
  --error_analysis
```

### Step 6: Upload to Hugging Face Hub

```python
# Cell 8: Login to HF Hub
from huggingface_hub import login
login()  # Enter your HF token when prompted
```

```python
# Cell 9: Upload model
!python src/upload_to_hub.py \
  --model_path models/production_typo_fixer \
  --hub_model_name YOUR_USERNAME/distilbert-typo-fixer-v1 \
  --commit_message "Production model: 100K examples, 3 epochs, 92%+ accuracy"
```

### Step 7: Download for Local ANE Conversion

**On your local Mac:**

```bash
# Download the trained model
python src/download_from_hub.py \
  --hub_model_name YOUR_USERNAME/distilbert-typo-fixer-v1 \
  --local_path models/production_model

# Convert to Apple Neural Engine format
python src/apple_ane_conversion.py \
  --input_model models/production_model \
  --ane_model_path models/production_model_ane \
  --coreml_output models/production_model_ANE.mlpackage

# Benchmark ANE vs CPU performance
python src/ane_vs_cpu_benchmark.py
```

---

## Option B: Vast.ai (Most Cost-Effective)

### Step 1: Setup Vast.ai Instance

1. **Go to https://vast.ai**
2. **Filter instances:**
   - GPU: RTX 4090
   - RAM: ≥32GB  
   - CUDA: ≥11.8
   - Template: PyTorch
3. **Rent instance** (~$0.20/hour)
4. **SSH into instance** using provided command

### Step 2: Setup Environment

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Clone repository
git clone https://github.com/YOUR_USERNAME/train-typo-fixer.git
cd train-typo-fixer

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-cloud.txt

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 3: Generate Data & Train

```bash
# Generate training data (30-45 minutes)
python src/data_generation.py \
  --num_examples 100000 \
  --output data/processed/train_100k.jsonl \
  --dataset wikitext

# Generate validation data (5 minutes)
python src/data_generation.py \
  --num_examples 10000 \
  --output data/processed/validation_10k.jsonl \
  --dataset wikitext

# Train model (2-3 hours) - use screen to prevent disconnection
screen -S training
python src/train.py \
  --train_file data/processed/train_100k.jsonl \
  --validation_file data/processed/validation_10k.jsonl \
  --output_dir models/production_typo_fixer \
  --num_train_epochs 3 \
  --per_device_train_batch_size 64 \
  --run_name "production-100k-3ep"

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r training
```

### Step 4: Validate & Upload

```bash
# Validate model
python src/validate.py \
  --model_dir models/production_typo_fixer \
  --target_accuracy 0.92 \
  --benchmark_speed

# Upload to HF Hub
export HF_TOKEN="your_token_here"  # Get from huggingface.co/settings/tokens
python src/upload_to_hub.py \
  --model_path models/production_typo_fixer \
  --hub_model_name YOUR_USERNAME/distilbert-typo-fixer-v1
```

---

## Expected Results

### Training Progress
- **Data generation**: 30-60 minutes for 100K examples
- **Training time**: 1.5-3 hours depending on GPU
- **Final accuracy**: 92%+ token accuracy expected
- **Model size**: ~66M parameters → 1.5M trainable

### Performance Metrics
- **Token accuracy**: >92% (vs 85% from proof-of-concept)  
- **Perfect match rate**: >60%
- **Correction attempt rate**: >80%
- **ANE speedup**: 1.3x+ (maintained from proof-of-concept)

### Cost Estimate
| Provider | GPU | Time | Total Cost |
|----------|-----|------|------------|
| GCP Vertex AI | V100 | 2 hours | ~$20 |
| Vast.ai | RTX 4090 | 3 hours | ~$10 |
| RunPod | RTX 4090 | 3 hours | ~$15 |

---

## Troubleshooting

### Common Issues

**🚨 No GPU Detected (CUDA available: False)**
```python
# MOST COMMON ISSUE: GPU not enabled in runtime
# 1. Click "Runtime" → "Change runtime type"
# 2. Hardware accelerator: Select "GPU" (T4, V100, or A100)
# 3. Click "Save"
# 4. Click "Runtime" → "Restart runtime" 
# 5. Re-run your cells - you should see "CUDA available: True"
```

**🚨 Google Colab Runtime Issues**
```python
# If you see dependency warnings or runtime won't connect:
# 1. Click "Runtime" → "Disconnect and delete runtime"
# 2. Wait 30 seconds, then click "Connect" to get a fresh runtime
# 3. Make sure you have GPU enabled: Runtime → Change runtime type → GPU
# 4. Re-run all cells from the beginning
# 5. This is normal when upgrading packages in Colab
```

**🚨 CUDA Out of Memory**
```bash
# Reduce batch size
--per_device_train_batch_size 32
# Or enable gradient accumulation  
--gradient_accumulation_steps 2
```

**🚨 Instance Disconnection**
```bash
# Use screen or tmux for long training
screen -S training
python src/train.py [arguments]
# Detach: Ctrl+A, D
# Reattach: screen -r training
```

**🚨 Data Generation Too Slow**
```bash
# Start with smaller test first
python src/data_generation.py --num_examples 1000 --output data/test.jsonl
# Then scale up to 100K
```

### Monitoring Commands

```bash
# GPU usage
watch -n 1 nvidia-smi

# Disk space
df -h

# Training logs
tail -f models/production_typo_fixer/training_info.json

# Check model files
ls -la models/production_typo_fixer/
```

---

## Success Checklist

- [ ] ✅ Cloud instance launched with GPU
- [ ] ✅ Repository cloned and dependencies installed
- [ ] ✅ 100K training examples generated
- [ ] ✅ 10K validation examples generated
- [ ] ✅ Model trained for 3 epochs
- [ ] ✅ Validation accuracy >92%
- [ ] ✅ Model uploaded to Hugging Face Hub
- [ ] ✅ Model downloaded locally
- [ ] ✅ ANE conversion completed
- [ ] ✅ ANE benchmark shows speedup
- [ ] ✅ Cloud instance terminated

---

## Next Steps After Training

1. **Local Deployment**: Use `bc1s_inference.py` for fast local corrections
2. **Integration**: Add model to your applications via HF Hub
3. **Monitoring**: Track model performance in production
4. **Updates**: Retrain with more data or different domains as needed

**🎉 Congratulations!** You now have a production-ready typo correction model optimized for Apple Neural Engine with proven 1.3x+ speedup!
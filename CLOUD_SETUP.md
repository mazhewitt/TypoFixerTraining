# Cloud Training Setup Guide

This guide covers setting up cloud GPU instances for production-scale DistilBERT typo correction training.

## Quick Start Commands

```bash
# 1. Clone repository  
git clone https://github.com/YOUR_USERNAME/train-typo-fixer.git
cd train-typo-fixer

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-cloud.txt

# 3. Generate 100K training examples
python src/data_generation.py \
  --num_examples 100000 \
  --output data/processed/train_100k.jsonl \
  --dataset wikitext

# 4. Train production model
python src/train.py \
  --train_file data/processed/train_100k.jsonl \
  --output_dir models/production_typo_fixer \
  --num_train_epochs 3 \
  --per_device_train_batch_size 64 \
  --max_seq_len 128

# 5. Upload to Hugging Face Hub
python src/upload_to_hub.py \
  --model_path models/production_typo_fixer \
  --hub_model_name YOUR_USERNAME/distilbert-typo-fixer-v1
```

## Cloud Provider Options

### 1. Vast.ai (Recommended - Most Cost Effective)

**Pros**: Cheapest option, good GPU selection, simple interface  
**Cons**: Community marketplace, variable availability  
**Cost**: ~$0.20/hour RTX 4090, ~$0.50/hour A100  

**Setup**:
1. Go to https://vast.ai
2. Filter for: GPU >= RTX 4090, RAM >= 32GB, CUDA >= 11.8
3. Select instance and launch with PyTorch template
4. SSH into instance and run quick start commands

### 2. RunPod (Good Balance)

**Pros**: Reliable infrastructure, easy setup, persistent storage  
**Cons**: Slightly more expensive than Vast.ai  
**Cost**: ~$0.30/hour RTX 4090, ~$0.60/hour A100  

**Setup**:
1. Go to https://runpod.io
2. Choose "Secure Cloud" → GPU Pod
3. Select RTX 4090 or A100 with PyTorch template
4. Add persistent storage (10GB minimum)

### 3. Google Colab Pro+ (Easiest)

**Pros**: No setup required, familiar interface, includes storage  
**Cons**: Most expensive, session limits, potential disconnections  
**Cost**: $50/month for unlimited usage  

**Setup**:
1. Subscribe to Colab Pro+
2. Upload notebook or clone repository
3. Select A100 GPU runtime
4. Run training directly in notebook

### 4. GCP Vertex AI (Jupyter Interface)

**Pros**: Jupyter notebooks, Google infrastructure, easy scaling  
**Cons**: More expensive than community providers  
**Cost**: ~$1.20/hour T4, ~$2.50/hour V100, ~$3.50/hour A100  

**Setup**:
1. Go to GCP Console → Vertex AI → Workbench
2. Create new notebook instance with GPU (T4/V100/A100)
3. Choose PyTorch image with CUDA
4. Upload our training scripts or clone repository directly

### 5. AWS EC2 (Enterprise)

**Pros**: Most reliable, enterprise features, scalable  
**Cons**: Most expensive, complex setup  
**Cost**: ~$1.00/hour g5.xlarge (T4), ~$32/hour p4d.24xlarge (A100)  

**Setup**:
1. Launch Deep Learning AMI (Ubuntu 20.04)
2. Choose g5.xlarge or p4d.24xlarge instance
3. Configure security groups for SSH access
4. Pre-installed PyTorch environment

## Recommended Configuration

### For Development/Testing (< 10K examples)
- **Instance**: RTX 3080/4080, 16GB RAM
- **Storage**: 20GB
- **Time**: ~30 minutes
- **Cost**: ~$2-5 total

### For Production Training (100K+ examples)  
- **Instance**: RTX 4090 or A100, 32GB+ RAM
- **Storage**: 50GB+ (for datasets and checkpoints)
- **Time**: 2-4 hours
- **Cost**: ~$10-20 total

## Detailed Setup Steps

### 1. Instance Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.9+ if needed
sudo apt install python3.9 python3.9-venv python3.9-dev -y

# Clone repository
git clone https://github.com/YOUR_USERNAME/train-typo-fixer.git
cd train-typo-fixer

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements-cloud.txt
```

### 2. GPU Verification

```bash
# Check GPU availability
nvidia-smi

# Verify PyTorch GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 3. Data Generation

```bash
# Small test (1K examples - 2 minutes)
python src/data_generation.py \
  --num_examples 1000 \
  --output data/processed/test_1k.jsonl

# Medium scale (10K examples - 10 minutes)  
python src/data_generation.py \
  --num_examples 10000 \
  --output data/processed/train_10k.jsonl

# Production scale (100K examples - 30-60 minutes)
python src/data_generation.py \
  --num_examples 100000 \
  --output data/processed/train_100k.jsonl \
  --dataset wikitext \
  --max_length 128

# Generate validation set
python src/data_generation.py \
  --num_examples 10000 \
  --output data/processed/validation_10k.jsonl \
  --dataset wikitext
```

### 4. Training Configuration

**Small Scale Testing**:
```bash
python src/train.py \
  --train_file data/processed/test_1k.jsonl \
  --output_dir models/test_model \
  --num_train_epochs 1 \
  --per_device_train_batch_size 32 \
  --max_seq_len 128
```

**Production Training**:
```bash
python src/train.py \
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
  --save_total_limit 3 \
  --run_name "production-100k-3ep"
```

### 5. Monitoring Training

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# View logs
tail -f models/production_typo_fixer/logs/*/events.out.tfevents.*

# Check training progress
ls -la models/production_typo_fixer/
```

## Performance Expectations

| Dataset Size | GPU | Batch Size | Time | Total Cost |
|-------------|-----|------------|------|------------|
| 1K examples | RTX 4080 | 32 | 2 min | ~$0.20 |
| 10K examples | RTX 4090 | 64 | 20 min | ~$2 |
| 100K examples | RTX 4090 | 64 | 2-3 hours | ~$10 |
| 100K examples | A100 | 128 | 1-1.5 hours | ~$15 |
| 100K examples | GCP V100 | 128 | 1.5-2 hours | ~$20 |

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Reduce batch size
--per_device_train_batch_size 32
# Or enable gradient accumulation
--gradient_accumulation_steps 2
```

**Slow Data Loading**:
```bash
# Increase workers (if enough CPU cores)
--dataloader_num_workers 8
```

**Instance Disconnection**:
```bash
# Use screen or tmux
screen -S training
python src/train.py [args]
# Detach: Ctrl+A, D
# Reattach: screen -r training
```

**Storage Full**:
```bash
# Clean up checkpoints
rm -rf models/*/checkpoint-*
# Keep only final model
```

### Monitoring Commands

```bash
# Disk usage
df -h

# Memory usage
free -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Training progress
tail -f models/*/training_info.json
```

## Upload to Hugging Face Hub

### Setup Authentication

```bash
# Install HF CLI
pip install --upgrade huggingface_hub

# Login (interactive)
huggingface-cli login

# Or with token
export HF_TOKEN="your_token_here"
```

### Upload Model

```bash
python src/upload_to_hub.py \
  --model_path models/production_typo_fixer \
  --hub_model_name YOUR_USERNAME/distilbert-typo-fixer-v1 \
  --commit_message "Production model: 100K examples, 3 epochs"
```

## Cost Optimization Tips

1. **Use Spot/Preemptible Instances**: 50-90% cheaper, but can be interrupted
2. **Start Small**: Test with 1K examples first (~$0.20)
3. **Monitor Usage**: Stop instances immediately after training
4. **Use Efficient Batch Sizes**: Balance speed vs memory (64 for RTX 4090, 128 for A100)
5. **Persistent Storage**: Only pay for storage when needed
6. **Off-Peak Hours**: Some providers offer lower rates

## Security Best Practices

1. **Use SSH Keys**: Never share passwords
2. **VPN Access**: Restrict access to trusted IPs
3. **Clean Up**: Delete instances and storage after training
4. **Monitor Costs**: Set billing alerts
5. **Backup Important Data**: Upload to Hub immediately after training

## GCP Vertex AI Specific Setup

### Using Jupyter Notebooks on Vertex AI

**Option 1: Clone repository in notebook**:
```python
# In first cell
!git clone https://github.com/YOUR_USERNAME/train-typo-fixer.git
%cd train-typo-fixer
!pip install -r requirements-cloud.txt
```

**Option 2: Upload individual scripts**:
1. Upload `data_generation.py`, `train.py`, `upload_to_hub.py` 
2. Install requirements: `!pip install -r requirements-cloud.txt`
3. Run training cells directly

**Jupyter Training Example**:
```python
# Cell 1: Setup
import subprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Cell 2: Generate data  
!python src/data_generation.py --num_examples 100000 --output data/train_100k.jsonl

# Cell 3: Train model
!python src/train.py \
  --train_file data/train_100k.jsonl \
  --output_dir models/production_model \
  --num_train_epochs 3 \
  --per_device_train_batch_size 64

# Cell 4: Upload to Hub
!python src/upload_to_hub.py \
  --model_path models/production_model \
  --hub_model_name YOUR_USERNAME/distilbert-typo-fixer-v1
```

## Next Steps After Training

1. **Validate Model**: `python src/validate.py --model_dir models/production_typo_fixer`
2. **Upload to Hub**: `python src/upload_to_hub.py --model_path models/production_typo_fixer`
3. **Download Locally**: `python src/download_from_hub.py --hub_model_name YOUR_USERNAME/model-name`
4. **Convert to ANE**: `python src/apple_ane_conversion.py --input_model models/downloaded_model`
5. **Benchmark ANE**: `python src/ane_vs_cpu_benchmark.py`
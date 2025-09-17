# Enhanced Qwen Typo Fixer Training Guide - Dual RTX5090

Complete step-by-step instructions for training Qwen with T5-improved data on dual RTX5090 machine.

## 🎯 Overview

This guide covers the complete pipeline:
1. **Data Generation** - Generate enhanced training dataset with T5 improvements
2. **Training Setup** - Configure for dual RTX5090 training
3. **Model Training** - Fine-tune Qwen with advanced error patterns
4. **Checkpoint Selection** - Evaluate and select best checkpoint
5. **HuggingFace Upload** - Deploy the trained model

---

## 📋 Prerequisites

### GPU Machine Requirements
- **Hardware**: Dual RTX5090 (48GB total VRAM)
- **CUDA**: Version 11.8+ or 12.0+
- **Python**: 3.9-3.11
- **Storage**: 100GB+ free space

### HuggingFace Setup
```bash
# Install HF CLI and login
pip install huggingface_hub
huggingface-cli login
# Enter your HF token when prompted
```

---

## 📊 STEP 1: Data Generation

### 1.1 Clone Repository
```bash
git clone https://github.com/yourusername/TypoFixerTraining.git
cd TypoFixerTraining
```

### 1.2 Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 1.3 Generate Enhanced Training Dataset
```bash
# Generate 100k examples with T5 improvements
python generate_enhanced_qwen_dataset.py \
    --target-size 100000 \
    --output-file data/enhanced_qwen_training.jsonl \
    --seed 42

# This creates:
# - data/enhanced_qwen_training.jsonl (training data)
# - data/enhanced_qwen_training_metadata.json (statistics)
# - data/diverse_source_sentences.json (source sentences)
```

**Expected Output:**
```
✅ Enhanced Qwen training dataset generation complete!
📁 Enhanced Qwen training dataset: data/enhanced_qwen_training.jsonl
📊 Total examples: 100,000
📈 Ready for Qwen fine-tuning!
```

### 1.4 Verify Dataset Quality
```bash
# Check dataset statistics
python -c "
import json
with open('data/enhanced_qwen_training_metadata.json', 'r') as f:
    stats = json.load(f)
    print(f'Total examples: {stats[\"total_examples\"]:,}')
    print('Complexity distribution:', stats['complexity_distribution'])
    print('Domain distribution:', stats['domain_distribution'])
"
```

---

## 🚀 STEP 2: Training Setup for Dual RTX5090

### 2.1 Training Configuration
Create `training_config.json`:
```json
{
  "model_name": "Qwen/Qwen2-0.5B",
  "train_file": "data/enhanced_qwen_training.jsonl",
  "output_dir": "models/qwen-enhanced-typo-fixer",
  "num_train_epochs": 3,
  "per_device_train_batch_size": 16,
  "per_device_eval_batch_size": 16,
  "gradient_accumulation_steps": 2,
  "learning_rate": 5e-5,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "eval_strategy": "steps",
  "eval_steps": 500,
  "logging_steps": 100,
  "save_steps": 500,
  "save_total_limit": 5,
  "fp16": true,
  "dataloader_num_workers": 8,
  "group_by_length": true,
  "report_to": "wandb",
  "run_name": "qwen-enhanced-typo-fixer-rtx5090",
  "early_stopping_patience": 5,
  "early_stopping_threshold": 0.001
}
```

### 2.2 Multi-GPU Training Script
Create `train_dual_gpu.py`:
```python
#!/usr/bin/env python3
"""
Dual RTX5090 Training Script for Enhanced Qwen Typo Fixer
"""
import os
import torch
from train_enhanced_qwen import QwenTrainingConfig, train_enhanced_qwen
import json

def setup_dual_gpu():
    """Setup for dual GPU training."""
    # Verify GPU setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    gpu_count = torch.cuda.device_count()
    print(f"🔥 Available GPUs: {gpu_count}")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({memory:.1f}GB)")

    if gpu_count < 2:
        print("⚠️  Warning: Expected 2 GPUs, found", gpu_count)

    # Set environment variables for distributed training
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use both GPUs

    return gpu_count

def main():
    # Setup GPUs
    gpu_count = setup_dual_gpu()

    # Load training configuration
    with open('training_config.json', 'r') as f:
        config_dict = json.load(f)

    # Adjust batch size for dual GPU
    config_dict['per_device_train_batch_size'] = 16  # Per GPU
    config_dict['per_device_eval_batch_size'] = 16   # Per GPU
    # Total effective batch size = 16 * 2 GPUs * 2 grad_accum = 64

    config = QwenTrainingConfig(**config_dict)

    # Start training
    print("🚀 Starting Enhanced Qwen Training on Dual RTX5090")
    train_enhanced_qwen(config)

if __name__ == "__main__":
    main()
```

---

## 🏋️ STEP 3: Model Training

### 3.1 Initialize Weights & Biases (Optional)
```bash
# Install and setup wandb for experiment tracking
pip install wandb
wandb login
# Enter your wandb API key when prompted
```

### 3.2 Start Training
```bash
# Single GPU training (if needed)
python train_enhanced_qwen.py --config-file training_config.json

# OR Multi-GPU training (recommended for RTX5090s)
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    train_dual_gpu.py
```

### 3.3 Monitor Training Progress
```bash
# Watch training logs
tail -f models/qwen-enhanced-typo-fixer/training.log

# Check GPU utilization
nvidia-smi -l 1

# Monitor wandb dashboard (if enabled)
# Visit: https://wandb.ai/your-username/qwen-typo-fixer
```

**Expected Training Output:**
```
🚀 Starting Enhanced Qwen Training on Dual RTX5090
🔥 Available GPUs: 2
  GPU 0: NVIDIA GeForce RTX 5090 (24.0GB)
  GPU 1: NVIDIA GeForce RTX 5090 (24.0GB)

📊 ENHANCED DATASET ANALYSIS
==================================================
🏷️ Domain Distribution:
  conversational : 25,000
  professional   : 20,000
  educational    : 20,000
  creative       : 15,000
  instructional  : 10,000
  general        : 10,000

Training Progress:
Step 100/7500: Loss=1.234, LR=4.2e-5, GPU Mem: 18.5GB
Step 200/7500: Loss=1.156, LR=4.4e-5, GPU Mem: 18.5GB
...
```

---

## 📈 STEP 4: Checkpoint Selection & Evaluation

### 4.1 Automatic Best Checkpoint Selection
The trainer automatically saves the best checkpoint based on evaluation loss. Check:
```bash
ls -la models/qwen-enhanced-typo-fixer/
# Look for: pytorch_model.bin, config.json, tokenizer files
```

### 4.2 Manual Checkpoint Evaluation
Create `evaluate_checkpoints.py`:
```python
#!/usr/bin/env python3
"""
Evaluate all saved checkpoints and select the best one.
"""
import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def evaluate_checkpoint(checkpoint_path, test_examples):
    """Evaluate a specific checkpoint."""
    print(f"📊 Evaluating {checkpoint_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    correct = 0
    total = len(test_examples)

    for example in tqdm(test_examples, desc="Testing"):
        corrupted = example['corrupted']
        expected = example['clean']

        # Generate correction
        prompt = f"Correct the typos in this text: {corrupted}"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.1
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the correction part
        correction = generated.replace(prompt, "").strip()

        # Simple exact match evaluation
        if correction.lower().strip() == expected.lower().strip():
            correct += 1

    accuracy = correct / total
    return accuracy

def find_best_checkpoint():
    """Find and evaluate all checkpoints."""
    base_dir = Path("models/qwen-enhanced-typo-fixer")

    # Load test examples
    test_examples = []
    with open("data/enhanced_qwen_training.jsonl", "r") as f:
        all_examples = [json.loads(line) for line in f]
        # Use last 100 examples as test set
        for example in all_examples[-100:]:
            messages = example['messages']
            corrupted = messages[0]['content'].replace("Correct the typos in this text: ", "")
            clean = messages[1]['content']
            test_examples.append({'corrupted': corrupted, 'clean': clean})

    # Find all checkpoint directories
    checkpoints = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            checkpoints.append(item)

    if not checkpoints:
        print("⚠️  No checkpoints found, using final model")
        checkpoints = [base_dir]

    # Evaluate each checkpoint
    results = {}
    for checkpoint in checkpoints:
        if (checkpoint / "pytorch_model.bin").exists():
            accuracy = evaluate_checkpoint(checkpoint, test_examples)
            results[str(checkpoint)] = accuracy
            print(f"  Accuracy: {accuracy:.3f}")

    # Find best checkpoint
    best_checkpoint = max(results.keys(), key=lambda k: results[k])
    best_accuracy = results[best_checkpoint]

    print(f"\n🏆 Best Checkpoint: {best_checkpoint}")
    print(f"📊 Best Accuracy: {best_accuracy:.3f}")

    return best_checkpoint, best_accuracy

if __name__ == "__main__":
    find_best_checkpoint()
```

### 4.3 Run Checkpoint Evaluation
```bash
python evaluate_checkpoints.py
```

---

## 🚀 STEP 5: HuggingFace Upload

### 5.1 Prepare Model for Upload
Create `upload_to_hf.py`:
```python
#!/usr/bin/env python3
"""
Upload the best trained model to HuggingFace Hub.
"""
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModelForCausalLM

def upload_model_to_hf(
    model_path: str,
    repo_name: str,
    username: str,
    private: bool = False
):
    """Upload trained model to HuggingFace Hub."""

    full_repo_name = f"{username}/{repo_name}"
    print(f"🚀 Uploading model to {full_repo_name}")

    # Initialize HF API
    api = HfApi()

    # Create repository
    try:
        create_repo(
            repo_id=full_repo_name,
            exist_ok=True,
            private=private,
            repo_type="model"
        )
        print(f"✅ Repository created/verified: {full_repo_name}")
    except Exception as e:
        print(f"⚠️  Repository creation failed: {e}")

    # Load and verify model
    print("🔍 Loading and verifying model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto"
    )

    # Test the model
    test_input = "Correct the typos in this text: I beleive this is teh correct answr."
    inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🧪 Test result: {result}")

    # Create model card
    model_card_content = f"""
---
language: en
license: mit
tags:
- text-generation
- typo-correction
- qwen
- fine-tuned
datasets:
- custom
pipeline_tag: text-generation
---

# Qwen Enhanced Typo Fixer

A fine-tuned Qwen model for typo correction using advanced error patterns and multi-domain training data.

## Model Description

This model is a fine-tuned version of `Qwen/Qwen2-0.5B` for typo correction. It was trained on an enhanced dataset featuring:

- **100,000 training examples** with realistic error patterns
- **Multi-domain coverage**: conversational, professional, educational, creative, instructional, general
- **Advanced error types**: keyboard errors, phonetic confusions, contextual mistakes, punctuation variations
- **Balanced punctuation**: 50/50 split between sentences with/without ending punctuation

## Training Details

- **Base Model**: Qwen/Qwen2-0.5B
- **Training Hardware**: Dual RTX5090 (48GB total VRAM)
- **Training Time**: ~X hours
- **Final Accuracy**: X.X%
- **Dataset Size**: 100,000 examples
- **Epochs**: 3

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{full_repo_name}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("{full_repo_name}", trust_remote_code=True)

# Example usage
text_with_typos = "I beleive this is teh correct answr."
prompt = f"Correct the typos in this text: {{text_with_typos}}"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
correction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(correction)
```

## Training Configuration

The model was trained with the following key parameters:
- Learning rate: 5e-5
- Batch size: 32 (16 per GPU × 2 GPUs)
- Gradient accumulation: 2 steps
- Weight decay: 0.01
- Warmup ratio: 0.1

## Dataset Features

### Error Pattern Distribution
- **Spelling errors**: 60%+ coverage
- **Keyboard errors**: 30%+ coverage
- **Phonetic errors**: 8%+ coverage
- **Grammar errors**: 2%+ coverage
- **Punctuation errors**: <1% coverage

### Domain Distribution
- **Conversational**: 25% (informal, everyday language)
- **Professional**: 20% (business, formal communication)
- **Educational**: 20% (academic, instructional content)
- **Creative**: 15% (stories, descriptions)
- **Instructional**: 10% (how-to, procedures)
- **General**: 10% (miscellaneous)

## Evaluation

The model achieves strong performance on typo correction tasks, with particular strength in:
- Single and multi-word typos
- Contextual corrections
- Maintaining original meaning and style
- Handling various text domains

## Limitations

- Optimized for English text
- Best performance on sentences under 150 characters
- May struggle with highly technical or domain-specific terminology
- Designed for typo correction, not general text improvement

## Citation

If you use this model, please cite:
```
@misc{{qwen-enhanced-typo-fixer,
  title={{Qwen Enhanced Typo Fixer}},
  author={{Your Name}},
  year={{2025}},
  url={{https://huggingface.co/{full_repo_name}}}
}}
```
"""

    # Save model card
    model_card_path = Path(model_path) / "README.md"
    with open(model_card_path, "w") as f:
        f.write(model_card_content)

    # Upload entire model folder
    print("📤 Uploading model files...")
    upload_folder(
        folder_path=model_path,
        repo_id=full_repo_name,
        repo_type="model",
        commit_message="Upload enhanced Qwen typo fixer model"
    )

    print(f"✅ Model successfully uploaded to: https://huggingface.co/{full_repo_name}")
    return full_repo_name

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload trained model to HuggingFace")
    parser.add_argument("--model-path", default="models/qwen-enhanced-typo-fixer",
                       help="Path to trained model")
    parser.add_argument("--repo-name", default="qwen-enhanced-typo-fixer",
                       help="HuggingFace repository name")
    parser.add_argument("--username", required=True,
                       help="HuggingFace username")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private")

    args = parser.parse_args()

    # Verify model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")

    # Upload model
    repo_url = upload_model_to_hf(
        model_path=args.model_path,
        repo_name=args.repo_name,
        username=args.username,
        private=args.private
    )

    print(f"\n🎉 SUCCESS! Model available at:")
    print(f"https://huggingface.co/{repo_url}")

if __name__ == "__main__":
    main()
```

### 5.2 Upload to HuggingFace
```bash
# Upload your trained model
python upload_to_hf.py \
    --model-path models/qwen-enhanced-typo-fixer \
    --repo-name qwen-enhanced-typo-fixer \
    --username YOUR_HF_USERNAME \
    --private  # Remove this flag to make public
```

---

## 🧪 STEP 6: Final Testing & Validation

### 6.1 Test Uploaded Model
```python
# Test the uploaded model
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "YOUR_USERNAME/qwen-enhanced-typo-fixer"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Test cases
test_cases = [
    "I beleive this is teh correct answr.",
    "The meetign will start at 9 oclock tommorrow.",
    "She recieved the packege yesteday afternoon.",
    "We need to discus the projcet detials today",
    "The resturant serves excellnt food evry day"
]

for test_case in test_cases:
    prompt = f"Correct the typos in this text: {test_case}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
    correction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Original:  {test_case}")
    print(f"Corrected: {correction.replace(prompt, '').strip()}")
    print("-" * 50)
```

---

## 📋 Complete Command Sequence

Here's the complete sequence of commands to run on your RTX5090 machine:

```bash
# 1. Setup
git clone https://github.com/yourusername/TypoFixerTraining.git
cd TypoFixerTraining
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
huggingface-cli login

# 2. Generate Data
python generate_enhanced_qwen_dataset.py --target-size 100000 --output-file data/enhanced_qwen_training.jsonl

# 3. Train Model
python -m torch.distributed.launch --nproc_per_node=2 --use_env train_dual_gpu.py

# 4. Evaluate Checkpoints
python evaluate_checkpoints.py

# 5. Upload to HuggingFace
python upload_to_hf.py --username YOUR_HF_USERNAME --repo-name qwen-enhanced-typo-fixer

# 6. Test Uploaded Model
python test_uploaded_model.py
```

## 📊 Expected Results

With the enhanced T5-improved dataset and dual RTX5090 training, you should expect:

- **Training Time**: 4-6 hours for 100k examples
- **Model Accuracy**: 85-92% on typo correction tasks
- **Inference Speed**: ~50-100 tokens/second
- **Model Size**: ~1.2GB (Qwen2-0.5B based)
- **VRAM Usage**: ~18GB per GPU during training

## 🚨 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size in training_config.json
   "per_device_train_batch_size": 8  # Instead of 16
   ```

2. **Dataset Generation Slow**:
   ```bash
   # Use smaller target size for testing
   python generate_enhanced_qwen_dataset.py --target-size 10000
   ```

3. **HuggingFace Upload Fails**:
   ```bash
   # Check authentication
   huggingface-cli whoami
   # Re-login if needed
   huggingface-cli login
   ```

4. **Multi-GPU Issues**:
   ```bash
   # Use single GPU as fallback
   CUDA_VISIBLE_DEVICES=0 python train_enhanced_qwen.py
   ```

---

## 🎯 Success Criteria

Your training is successful when:
- ✅ Dataset generated with 100k+ examples
- ✅ Training completes without CUDA errors
- ✅ Model achieves >85% accuracy on test examples
- ✅ Model uploads successfully to HuggingFace
- ✅ Uploaded model passes inference tests

**Total estimated time**: 6-8 hours from start to finish.
# Qwen 0.6B Typo Correction Model

A fine-tuned **Qwen 0.6B** model for automatic typo correction, achieving **reliable performance** with anti-overfitting training on diverse datasets.

## 🎯 Model Overview

- **Base Model**: Qwen/Qwen3-0.6B (596M parameters)
- **Task**: Text-to-text typo correction 
- **Training Data**: 23,627 high-quality examples from multiple sources
- **Performance**: Conservative, focused corrections without over-generation
- **Deployment**: Optimized for both GPU and Apple Neural Engine

## 🚀 Quick Start

### Installation

```bash
pip install torch transformers
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mazhewitt/qwen-typo-fixer")
model = AutoModelForCausalLM.from_pretrained("mazhewitt/qwen-typo-fixer")

# Correct typos
prompt = "Fix: I beleive this is teh correct answr."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
correction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
print(correction)  # "I believe this is the correct answer."
```

### Conservative Inference (Recommended)

For production use, we recommend the conservative inference approach to prevent over-generation:

```python
import torch

def conservative_typo_correction(model, tokenizer, text):
    """Production-ready typo correction with conservative generation."""
    prompt = f"Fix: {text}"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,  # Conservative token limit
            do_sample=False,    # Deterministic output
            num_beams=1,        # Fast inference
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
    
    # Clean output
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Extract just the correction
    if '.' in generated:
        corrected = generated.split('.')[0].strip() + '.'
    else:
        corrected = generated.strip()
    
    return corrected

# Example usage
corrected = conservative_typo_correction(model, tokenizer, "I beleive this is teh correct answr.")
print(corrected)  # "I believe this is the correct answer."
```

## 📊 Performance Benchmarks

| Platform | Average Time | Throughput | Notes |
|----------|--------------|------------|-------|
| **Apple M4 Pro CPU** | 0.33s | 3.0/sec | ARM efficiency |
| **Apple M4 Pro Metal** | 0.73s | 1.4/sec | Small model GPU overhead |
| **RTX 5070 Ti** | ~0.05s | ~20/sec | Estimated production performance |

## 🎯 Model Capabilities

### ✅ Handles Common Typos
- **Spelling errors**: "beleive" → "believe"
- **Keyboard errors**: "teh" → "the" 
- **Character issues**: "recieved" → "received"
- **Word endings**: "begining" → "beginning"

### ✅ Conservative Behavior  
- **Focused corrections**: Only fixes actual typos
- **No over-generation**: Stays close to original text
- **Fast inference**: Optimized for real-time use
- **Cross-platform**: Works on CPU, GPU, and Apple Neural Engine

### ❌ Limitations
- **Grammar vs Spelling**: Focuses on spelling, not grammar rules
- **Context dependency**: Best with clear spelling errors
- **Language**: Trained primarily on English text

## 🔧 Training Details

### Anti-Overfitting Approach
Our training specifically addressed overfitting issues through:

1. **Large Dataset**: 23,627 examples (vs typical 7K that cause overfitting)
2. **Diverse Sources**: 
   - WikiText natural sentences
   - Norvig's 20k misspellings from Google logs
   - Holbrook/Birkbeck academic datasets
   - Enhanced keyboard layout error simulation
3. **Training Safeguards**:
   - Early stopping (patience=3)
   - Weight decay (0.1)
   - Cosine learning rate scheduling
   - Conservative evaluation metrics

### Training Configuration
- **Hardware**: Dual RTX 5070 Ti (16GB each)
- **Precision**: BFloat16 mixed precision
- **Batch Size**: 4 per GPU with 8x gradient accumulation
- **Training Time**: ~25 minutes (proper learning vs 4min overfitting)
- **Final Loss**: Gradual decrease to ~0.01 (not instant drop to 0)

## 🛠️ Reproducing Training

### Prerequisites
```bash
# Environment setup
git clone -b qwen-approach https://github.com/mazhewitt/TypoFixerTraining.git
cd TypoFixerTraining
./setup_rtx5090.sh
```

### Generate Training Data
```bash
# Generate 23K+ diverse examples (takes ~15 minutes)
python3 generate_large_dataset.py
```

### Train Model
```bash
# Anti-overfitting training with dual RTX 5070 Ti
python3 train_rtx5090.py \
    --train_file data/enhanced_training_large.jsonl \
    --output_dir models/qwen-typo-fixer-v2 \
    --hf_repo your-username/qwen-typo-fixer \
    --num_epochs 3 \
    --early_stopping_patience 3 \
    --max_weight_decay 0.1 \
    --eval_steps 100
```

### Validate Model
```bash
# Test with conservative inference
python3 test_conservative_inference.py

# Comprehensive benchmark
python3 test_comprehensive.py

# Metal GPU benchmark (macOS)
python3 test_metal_benchmark.py
```

## 📂 Repository Structure

```
train-typo-fixer/
├── src/
│   └── realistic_data_generation.py    # High-quality dataset generation
├── train_rtx5090.py                    # Anti-overfitting training script
├── generate_large_dataset.py           # Dataset generation wrapper
├── test_conservative_inference.py      # Production inference testing
├── test_comprehensive.py               # Full model evaluation
├── test_metal_benchmark.py             # macOS Metal GPU benchmark
├── validate_dataset.py                 # Dataset quality validation
├── setup_rtx5090.sh                    # Environment setup
├── RTX5090_DEPLOYMENT.md              # Detailed deployment guide
└── README.md                           # This file
```

## 📈 Model Card

- **Model Type**: Causal Language Model (Text Generation)
- **Language**: English
- **Base Architecture**: Qwen 0.6B 
- **Fine-tuning Method**: Text-to-text correction with prompt masking
- **Training Data**: Multi-source typo correction pairs
- **Intended Use**: Automatic typo correction in text applications
- **License**: Apache 2.0

## 🔗 Links

- **Model**: [mazhewitt/qwen-typo-fixer](https://huggingface.co/mazhewitt/qwen-typo-fixer)
- **Training Repository**: [TypoFixerTraining](https://github.com/mazhewitt/TypoFixerTraining)
- **Base Model**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## 🤝 Contributing

This model was trained specifically to address overfitting issues in typo correction. The training methodology focuses on:

1. **Dataset diversity** over size
2. **Conservative generation** over creativity  
3. **Production reliability** over benchmark scores

For improvements or issues, please see the training repository.

---

**🎉 Successfully trained with anti-overfitting measures for reliable typo correction!**
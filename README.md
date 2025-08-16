# Qwen 0.6B Typo Correction Model

A fine-tuned **Qwen 0.6B** model for automatic typo correction, achieving **88.5% sentence accuracy** through optimized few-shot prompting and diverse training data.

## 🎯 Model Overview

- **Base Model**: Qwen/Qwen3-0.6B (596M parameters)
- **Task**: Text-to-text typo correction 
- **Training Data**: 23,627 high-quality examples from multiple sources
- **Accuracy**: 88.5% sentence accuracy (1.5% from 90% target)
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

### Optimized Inference (Recommended - 88.5% Accuracy)

For best performance, use few-shot prompting with examples:

```python
import torch

def optimized_typo_correction(model, tokenizer, corrupted_sentence):
    """Optimized inference achieving 88.5% sentence accuracy."""
    
    # Few-shot prompt with examples
    full_prompt = f"""Fix typos in these sentences:

Input: I beleive this is teh answer.
Output: I believe this is the answer.

Input: She recieved her degre yesterday.
Output: She received her degree yesterday.

Input: The resturant serves good food.
Output: The restaurant serves good food.

Input: {corrupted_sentence}
Output:"""
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=256)
    
    # Generate with optimal parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,  # Greedy decoding works best
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.02,
        )
    
    # Decode and clean output
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Extract just the corrected sentence
    generated = ' '.join(generated.split())
    if '\n' in generated:
        generated = generated.split('\n')[0].strip()
    
    # Remove suffixes that may appear
    for suffix in ['Input:', 'Output:', 'Human:', 'Assistant:']:
        if suffix.lower() in generated.lower():
            generated = generated.split(suffix)[0].strip()
    
    if '.' in generated:
        corrected = generated.split('.')[0].strip() + '.'
    else:
        corrected = generated.strip()
    
    return corrected

# Example usage
corrected = optimized_typo_correction(model, tokenizer, "I beleive this is teh correct answr.")
print(corrected)  # "I believe this is the correct answer."
```

## 📊 Performance Benchmarks

### Accuracy Results

| Approach | Accuracy | Notes |
|----------|----------|-------|
| **Original Fine-tuned** | 43.4% | Basic "Fix:" prompting |
| **Few-shot Prompting** | 40% | On challenging cases |
| **Optimized (Few-shot + Parameters)** | **88.5%** | Production-ready approach |
| **Base Model (No Training)** | 0% | Complete failure on typo correction |

### Speed Benchmarks

| Platform | Average Time | Throughput | Notes |
|----------|--------------|------------|-------|
| **Apple M4 Pro CPU** | 0.33s | 3.0/sec | ARM efficiency |
| **Apple M4 Pro Metal** | 0.63s | 1.6/sec | Metal overhead for small model |
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

### Quick Demo
```bash
# Run the demonstration script
python3 demo.py
```

### Prerequisites
```bash
# Environment setup
git clone -b qwen-approach https://github.com/mazhewitt/TypoFixerTraining.git
cd TypoFixerTraining
./scripts/training/setup_rtx5090.sh
```

### Generate Training Data
```bash
# Generate 23K+ diverse examples (takes ~15 minutes)
python3 src/realistic_data_generation.py --output data/enhanced_training_large.jsonl --num_examples 50000
```

### Train Model
```bash
# Anti-overfitting training with dual RTX 5070 Ti
python3 scripts/training/train_rtx5090.py \
    --train_file data/enhanced_training_large.jsonl \
    --output_dir models/qwen-typo-fixer-v2 \
    --hf_repo your-username/qwen-typo-fixer \
    --num_epochs 3 \
    --early_stopping_patience 3 \
    --max_weight_decay 0.1 \
    --eval_steps 100
```

### Test Model Performance
```bash
# Test optimized 88.5% accuracy approach (RECOMMENDED)
python3 scripts/testing/optimized_accuracy_test.py

# Compare fine-tuned vs base model performance
python3 scripts/testing/compare_models.py

# Metal GPU benchmark (macOS)
python3 scripts/testing/test_metal_benchmark.py
```

## 📂 Repository Structure

```
train-typo-fixer/
├── src/
│   └── realistic_data_generation.py       # High-quality dataset generation
├── scripts/
│   ├── training/
│   │   ├── train_rtx5090.py               # Anti-overfitting training script
│   │   ├── setup_rtx5090.sh               # Environment setup
│   │   └── validate_dataset.py            # Dataset quality validation
│   └── testing/
│       ├── optimized_accuracy_test.py     # 88.5% accuracy test (RECOMMENDED)
│       ├── compare_models.py              # Base vs fine-tuned comparison
│       ├── test_metal_benchmark.py        # macOS Metal GPU benchmark
│       └── test_conservative_inference.py # Basic inference testing
├── data/
│   └── enhanced_training_full.jsonl       # Training dataset
├── models/                                # Trained models and ANE conversions
├── anemll/                               # Apple Neural Engine conversion toolkit
├── demo.py                               # Interactive demonstration script
├── README.md                             # This file
└── CLAUDE.md                             # Project documentation
```

### Key Scripts:
- **`demo.py`** - Interactive demonstration of basic vs optimized approaches
- **`scripts/testing/optimized_accuracy_test.py`** - Test the 88.5% accuracy approach (recommended)
- **`scripts/testing/compare_models.py`** - Compare fine-tuned vs base model performance  
- **`scripts/testing/test_metal_benchmark.py`** - macOS performance benchmarking  
- **`scripts/training/train_rtx5090.py`** - Anti-overfitting training pipeline

## 🧪 Run on Apple Neural Engine (CoreML ANE-flex)

This repo includes CoreML artifacts exported for Apple Neural Engine with a fixed prefill sequence length S=128 and a single-token infer path. Use the provided test script to run an end-to-end few-shot example locally on macOS.

### Prerequisites
- macOS with CoreML runtime (Xcode CLT recommended)
- Python deps from this repo’s `requirements.txt` (includes coremltools)

### Model files
Ensure these files exist under `models/qwen-typo-fixer-ane-flex/`:
- `qwen-typo-fixer_embeddings.mlpackage`
- `qwen-typo-fixer_prefill_chunk_01of01.mlpackage` (prefill S=128)
- `qwen-typo-fixer_FFN_chunk_01of01.mlpackage` (single-token infer)
- `qwen-typo-fixer_lm_head.mlpackage`

### Verify IO shapes (optional)
```bash
python3 scripts/testing/inspect_coreml_shapes.py models/qwen-typo-fixer-ane-flex
```
You should see:
- Embeddings input_ids enumerated shapes: [1,1] and [1,128]
- Prefill inputs: hidden_states [1,128,1024], position_ids [128], causal_mask [1,1,128,256], current_pos [1]
- Infer inputs: single-token hidden_states [1,1,1024], position_ids [1], causal_mask [1,1,1,256], current_pos [1]

### Run the few-shot example
```bash
PYTHONPATH=. python3 scripts/testing/test_fewshot_long.py
```
What it does:
- Builds a few-shot prompt, prefill-runs the prompt (S up to 128), then generates up to 60 tokens greedily.
- The pipeline cleans the output to return the last complete corrected sentence (removing any stray “Human:”/“Assistant:” text).

Notes:
- The script uses the tokenizer `mazhewitt/qwen-typo-fixer` locally; network access is only needed the first time to cache it.
- Warnings like “scikit-learn version … not supported” or “Torch … not tested with coremltools” are benign here.
- If you want the simpler basic prompt, change `use_basic=False` to `True` in `scripts/testing/test_fewshot_long.py`.

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
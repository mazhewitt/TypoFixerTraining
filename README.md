# Enhanced Qwen Typo Fixer

A fine-tuned **Qwen 2-0.5B** model for automatic typo correction using **advanced T5-improved training data** with sophisticated error patterns, multi-domain coverage, and balanced punctuation handling.

## 🎯 Model Overview

- **Base Model**: Qwen/Qwen2-0.5B (500M parameters)
- **Task**: Text-to-text typo correction with instruction following
- **Training Data**: 70,000+ enhanced examples with T5-improved generation
- **Features**: Multi-domain coverage, advanced error patterns, punctuation balance
- **Performance**: Enhanced accuracy through sophisticated data generation
- **Deployment**: GPU-optimized for RTX5090 and other high-end hardware

## 🚀 Quick Start

### Complete Training Pipeline (RTX5090)

For the complete end-to-end training experience:

```bash
# 1. Setup
git clone https://github.com/mazhewitt/TypoFixerTraining.git
cd TypoFixerTraining
./setup_safe.sh

# 2. Generate Enhanced Dataset (100K examples)
python generate_enhanced_qwen_dataset.py --target-size 100000

# 3. Train Model (Single GPU - Recommended)
python train_single_gpu.py

# 4. Evaluate Best Checkpoint
python evaluate_checkpoints.py

# 5. Upload to HuggingFace
python upload_to_hf.py --username YOUR_HF_USERNAME
```

**Expected Results:**
- **Training Time**: 6-8 hours on RTX5090
- **Dataset**: 70,000+ enhanced examples with T5 improvements
- **Model Accuracy**: Expected 85-92% on typo correction tasks
- **Model Size**: ~1.2GB (Qwen2-0.5B based)

### Using Pre-trained Model

```bash
pip install torch transformers huggingface_hub
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/qwen-enhanced-typo-fixer", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/qwen-enhanced-typo-fixer", trust_remote_code=True)

# Correct typos using instruction format
prompt = "<|im_start|>user\nCorrect the typos in this text: I beleive this is teh correct answr.<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=False)
correction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(correction)  # Shows the corrected text
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

## 🛠️ Enhanced Training Pipeline

### Prerequisites (RTX5090 or Similar GPU)
```bash
# Clone repository
git clone https://github.com/yourusername/TypoFixerTraining.git
cd TypoFixerTraining

# Safe setup for preconfigured GPU machines
chmod +x setup_safe.sh
./setup_safe.sh
```

### Generate Enhanced Training Data
```bash
# Generate 100K examples with T5 improvements (takes ~30 minutes)
python generate_enhanced_qwen_dataset.py \
    --target-size 100000 \
    --output-file data/enhanced_qwen_training.jsonl \
    --seed 42
```

**Enhanced Dataset Features:**
- **Advanced Error Patterns**: Keyboard layouts, homophones, phonetic confusions
- **Multi-Domain Coverage**: Conversational, professional, educational, creative, instructional, general
- **Punctuation Balance**: 50/50 split with/without ending punctuation
- **Complex Multi-Error Sentences**: Realistic corruption patterns
- **Quality Filtering**: Advanced sentence selection and validation

### Train Model

**Option A: Single GPU (Recommended - More Stable)**
```bash
# Single RTX5090 training (batch size optimized for one GPU)
python train_single_gpu.py
```

**Option B: Dual GPU (If No CUDA Issues)**
```bash
# Dual RTX5090 distributed training
torchrun --nproc_per_node=2 train_dual_gpu.py
```

**Training Features:**
- **Model**: Qwen/Qwen2-0.5B with instruction format
- **Batch Size**: 32 (single) or 16×2 (dual GPU)
- **Mixed Precision**: FP16 for efficiency
- **Early Stopping**: Prevents overfitting
- **Wandb Integration**: Experiment tracking

### Evaluate and Deploy

**Evaluate Checkpoints:**
```bash
# Find best checkpoint and test accuracy
python evaluate_checkpoints.py
```

**Upload to HuggingFace:**
```bash
# Upload trained model to HF Hub
python upload_to_hf.py \
    --username YOUR_HF_USERNAME \
    --repo-name qwen-enhanced-typo-fixer \
    --model-path models/qwen-enhanced-typo-fixer
```

**Test Deployed Model:**
```bash
# Test uploaded model from HF Hub
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'YOUR_USERNAME/qwen-enhanced-typo-fixer'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print('✅ Model loaded successfully from HuggingFace!')
"
```

## 📂 Repository Structure

```
TypoFixerTraining/
├── Enhanced Training Pipeline:
│   ├── generate_enhanced_qwen_dataset.py     # T5-improved dataset generation
│   ├── train_enhanced_qwen.py                # Main training script
│   ├── train_single_gpu.py                   # Single GPU training (recommended)
│   ├── train_dual_gpu.py                     # Dual GPU distributed training
│   ├── evaluate_checkpoints.py               # Checkpoint evaluation & selection
│   └── upload_to_hf.py                       # HuggingFace model upload
│
├── Advanced Data Generation (T5 Improvements):
│   ├── scripts/advanced_training_data_generator.py  # Advanced error patterns
│   ├── scripts/error_pattern_library.py             # Sophisticated error types
│   ├── scripts/source_text_diversifier.py           # Multi-domain text collection
│   └── scripts/quality_validator.py                 # Data quality validation
│
├── Setup & Configuration:
│   ├── setup_safe.sh                         # Safe environment setup
│   ├── training_config.json                  # Training hyperparameters
│   ├── requirements.txt                      # Minimal dependencies
│   └── RTX5090_TRAINING_GUIDE.md            # Complete training guide
│
├── Legacy & T5 Research:
│   ├── src/realistic_data_generation.py      # Original data generation
│   ├── t5/                                   # T5 model experiments
│   ├── qwen/                                 # Original Qwen implementation
│   └── anemll/                               # Apple Neural Engine toolkit
│
└── Documentation:
    ├── README.md                             # This file (updated)
    └── CLAUDE.md                             # Project evolution log
```

### Key Scripts:
- **`generate_enhanced_qwen_dataset.py`** - Generate T5-improved training data
- **`train_single_gpu.py`** - Stable single GPU training (recommended)
- **`evaluate_checkpoints.py`** - Find best checkpoint automatically
- **`upload_to_hf.py`** - Deploy trained model to HuggingFace
- **`setup_safe.sh`** - Safe setup for preconfigured GPU machines

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
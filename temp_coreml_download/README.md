# Qwen Typo Fixer - CoreML Standalone

This is a standalone CoreML implementation of the Qwen typo correction model, containing all necessary components for typo correction inference using Apple's CoreML framework.

## Contents

- **CoreML Models** (3 components):
  - `qwen-typo-fixer_embeddings.mlpackage` - Token embeddings
  - `qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage` - FFN with prefill/infer functions
  - `qwen-typo-fixer_lm_head_lut6.mlpackage` - Language model head

- **Tokenizer Files**:
  - `tokenizer.json` - Main tokenizer configuration
  - `tokenizer_config.json` - Tokenizer metadata
  - `vocab.json` - Vocabulary mapping
  - `merges.txt` - BPE merge rules
  - `config.json` - Model configuration

- **Python Implementation**:
  - `typo_fixer_complete.py` - Complete working implementation

## Usage

### Prerequisites
```bash
pip install torch numpy coremltools transformers
```

### Run Typo Correction
```bash
python3 typo_fixer_complete.py
```

### Example Output
```
Original:  'I beleive this is teh correct answr.'
Corrected: 'I believe this is the correct answer.'
Status:    ✅ Typos likely fixed!
```

## Model Architecture

This uses a 3-component ANEMLL (Apple Neural Engine Multi-component Language Model) architecture:

1. **Embeddings**: Converts token IDs to hidden representations
2. **FFN+Prefill**: Dual-function model with prefill and infer functions for efficient generation
3. **LM Head**: Final linear layer producing vocabulary logits

## Performance

- **Model Size**: 596M parameters (Qwen3-0.6B based)
- **Accuracy**: 88.5% sentence accuracy on typo correction
- **Inference Speed**: ~300ms per correction
- **Memory**: Optimized for Apple Neural Engine acceleration

## Fine-tuned Model Details

- **Base Model**: Qwen/Qwen3-0.6B
- **Fine-tuned For**: Typo correction with 88.5% accuracy
- **Training Data**: Enhanced typo correction dataset
- **Original HuggingFace Model**: `mazhewitt/qwen-typo-fixer`

This standalone package contains the working CoreML models that have been tested and verified to work correctly.
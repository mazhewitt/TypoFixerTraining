# Qwen ANE Typo Fixer - Complete Architecture & Demo

## PROJECT OVERVIEW

This project demonstrates the complete pipeline from fine-tuned typo correction models to production deployment using Apple Neural Engine (ANE) acceleration via the candle-coreml library.

### 🎯 Current Status (August 2025)

✅ **COMPLETE END-TO-END PIPELINE:**
- Fine-tuned Qwen models for typo correction (88.5% accuracy)
- CoreML models converted from safetensors and deployed to HuggingFace
- Working Python implementation with multi-component ANEMLL architecture
- Demo Rust CLI app using candle-coreml library for model integration

## ARCHITECTURE OVERVIEW

### 1. Fine-Tuned Models
- **Base Model**: `Qwen/Qwen3-0.6B` (596M parameters)
- **Fine-tuned Version**: `mazhewitt/qwen-typo-fixer` 
- **Performance**: 88.5% sentence accuracy on typo correction
- **Deployment**: Available on HuggingFace with CoreML models

### 2. CoreML Multi-Component Pipeline (ANEMLL Approach)
The typo fixer uses a 3-component architecture optimized for Apple Neural Engine:

```
Input Tokens → [Embeddings] → [FFN+Prefill/Infer] → [LM Head] → Output Logits
               ↓              ↓                      ↓
               qwen-typo-     qwen-typo-fixer_       qwen-typo-fixer_
               fixer_         FFN_chunk_01of01.      lm_head.
               embeddings.    mlpackage &            mlpackage
               mlpackage      qwen-typo-fixer_
                              prefill_chunk_01of01.
                              mlpackage
```

#### Component Details:
1. **Embeddings Model** (`qwen-typo-fixer_embeddings.mlpackage`)
   - Converts token IDs to hidden representations
   - Input: `input_ids` [batch, seq_len]
   - Output: `hidden_states` [batch, seq_len, 1024]

2. **FFN Models**: 
   - **FFN Model** (`qwen-typo-fixer_FFN_chunk_01of01.mlpackage`) - Inference function
   - **Prefill Model** (`qwen-typo-fixer_prefill_chunk_01of01.mlpackage`) - Prefill function
   - **Prefill**: Processes full prompt context (batch processing)
   - **Infer**: Generates single tokens autoregressively
   - Maintains KV-cache state for efficient generation

3. **LM Head Model** (`qwen-typo-fixer_lm_head.mlpackage`)
   - Final linear layer producing vocabulary logits
   - Input: `hidden_states` [batch, 1, 1024]
   - Output: 16-part logits that concatenate to [batch, 1, vocab_size]

### 3. Model Locations

#### HuggingFace Hub
- **Primary Model**: https://huggingface.co/mazhewitt/qwen-typo-fixer
- **CoreML Repository**: https://huggingface.co/mazhewitt/qwen-typo-fixer-coreml
- **CoreML Models**: Contains all `.mlpackage` files for deployment
- **Tokenizer**: Standard Qwen tokenizer with typo correction fine-tuning

#### Local Development
- **Training Project**: `/Users/mazdahewitt/projects/train-typo-fixer`
- **Working Python Implementation**: `/Users/mazdahewitt/projects/train-typo-fixer/typo_fixer_complete.py`
- **Demo Rust CLI**: `/Users/mazdahewitt/projects/typo-fixer-cli`

## WORKING IMPLEMENTATIONS

### 1. Python Reference Implementation (`typo_fixer_complete.py`)

✅ **Fully Working** - Demonstrates complete ANEMLL pipeline:
- Multi-component model loading with separate functions
- Prefill/infer autoregressive generation
- KV-cache state management
- 88.5% accuracy typo correction

Key features:
```python
class CoreMLTypoFixer:
    def __init__(self, model_dir, tokenizer_path):
        # Loads all 3 components: embeddings, FFN (prefill/infer), LM head
    
    def fix_typos(self, text_with_typos):
        # Complete pipeline: tokenize → prefill → generate → clean output
```

### 2. Rust CLI Demo App (`/Users/mazdahewitt/projects/typo-fixer-cli`)

🔧 **In Progress** - Demonstrates candle-coreml integration:
- CLI interface for typo correction
- Integration with HuggingFace model downloads
- Uses candle-coreml library for CoreML model execution
- **Goal**: Prove candle-coreml works with custom fine-tuned models

## CANDLE-COREML INTEGRATION

### Library Role
- **candle-coreml**: Generic CoreML integration for Candle tensors
- **Keeps Generic**: Not hardcoded to specific model implementations  
- **Provides**: `QwenModel`, `CoreMLModel`, model downloading, tensor handling

### Integration Pattern (From candle-coreml README)
```rust
use candle_coreml::{QwenModel, model_downloader};

// Download model from HuggingFace Hub
let model_path = model_downloader::ensure_model_downloaded(model_id, verbose)?;

// Load multi-component model
let model = QwenModel::load_from_directory(&model_path, Some(config))?;

// Generate text
let result = model.generate_text(prompt, max_tokens, temperature)?;
```

## NEXT STEPS

### 🎯 Current Task: Integrate Fine-tuned Models with Rust CLI

Based on candle-coreml README instructions:

1. **Modify typo-fixer-cli** to use `mazhewitt/qwen-typo-fixer` models
2. **Reference typo_fixer_complete.py** for multi-component architecture patterns
3. **Update QwenModel in candle-coreml** if needed for typo fixer model structure
4. **Test end-to-end** typo correction in Rust CLI

### Implementation Strategy
- Use candle-coreml's built-in model downloader for HuggingFace integration
- Adapt `QwenModel` to handle typo fixer's specific component naming
- Implement same prefill/infer pattern as Python reference
- Maintain generic candle-coreml library design

## PROJECT STRUCTURE

```
/Users/mazdahewitt/projects/train-typo-fixer/    # Training & Python reference
├── typo_fixer_complete.py                       # ✅ Working Python implementation
├── models/qwen-typo-fixer-ane-flex/             # Local CoreML models (latest)
├── src/                                         # Training scripts
├── scripts/testing/                             # Validation scripts
└── CLAUDE.md                                    # This documentation

/Users/mazdahewitt/projects/typo-fixer-cli/      # 🔧 Rust CLI demo
├── src/
│   ├── main.rs                                  # CLI entry point
│   ├── typo_fixer.rs                            # Model integration (needs update)
│   └── prompt.rs                                # Few-shot prompt engineering
├── Cargo.toml                                   # candle-coreml dependency
└── README.md                                    # Demo app documentation

/Users/mazdahewitt/projects/candle-coreml/       # Generic CoreML library
├── src/qwen.rs                                  # QwenModel implementation
├── examples/                                    # ANEMLL examples
└── README.md                                    # Integration instructions
```

## VALIDATION RESULTS

### Python Implementation (typo_fixer_complete.py)
✅ **Test Results:**
```
Original:  'I beleive this is teh correct answr.'
Corrected: 'I believe this is the correct answer.'
Status:    ✅ Typos likely fixed!
```

### Performance Benchmarks
- **Model Loading**: ~8 seconds (first time download)
- **Inference**: ~300ms per correction
- **Accuracy**: 88.5% sentence accuracy
- **Memory**: Efficient multi-component architecture

## CONCLUSION

This project successfully demonstrates:

1. **✅ Complete Training Pipeline** - From base Qwen to fine-tuned typo correction
2. **✅ CoreML Deployment** - Multi-component ANEMLL architecture working
3. **✅ Python Reference** - Fully functional typo correction with 88.5% accuracy
4. **🔧 Rust Integration** - Demo CLI proving candle-coreml library effectiveness

**Next Phase**: Complete the Rust CLI demo to showcase candle-coreml as a production-ready library for custom fine-tuned models.
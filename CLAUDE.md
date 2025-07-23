# Qwen ANE Typo Fixer

## PURPOSE
Fine-tune Qwen 0.6B for typo correction and deploy with Apple Neural Engine (ANE) acceleration using the anemll conversion pipeline.

## CURRENT STATUS (January 2025)
✅ **PHASE 1 & 2 COMPLETED:**
- Successfully converted Qwen 0.6B to ANE using anemll toolkit
- Established performance baselines: GPU (24.5 TPS) vs ANE (6.7 TPS)
- Created realistic single-sentence data generation from WikiText
- Validated ANE model functionality with typo correction tests
- **DEPLOYMENT READY**: Both GPU and ANE conversion pipelines validated

## ARCHITECTURE
### Base Model
- **Standard Training**: `Qwen/Qwen3-0.6B` (596M parameters)
- **ANE Deployment**: anemll-converted CoreML model with chunked FFN
- **Task**: Text-to-text typo correction with natural prompts
- **Context Length**: 256 tokens (ANE optimized)

### Performance Results (Phase 2 Validated)
- **Apple GPU (MPS)**: 24.5 tokens/second
- **Apple Neural Engine**: 6.7 tokens/second  
- **Tradeoff**: ANE is 3.7x slower but more power-efficient
- **Deployment Options**: Choose GPU for speed or ANE for efficiency

## PROJECT STRUCTURE (Clean)
```
train-typo-fixer/
├── src/
│   ├── realistic_data_generation.py    # Extract real sentences from WikiText
│   ├── realistic_validation.py         # Sentence-focused validation pipeline
│   ├── train.py                       # Fine-tuning pipeline (Phase 3)
│   ├── qwen_baseline_test.py          # Baseline performance testing
│   └── download_qwen.py               # Model downloading utility
├── models/
│   ├── qwen-0.6b/                     # Base model for fine-tuning
│   └── qwen-ane-test/                 # ANE converted model (6.7 TPS)
├── data/
│   └── realistic_sample.jsonl         # High-quality single sentences
├── anemll/                            # Complete ANE conversion toolkit
├── phase2/
│   ├── anemll_performance_test.py     # ANE performance testing
│   ├── qwen_baseline_tps_test.py      # GPU baseline testing (24.5 TPS)
│   └── *_results.json                # Benchmark results
└── CLAUDE.md                          # This file
```

## USAGE

### 1. Generate Realistic Training Data
```bash
python src/realistic_data_generation.py \
  --dataset wikitext \
  --num_sentences 10000 \
  --output data/processed/realistic_train.jsonl \
  --corruption_rate 0.15
```

### 2. Fine-tune Qwen (Phase 3)
```bash
python src/train.py \
  --model_name models/qwen-0.6b \
  --train_file data/processed/realistic_train.jsonl \
  --output_dir models/qwen-typo-fixer \
  --num_train_epochs 3
```

### 3. Convert Fine-tuned Model to ANE
```bash
./anemll/utils/convert_model.sh \
  --model models/qwen-typo-fixer \
  --output models/qwen-typo-fixer-ane \
  --context 256 --chunk 1
```

### 4. Performance Testing
```bash
# Test GPU performance
python phase2/qwen_baseline_tps_test.py --model models/qwen-typo-fixer

# Test ANE performance
python phase2/anemll_performance_test.py --model_path models/qwen-typo-fixer-ane
```

## PHASE 3: FINE-TUNING (READY TO START)

### Objectives
- Fine-tune Qwen 0.6B on realistic single-sentence typo correction
- Achieve >90% word-level accuracy on validation set
- Maintain or improve inference speed
- Deploy with both GPU and ANE options

### Data Strategy
- **Source**: WikiText single sentences (natural, not synthetic paragraphs)
- **Corruption Types**: Keyboard errors, homophones, character drops/doubles
- **Format**: Direct text-to-text correction prompts
- **Size**: 10,000+ sentence pairs for robust training

### Training Approach
- **Method**: Text-to-text fine-tuning (not MLM)
- **Prompt**: `"Correct the typos: {corrupted_text}"`
- **Target**: `{clean_text}`
- **Optimization**: Focus on single-sentence accuracy over speed

## PROVEN RESULTS
✅ **ANE Conversion**: Complete pipeline working with anemll
✅ **Performance Benchmarks**: GPU 24.5 TPS, ANE 6.7 TPS
✅ **Data Pipeline**: Realistic sentence extraction validated
✅ **Deployment Options**: Both GPU and ANE paths ready

**Ready for Phase 3 fine-tuning when requested.**
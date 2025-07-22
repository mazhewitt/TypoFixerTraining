# Source Code Directory

This directory contains the final, working scripts for the DistilBERT ANE Typo Fixer project.

## Core Scripts

### Data & Training
- **`data_generation.py`** - Generates synthetic typo data with keyboard errors and homophone confusions
- **`train.py`** - Fine-tunes DistilBERT on typo correction data (standard PyTorch training)
- **`validate.py`** - Validates model performance on token-level accuracy

### ANE Deployment
- **`apple_ane_conversion.py`** - Converts fine-tuned weights to Apple's ANE format using proven approach
- **`bc1s_inference.py`** - BC1S inference interface for Apple's ANE Core ML models
- **`ane_vs_cpu_benchmark.py`** - Benchmarks ANE vs CPU performance to prove acceleration

### Architecture
- **`ane_models/`** - Apple's official ANE DistilBERT architecture files
  - `modeling_distilbert_ane.py` - Conv2d-optimized DistilBERT for ANE
  - `configuration_distilbert_ane.py` - ANE-specific configuration

## Usage Pipeline

```bash
# 1. Generate training data
python data_generation.py

# 2. Train model
python train.py --output_dir models/test_model

# 3. Convert to ANE format  
python apple_ane_conversion.py --input_model models/test_model

# 4. Benchmark ANE performance
python ane_vs_cpu_benchmark.py --coreml_model apple-ane-distilbert/DistilBERT_fp16.mlpackage

# 5. Run inference
python bc1s_inference.py --text "Thi sis a test with typos"
```

## Key Technical Features

- **Apple ANE Integration**: Uses Apple's proven Conv2d architecture and weight conversion
- **BC1S Format**: Handles Batch-Channel-1-Sequence tensor layout for Neural Engine
- **Performance Validation**: Confirmed 1.3x speedup over CPU-only inference
- **Production Ready**: Clean pipeline from training to deployment

All experimental conversion scripts have been removed - these represent the final, working implementation.
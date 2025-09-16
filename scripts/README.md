# Training Scripts Directory

Clean, organized scripts for the advanced typo correction training pipeline.

## 🚀 **Advanced Training Data Pipeline**

### Core Pipeline Components
- **`error_pattern_library.py`** - Sophisticated human-like error patterns (400+ patterns)
- **`source_text_diversifier.py`** - Multi-domain real dataset collection 
- **`advanced_training_data_generator.py`** - Advanced generation with 5 methods
- **`quality_validator.py`** - Comprehensive quality validation (15+ rules)
- **`training_data_analyzer.py`** - Statistical analysis + visualizations
- **`generate_advanced_dataset.py`** - Master pipeline orchestrator

### Quick Start
```bash
# Generate production dataset (100K examples)
../generate_production_dataset.sh

# Or run manually
python generate_advanced_dataset.py --target-size 100000 --validate-quality --create-analysis
```

## 🎯 **Training & Evaluation**

### Training
- **`train_byt5_nocallback.py`** - Stable ByT5 training with frequent checkpoints
- **`training/train_byt5_rtx5090.py`** - RTX 5090 optimized training
- **`training/setup_rtx5090.sh`** - RTX 5090 environment setup

### Evaluation  
- **`evaluate_all_checkpoints.py`** - Systematic checkpoint evaluation to find optimal stopping point

### Deployment
- **`upload_and_replace_model.py`** - Upload trained models to HuggingFace Hub

## 🔧 **Utilities**

### Testing
- **`testing/compare_models.py`** - Compare model performance
- **`testing/optimized_accuracy_test.py`** - Comprehensive accuracy testing
- **`testing/test_conservative_inference.py`** - Conservative inference testing
- **`testing/test_fewshot_long.py`** - Few-shot learning tests
- **`testing/test_metal_benchmark.py`** - Metal performance benchmarks
- **`testing/inspect_coreml_shapes.py`** - CoreML model inspection

### Publishing
- **`publish/publish_to_hf.py`** - HuggingFace publication utilities
- **`publish/publish_coreml_repo.sh`** - CoreML repository publishing
- **`publish/README_COREML_REPO_TEMPLATE.md`** - CoreML repo template

## 📊 **Workflow**

### 1. Generate Advanced Dataset
```bash
python generate_advanced_dataset.py --target-size 100000 --validate-quality --create-analysis
```

### 2. Train Model
```bash
python train_byt5_nocallback.py --train-file data/advanced_dataset/training_dataset.jsonl
```

### 3. Find Best Checkpoint
```bash  
python evaluate_all_checkpoints.py --model-path models/your-model
```

### 4. Upload Best Model
```bash
python upload_and_replace_model.py --model-path models/your-model/checkpoint-XXXX --hub-model-id your-id/model-name
```

---

**All old/obsolete scripts have been removed.** This directory now contains only the advanced pipeline and proven utilities! 🚀
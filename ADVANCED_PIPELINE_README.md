# Advanced Typo Correction Training Data Pipeline

A sophisticated system for generating high-quality training data for typo correction models, designed to improve upon previous approaches that achieved only 27% accuracy.

## 🎯 Overview

This pipeline addresses the core limitations of simple typo generation by implementing:
- **Realistic human-like error patterns** instead of random corruptions
- **Multi-domain text sources** for diverse training contexts  
- **Balanced complexity distribution** from simple to complex scenarios
- **Comprehensive quality validation** with automated recommendations
- **Detailed analysis and visualization** for dataset insights

## 🏗️ Architecture

### Core Components

1. **Error Pattern Library** (`scripts/error_pattern_library.py`)
   - 400+ sophisticated error patterns based on human typing behavior
   - Keyboard adjacency errors, phonetic confusions, homophones
   - Contextual grammar errors, contractions, compound word mistakes
   - Difficulty scoring and complexity classification

2. **Source Text Diversifier** (`scripts/source_text_diversifier.py`)
   - Multi-domain sentence collection from real datasets (WikiText, BookCorpus, OpenWebText)
   - Domain classification: conversational, professional, educational, creative, instructional, general
   - Quality filtering and deduplication
   - Balanced distribution across domains

3. **Advanced Training Data Generator** (`scripts/advanced_training_data_generator.py`)
   - 5 generation methods with specific error patterns:
     - **Multi-error sentences** (40%): Complex realistic scenarios with 2-6 errors
     - **Contextual/phonetic** (20%): Sound-based confusions and grammar errors
     - **Advanced typography** (15%): Double letters, compound word issues
     - **Keyboard patterns** (15%): Adjacent key mistakes and fat-finger errors
     - **Simple single errors** (10%): Basic single-word corrections
   - Metadata tracking: difficulty scores, error types, complexity levels

4. **Quality Validator** (`scripts/quality_validator.py`)
   - 15+ validation rules for data quality assurance
   - Change ratio analysis, readability checks, meaningfulness validation
   - Distribution balance verification across complexity/domain/error types
   - Automatic quality recommendations

5. **Training Data Analyzer** (`scripts/training_data_analyzer.py`)
   - Comprehensive dataset statistics and visualizations
   - Distribution analysis with balance scoring (entropy-based)
   - Quality metrics and flags
   - Matplotlib/Seaborn visualizations for data insights

6. **Master Pipeline** (`scripts/generate_advanced_dataset.py`)
   - Orchestrates complete generation workflow
   - Integrated quality validation and analysis
   - Comprehensive reporting with actionable recommendations

## 🚀 Quick Start

### Prerequisites

```bash
pip install datasets matplotlib seaborn pandas numpy transformers torch
```

### Generate Production Dataset (100K examples)

```bash
# Run complete pipeline
./generate_production_dataset.sh

# Or manually with custom parameters
python scripts/generate_advanced_dataset.py \
    --target-size 100000 \
    --source-sentences 50000 \
    --validate-quality \
    --create-analysis \
    --output-dir data/production_dataset
```

### Train Model with Advanced Dataset

```bash
# Train ByT5 with advanced data
python scripts/train_byt5_nocallback.py \
    --train-file data/production_dataset/training_dataset.jsonl \
    --output-dir models/byt5-advanced-typo-fixer \
    --num-epochs 3 \
    --batch-size 8 \
    --learning-rate 3e-5

# Evaluate all checkpoints systematically  
python scripts/evaluate_all_checkpoints.py \
    --model-path models/byt5-advanced-typo-fixer
```

## 📊 Data Quality & Distribution

### Target Distribution
- **Complexity**: 30% simple, 50% medium, 20% complex
- **Error Types**: 40% spelling, 20% keyboard, 15% phonetic, 10% punctuation
- **Domains**: 25% conversational, 20% professional, 20% educational, 15% creative, 10% instructional, 10% general

### Quality Metrics
- **Change Ratios**: 5-50% character changes, 10-60% word changes
- **Readability**: Corrupted text maintains semantic meaning
- **Error Distribution**: Realistic patterns matching human typing behavior
- **Complexity Scoring**: 0-100 scale based on error types, sentence length, change ratio

### Sample Output
```json
{
  "corrupted": "The projct deadlien has been extemded untl next Friday.",
  "clean": "The project deadline has been extended until next Friday.", 
  "domain": "professional",
  "complexity": "medium",
  "error_types": ["spelling", "keyboard"],
  "num_errors": 3,
  "difficulty_score": 42.3,
  "source": "multi_error_advanced"
}
```

## 📈 Performance Improvements

### vs. Previous Simple Approach
- **Error Realism**: Human-like patterns vs random corruptions
- **Diversity**: Multi-domain sources vs single-source text
- **Complexity**: Balanced simple→complex vs all simple cases
- **Quality**: Comprehensive validation vs no quality checks
- **Coverage**: 5 error pattern types vs 2-3 basic types

### Expected Results
- **Target Accuracy**: 85-90% (up from 27% with previous approach)
- **Complex Scenario Handling**: Significant improvement in multi-error cases
- **Domain Robustness**: Better performance across different text types
- **Realistic Error Patterns**: Handles actual human typing mistakes

## 🔧 Customization

### Adjust Error Pattern Distribution
```python
# In advanced_training_data_generator.py
error_distribution_targets = {
    "multi_error_sentences": 0.50,    # Increase complex cases
    "contextual_phonetic": 0.25,      # More phonetic errors  
    "simple_single": 0.05,            # Reduce simple cases
    # ... other patterns
}
```

### Modify Complexity Targets
```python
# In advanced_training_data_generator.py
complexity_targets = {
    ComplexityLevel.SIMPLE: 0.20,     # Reduce simple
    ComplexityLevel.MEDIUM: 0.50,     # Keep medium
    ComplexityLevel.COMPLEX: 0.30,    # Increase complex
}
```

### Add Custom Domains
```python
# In source_text_diversifier.py  
domain_configs["technical"] = {
    "description": "Technical documentation",
    "keywords": ["function", "parameter", "method", "class"],
    "target_ratio": 0.15
}
```

## 📁 Output Structure

```
data/production_dataset/
├── training_dataset.jsonl          # Main training file (JSONL format)
├── training_dataset_metadata.json  # Generation statistics
├── source_sentences.json           # Collected source sentences
├── validation_report.json          # Quality validation results
├── analysis_report.json           # Comprehensive analysis
├── generation_summary.json        # Pipeline execution summary
└── plots/                         # Data visualization charts
    ├── distribution_overview.png   # Key distributions
    ├── difficulty_distribution.png # Difficulty score histogram
    └── corruption_ratios.png       # Change ratio analysis
```

## 🔍 Validation & Analysis

### Quality Validation
- **Automatic validation** with 15+ quality rules
- **Failed example detection** and filtering recommendations
- **Distribution balance** verification across all dimensions
- **Readability preservation** checks

### Comprehensive Analysis  
- **Statistical summaries** of all key metrics
- **Distribution visualizations** with balance scores
- **Quality flags** for potential issues
- **Actionable recommendations** for improvement

### Reports Generated
- **Validation Report**: Detailed quality assessment with specific issues
- **Analysis Report**: Comprehensive dataset statistics and metrics
- **Generation Summary**: Pipeline execution details and configuration

## 🎯 Integration with Existing Workflow

This advanced pipeline seamlessly integrates with your existing training scripts:

```bash
# Replace simple data generation
# OLD: python scripts/generate_100k_dataset_fixed.py 
# NEW: python scripts/generate_advanced_dataset.py

# Use with existing training
python scripts/train_byt5_nocallback.py --train-file [advanced_dataset]

# Use with existing evaluation  
python scripts/evaluate_all_checkpoints.py --model-path [model]

# Use with existing upload
python scripts/upload_and_replace_model.py --model-path [model]
```

## 🐛 Troubleshooting

### Low Success Rate
- Increase source sentence diversity
- Adjust complexity targets  
- Review quality validation thresholds

### High Validation Failures
- Reduce corruption intensity
- Increase minimum change ratios
- Review error pattern implementations

### Poor Domain Balance
- Adjust domain target ratios
- Improve domain classification rules
- Add more diverse source datasets

## 📝 Contributing

To extend the pipeline:
1. Add new error patterns in `error_pattern_library.py`
2. Implement new generation methods in `advanced_training_data_generator.py` 
3. Add custom validation rules in `quality_validator.py`
4. Extend analysis metrics in `training_data_analyzer.py`

---

**Ready to generate production-quality typo correction training data!** 🚀

The advanced pipeline should significantly improve your ByT5 model performance from 27% to the target 85-90% accuracy by providing realistic, diverse, and well-balanced training examples.
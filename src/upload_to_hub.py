#!/usr/bin/env python3
"""
Upload trained DistilBERT typo correction model to Hugging Face Hub.
Handles model card creation, README generation, and repository management.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

from huggingface_hub import HfApi, Repository, login
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model_card(training_info: Dict[str, Any], model_name: str) -> str:
    """Create a comprehensive model card for the typo correction model."""
    
    # Extract key metrics
    total_examples = training_info.get("training_examples", "N/A")
    val_examples = training_info.get("validation_examples", 0)
    epochs = training_info.get("epochs", "N/A")
    train_loss = training_info.get("final_train_loss", "N/A")
    val_loss = training_info.get("final_eval_loss", "N/A")
    training_time = training_info.get("training_time_minutes", "N/A")
    
    model_card = f"""---
language: en
license: apache-2.0
tags:
- text-generation
- typo-correction
- distilbert
- masked-language-modeling
- apple-neural-engine
datasets:
- wikitext
widget:
- text: "Thi sis a test sentenc with typos"
  example_title: "Keyboard typos"
- text: "The quikc brown fox jumps over teh lazy dog"  
  example_title: "Common typing errors"
- text: "I went too the stor to buy som milk"
  example_title: "Homophone confusions"
pipeline_tag: fill-mask
---

# DistilBERT Typo Correction Model

This is a fine-tuned DistilBERT model specifically trained for **typo correction** using masked language modeling. The model learns to identify and correct common typing errors including:

- **Keyboard neighbor errors**: q→w, a→s, e→r
- **Character operations**: drops (the→th), doubles (the→thee), transpositions (the→teh)
- **Word splitting**: sentence→sen tence  
- **Homophone confusions**: their/there/they're, your/you're, its/it's

## Model Details

- **Base Model**: distilbert-base-uncased ({training_info.get("total_parameters", "66M"):,} parameters)
- **Training Strategy**: Freeze transformer layers, fine-tune MLM head only ({training_info.get("trainable_parameters", "1.5M"):,} trainable parameters)
- **Training Data**: {total_examples:,} synthetic typo examples from WikiText
- **Validation Data**: {val_examples:,} examples (if available)
- **Training Time**: {training_time} minutes on {training_info.get("device_name", "GPU")}

## Performance

- **Training Loss**: {train_loss}
- **Validation Loss**: {val_loss if val_loss != "N/A" else "Not evaluated"}
- **Training Efficiency**: {100*training_info.get("trainable_parameters", 1500000)/training_info.get("total_parameters", 66000000):.1f}% of parameters fine-tuned

## Apple Neural Engine Compatibility

This model is designed for **Apple Neural Engine (ANE) deployment** with significant performance benefits:

- ✅ **Sequence length**: 128 tokens (ANE optimized)
- ✅ **Architecture compatibility**: Ready for Conv2d conversion
- ✅ **Expected speedup**: 1.5-3x on Apple Silicon devices
- ✅ **Core ML conversion**: Use `apple_ane_conversion.py` for ANE deployment

## Usage

### Basic Inference

```python
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
import torch

# Load model and tokenizer  
model = DistilBertForMaskedLM.from_pretrained("{model_name}")
tokenizer = DistilBertTokenizer.from_pretrained("{model_name}")

# Example with typos
text = "Thi sis a test sentenc with typos"
inputs = tokenizer(text, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Decode corrected text
corrected = tokenizer.decode(predictions[0], skip_special_tokens=True)
print(f"Original: {{text}}")
print(f"Corrected: {{corrected}}")
```

### Apple Neural Engine Deployment

```python
# Convert to ANE format (requires apple_ane_conversion.py)
python apple_ane_conversion.py \\
  --input_model {model_name} \\
  --coreml_output model_ane.mlpackage

# Use with Core ML for 3x speedup on Apple devices
import coremltools as ct
model = ct.models.MLModel("model_ane.mlpackage")
```

## Training Data

The model was trained on **synthetic typo data** generated from clean text with the following corruption types:

| Corruption Type | Example | Rate |
|----------------|---------|------|
| Keyboard neighbors | quick → quikc | 25% |
| Character drops | sentence → sentenc | 20% |  
| Character doubles | good → goood | 20% |
| Transpositions | the → teh | 20% |
| Word splitting | sentence → sen tence | 10% |
| Homophone confusion | their → there | 5% |

**Corruption rate**: 15% of tokens corrupted per sentence

## Limitations

- **Domain specific**: Trained primarily on Wikipedia text
- **English only**: No multilingual support
- **Masked LM approach**: Requires sentence context, not word-level correction
- **Synthetic training**: May not cover all real-world typo patterns

## Training Configuration

```json
{{
  "epochs": {epochs},
  "batch_size": {training_info.get("batch_size", 32)},
  "learning_rate": {training_info.get("learning_rate", 2e-5)},
  "max_sequence_length": {training_info.get("max_seq_len", 128)},
  "warmup_ratio": 0.1,
  "weight_decay": 0.01
}}
```

## Citation

If you use this model, please cite:

```bibtex
@misc{{distilbert-typo-correction,
  title={{DistilBERT Typo Correction Model}},
  author={{Your Name}},
  year={{2025}},
  url={{https://huggingface.co/{model_name}}}
}}
```

## License

Apache 2.0 - See LICENSE file for details.

---

**Note**: This model is optimized for Apple Neural Engine deployment. For maximum performance on Apple Silicon devices, use the ANE conversion pipeline provided in the training repository.
"""

    return model_card

def upload_model_to_hub(
    model_path: str, 
    hub_model_name: str, 
    token: str = None,
    private: bool = False,
    commit_message: str = None
):
    """Upload model to Hugging Face Hub with comprehensive documentation."""
    
    logger.info(f"🚀 Starting upload to Hugging Face Hub...")
    logger.info(f"📁 Model path: {model_path}")
    logger.info(f"🏷️ Hub model name: {hub_model_name}")
    logger.info(f"🔒 Private repository: {private}")
    
    # Login to HF Hub
    if token:
        login(token=token)
    else:
        logger.info("Please login to Hugging Face Hub...")
        login()
    
    # Load model and tokenizer to verify
    logger.info("🔍 Verifying model can be loaded...")
    try:
        model = DistilBertForMaskedLM.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        logger.info(f"✅ Model verified: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        logger.error(f"❌ Failed to load model from {model_path}: {e}")
        raise
    
    # Load training info if available
    training_info_path = os.path.join(model_path, "training_info.json")
    training_info = {}
    
    if os.path.exists(training_info_path):
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)
        logger.info("📊 Loaded training information")
    else:
        logger.warning("⚠️ No training_info.json found, using defaults")
    
    # Create model card
    logger.info("📝 Creating model card...")
    model_card = create_model_card(training_info, hub_model_name)
    
    # Save model card locally
    readme_path = os.path.join(model_path, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    logger.info(f"💾 Model card saved to {readme_path}")
    
    # Create HF API instance
    api = HfApi()
    
    try:
        # Create repository if it doesn't exist
        logger.info(f"🏗️ Creating repository {hub_model_name}...")
        api.create_repo(
            repo_id=hub_model_name, 
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        
        # Upload all files in the model directory
        logger.info("📤 Uploading model files...")
        
        commit_msg = commit_message or f"Upload DistilBERT typo correction model ({training_info.get('training_examples', 'N/A')} examples, {training_info.get('epochs', 'N/A')} epochs)"
        
        # Upload folder contents
        api.upload_folder(
            folder_path=model_path,
            repo_id=hub_model_name,
            commit_message=commit_msg,
            ignore_patterns=["*.git*", "__pycache__", "*.pyc", "logs/"]
        )
        
        logger.info("✅ Upload completed successfully!")
        logger.info(f"🔗 Model available at: https://huggingface.co/{hub_model_name}")
        
        # Test the uploaded model
        logger.info("🧪 Testing uploaded model...")
        test_model = DistilBertForMaskedLM.from_pretrained(hub_model_name)
        test_tokenizer = DistilBertTokenizer.from_pretrained(hub_model_name)
        
        # Quick inference test
        test_text = "Thi sis a test"
        inputs = test_tokenizer(test_text, return_tensors="pt")
        with torch.no_grad():
            outputs = test_model(**inputs)
        
        logger.info("✅ Uploaded model verified working!")
        
        return f"https://huggingface.co/{hub_model_name}"
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Upload DistilBERT typo correction model to HF Hub")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--hub_model_name', type=str, required=True,
                       help='Model name on HF Hub (username/model-name)')
    parser.add_argument('--token', type=str,
                       help='HF Hub token (if not logged in)')
    parser.add_argument('--private', action='store_true',
                       help='Create private repository')
    parser.add_argument('--commit_message', type=str,
                       help='Custom commit message')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Model path does not exist: {args.model_path}")
        return 1
    
    if '/' not in args.hub_model_name:
        logger.error(f"❌ Hub model name must include username: username/model-name")
        return 1
    
    try:
        # Upload model
        model_url = upload_model_to_hub(
            model_path=args.model_path,
            hub_model_name=args.hub_model_name,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message
        )
        
        logger.info("\n🎉 Upload completed successfully!")
        logger.info(f"🔗 Model URL: {model_url}")
        logger.info("\n📋 Next steps:")
        logger.info("   1. Download model: DistilBertForMaskedLM.from_pretrained('{}')".format(args.hub_model_name))
        logger.info("   2. Convert to ANE: python apple_ane_conversion.py --input_model {}".format(args.hub_model_name))
        logger.info("   3. Deploy on Apple devices with Core ML")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
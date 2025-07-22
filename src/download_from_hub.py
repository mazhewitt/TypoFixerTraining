#!/usr/bin/env python3
"""
Download DistilBERT typo correction model from Hugging Face Hub for local ANE conversion.
Handles authentication, model verification, and local setup.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download, login
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model_from_hub(
    hub_model_name: str,
    local_path: str,
    token: str = None,
    force_download: bool = False
):
    """Download model from HF Hub and verify it works locally."""
    
    logger.info(f"📥 Downloading model from Hugging Face Hub...")
    logger.info(f"🏷️ Hub model name: {hub_model_name}")
    logger.info(f"📁 Local path: {local_path}")
    
    # Create local directory
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Login if token provided
    if token:
        login(token=token)
        logger.info("🔐 Logged in with provided token")
    
    try:
        # Download model snapshot
        logger.info("📦 Downloading model files...")
        snapshot_path = snapshot_download(
            repo_id=hub_model_name,
            cache_dir=str(local_path.parent),
            local_dir=str(local_path),
            force_download=force_download,
            local_dir_use_symlinks=False  # Copy files instead of symlinks
        )
        
        logger.info(f"✅ Model downloaded to: {snapshot_path}")
        
        # Verify model can be loaded
        logger.info("🔍 Verifying downloaded model...")
        
        model = DistilBertForMaskedLM.from_pretrained(local_path)
        tokenizer = DistilBertTokenizer.from_pretrained(local_path)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model verified:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Vocab size: {tokenizer.vocab_size:,}")
        
        # Test inference
        logger.info("🧪 Testing inference...")
        test_examples = [
            "Thi sis a test sentenc with typos",
            "The quikc brown fox jumps over teh lazy dog"
        ]
        
        model.eval()
        for i, test_text in enumerate(test_examples, 1):
            inputs = tokenizer(test_text, return_tensors="pt", max_length=128, 
                             padding="max_length", truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                corrected = tokenizer.decode(predictions[0], skip_special_tokens=True)
            
            logger.info(f"   Example {i}: '{test_text}' → '{corrected.strip()}'")
        
        logger.info("✅ Model inference test successful!")
        
        # Check for training info
        training_info_path = local_path / "training_info.json"
        if training_info_path.exists():
            import json
            with open(training_info_path) as f:
                info = json.load(f)
            
            logger.info("📊 Training information found:")
            logger.info(f"   Training examples: {info.get('training_examples', 'N/A'):,}")
            logger.info(f"   Epochs: {info.get('epochs', 'N/A')}")
            logger.info(f"   Training time: {info.get('training_time_minutes', 'N/A')} minutes")
            logger.info(f"   Final loss: {info.get('final_train_loss', 'N/A')}")
        
        return str(local_path)
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise

def test_typo_correction(model_path: str, examples: list = None):
    """Test the downloaded model on typo correction examples."""
    
    if examples is None:
        examples = [
            "Thi sis a test sentenc with typos",
            "The quikc brown fox jumps over teh lazy dog",
            "I went too the stor to buy som milk",
            "Ther are many mistaks in this sentance",
            "Its a beutiful day outsid today"
        ]
    
    logger.info("🎯 Testing typo correction performance...")
    
    # Load model
    model = DistilBertForMaskedLM.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model.eval()
    
    corrections = []
    for i, text in enumerate(examples, 1):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=128,
                          padding="max_length", truncation=True)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Decode
        corrected = tokenizer.decode(predictions[0], skip_special_tokens=True)
        corrections.append((text, corrected.strip()))
        
        logger.info(f"Example {i}:")
        logger.info(f"  Input:  '{text}'")
        logger.info(f"  Output: '{corrected.strip()}'")
        logger.info("")
    
    return corrections

def main():
    parser = argparse.ArgumentParser(description="Download DistilBERT typo correction model from HF Hub")
    parser.add_argument('--hub_model_name', type=str, required=True,
                       help='Model name on HF Hub (username/model-name)')
    parser.add_argument('--local_path', type=str, default='models/downloaded_model',
                       help='Local path to save model')
    parser.add_argument('--token', type=str,
                       help='HF Hub token for private models')
    parser.add_argument('--force_download', action='store_true',
                       help='Force re-download even if model exists')
    parser.add_argument('--test_correction', action='store_true',
                       help='Test typo correction after download')
    parser.add_argument('--convert_to_ane', action='store_true',
                       help='Automatically convert to ANE format after download')
    
    args = parser.parse_args()
    
    try:
        # Download model
        model_path = download_model_from_hub(
            hub_model_name=args.hub_model_name,
            local_path=args.local_path,
            token=args.token,
            force_download=args.force_download
        )
        
        # Test typo correction if requested
        if args.test_correction:
            test_typo_correction(model_path)
        
        # Convert to ANE if requested
        if args.convert_to_ane:
            logger.info("🍎 Starting ANE conversion...")
            ane_output_path = f"{model_path}_ane"
            coreml_output_path = f"{model_path}_ANE.mlpackage"
            
            # Import and run ANE conversion
            import subprocess
            cmd = [
                "python", "src/apple_ane_conversion.py",
                "--input_model", model_path,
                "--ane_model_path", ane_output_path,  
                "--coreml_output", coreml_output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ ANE conversion completed successfully!")
                logger.info(f"📦 ANE Core ML model: {coreml_output_path}")
            else:
                logger.error(f"❌ ANE conversion failed: {result.stderr}")
        
        logger.info("\n🎉 Download completed successfully!")
        logger.info(f"📁 Model saved to: {model_path}")
        logger.info("\n📋 Next steps:")
        logger.info(f"   1. Test model: python src/validate.py --model_dir {model_path}")
        logger.info(f"   2. Convert to ANE: python src/apple_ane_conversion.py --input_model {model_path}")
        logger.info(f"   3. Benchmark ANE: python src/ane_vs_cpu_benchmark.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
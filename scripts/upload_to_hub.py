#!/usr/bin/env python3
"""
Upload trained model to HuggingFace Hub after training completes
"""

import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def upload_model():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--hub-model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--commit-message", default="Upload ByT5 typo fixer model", help="Commit message")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        return False
    
    print(f"🚀 Uploading model to HuggingFace Hub...")
    print(f"📁 Local path: {args.model_path}")
    print(f"🎯 Hub ID: {args.hub_model_id}")
    
    try:
        # Load model and tokenizer
        print("🔧 Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        
        # Upload to hub
        print("⬆️  Uploading tokenizer...")
        tokenizer.push_to_hub(args.hub_model_id, commit_message=args.commit_message)
        
        print("⬆️  Uploading model...")
        model.push_to_hub(args.hub_model_id, commit_message=args.commit_message)
        
        print(f"✅ Upload successful!")
        print(f"🔗 Model URL: https://huggingface.co/{args.hub_model_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"💡 Alternative: Use huggingface-cli upload {args.hub_model_id} {args.model_path}")
        return False

if __name__ == "__main__":
    upload_model()
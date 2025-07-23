#!/usr/bin/env python3
"""
Generate large dataset with better error handling and threading safety.
"""

import os
import sys
import subprocess

def main():
    print("🔄 Generating LARGE dataset (50K examples) with threading fixes...")
    
    # Set environment variables to prevent threading issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    # Remove old dataset
    if os.path.exists("data/enhanced_training_full.jsonl"):
        print(f"🗑️ Removing old dataset...")
        os.remove("data/enhanced_training_full.jsonl")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    try:
        # Run dataset generation with better error handling
        cmd = [
            "python3", "src/realistic_data_generation.py",
            "--output", "data/enhanced_training_large.jsonl",
            "--num_examples", "50000",
            "--corruption_rate", "0.15"
        ]
        
        print("📊 Running command:", " ".join(cmd))
        print("⏱️ This will take 10-15 minutes...")
        
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        
        # Validate output
        if os.path.exists("data/enhanced_training_large.jsonl"):
            with open("data/enhanced_training_large.jsonl", "r") as f:
                count = sum(1 for _ in f)
            
            print(f"✅ Dataset generated successfully!")
            print(f"📊 Total examples: {count:,}")
            
            if count < 20000:
                print(f"⚠️ Dataset smaller than expected. Minimum 20K needed for good training.")
                return False
            
            return True
        else:
            print("❌ Dataset file not created!")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset generation failed with error: {e}")
        print("🔧 Try running manually:")
        print("python3 src/realistic_data_generation.py --output data/enhanced_training_large.jsonl --num_examples 50000 --corruption_rate 0.15")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print("\n🎯 Next step:")
    print("python3 train_rtx5090.py --train_file data/enhanced_training_large.jsonl --output_dir models/qwen-typo-fixer-v2 --hf_repo mazhewitt/qwen-typo-fixer-v2")
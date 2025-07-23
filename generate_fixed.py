#!/usr/bin/env python3
"""
Run the original realistic_data_generation.py with proper threading isolation
to prevent the PyGILState_Release crash.
"""

import os
import subprocess
import sys

def main():
    print("🔧 Running original data generation with threading fixes...")
    
    # Set environment variables to prevent threading issues
    env = os.environ.copy()
    env.update({
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_OFFLINE": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_MAX_THREADS": "1"
    })
    
    # Remove corrupted file if it exists
    if os.path.exists("data/enhanced_training_large.jsonl"):
        print("🗑️ Removing corrupted dataset...")
        os.remove("data/enhanced_training_large.jsonl")
    
    # Run the original script with proper isolation
    cmd = [
        "python3", "src/realistic_data_generation.py",
        "--output", "data/enhanced_training_large.jsonl",
        "--num_examples", "40000",  # Slightly lower to avoid memory issues
        "--corruption_rate", "0.15"
    ]
    
    print("📊 Command:", " ".join(cmd))
    print("⏱️ This will take 10-15 minutes with real WikiText diversity...")
    print()
    
    try:
        # Run with isolated environment
        result = subprocess.run(cmd, env=env, check=True)
        
        # Check output
        if os.path.exists("data/enhanced_training_large.jsonl"):
            with open("data/enhanced_training_large.jsonl", "r") as f:
                count = sum(1 for _ in f)
            
            print(f"\n✅ SUCCESS! Generated {count:,} high-quality examples")
            print(f"🎯 Dataset ready for anti-overfitting training!")
            print(f"\n📋 Next command:")
            print(f"python3 train_rtx5090.py --train_file data/enhanced_training_large.jsonl --output_dir models/qwen-typo-fixer-v2 --hf_repo mazhewitt/qwen-typo-fixer-v2")
            return True
        else:
            print("❌ Dataset file not created!")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
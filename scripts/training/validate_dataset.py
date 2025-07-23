#!/usr/bin/env python3
"""
Quick dataset validation script to check dataset quality and size.
Run this after generating the large dataset to ensure it's ready for training.
"""

import json
import sys
from pathlib import Path

def validate_dataset(file_path):
    """Validate the generated dataset."""
    if not Path(file_path).exists():
        print(f"❌ Dataset file not found: {file_path}")
        return False
    
    print(f"📊 Validating dataset: {file_path}")
    
    examples = []
    corrupted_words = set()
    clean_words = set()
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                examples.append(data)
                
                # Collect vocabulary stats
                corrupted_words.update(data['corrupted'].lower().split())
                clean_words.update(data['clean'].lower().split())
                
                # Show first few examples
                if i < 3:
                    print(f"Example {i+1}:")
                    print(f"  Corrupted: {data['corrupted']}")
                    print(f"  Clean: {data['clean']}")
                    print()
                    
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON on line {i+1}")
                return False
    
    # Dataset statistics
    total_examples = len(examples)
    avg_corrupted_len = sum(len(ex['corrupted'].split()) for ex in examples) / total_examples
    avg_clean_len = sum(len(ex['clean'].split()) for ex in examples) / total_examples
    
    print(f"✅ Dataset Validation Results:")
    print(f"   Total examples: {total_examples:,}")
    print(f"   Average corrupted length: {avg_corrupted_len:.1f} words")
    print(f"   Average clean length: {avg_clean_len:.1f} words")
    print(f"   Corrupted vocabulary: {len(corrupted_words):,} unique words")
    print(f"   Clean vocabulary: {len(clean_words):,} unique words")
    
    # Quality checks
    if total_examples < 10000:
        print(f"⚠️ Dataset might be too small for proper training (recommended: 50K+)")
        return False
    
    if avg_corrupted_len > 20 or avg_clean_len > 20:
        print(f"⚠️ Sentences might be too long (recommended: <15 words)")
        return False
    
    print(f"✅ Dataset looks good for training!")
    return True

if __name__ == "__main__":
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "data/enhanced_training_large.jsonl"
    
    if validate_dataset(dataset_file):
        print(f"\n🚀 Ready to start training with: {dataset_file}")
        print(f"📋 Next command:")
        print(f"python3 train_rtx5090.py --train_file {dataset_file} --output_dir models/qwen-typo-fixer-v2 --hf_repo mazhewitt/qwen-typo-fixer-v2")
    else:
        print(f"\n❌ Dataset validation failed. Please regenerate the dataset.")
        sys.exit(1)
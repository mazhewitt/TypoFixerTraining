#!/usr/bin/env python3
"""
Master script to generate 100k training examples from scratch and create balanced dataset.
Optimized for ByT5 character-level training with high-quality data generation.

This script:
1. Generates 100k realistic typo examples from multiple sources
2. Creates balanced dataset (50/50 punctuation split)
3. Validates data quality
4. Prepares for training

Usage:
  python3 scripts/create_full_dataset_and_train.py
  python3 scripts/create_full_dataset_and_train.py --num-examples 150000 --corruption-rate 0.12
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command with proper error handling"""
    print(f"🔧 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("🔍 Checking dependencies...")
    
    required_files = [
        "src/realistic_data_generation.py",
        "t5/create_balanced_dataset.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file missing: {file_path}")
            return False
    
    # Check Python packages
    try:
        import datasets
        import transformers
        import tqdm
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing Python package: {e}")
        return False

def generate_training_data(num_examples: int, corruption_rate: float) -> bool:
    """Generate realistic training data using the fixed generator"""
    print(f"\n{'='*60}")
    print(f"📝 GENERATING {num_examples:,} TRAINING EXAMPLES")
    print(f"{'='*60}")
    
    # Use the fixed generator that avoids multiprocessing issues
    cmd = [
        "python3", "scripts/generate_100k_dataset_fixed.py",
        "--num-examples", str(num_examples),
        "--corruption-rate", str(corruption_rate)
    ]
    
    if not run_command(cmd, f"Generating {num_examples:,} examples (safe method)"):
        return False
    
    # Verify the output file exists and has content
    output_file = "data/enhanced_training_full.jsonl"
    if not os.path.exists(output_file):
        print(f"❌ Output file not created: {output_file}")
        return False
    
    # Count and validate examples
    examples_count = 0
    sample_example = None
    
    with open(output_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    example = json.loads(line.strip())
                    if sample_example is None:
                        sample_example = example
                    examples_count += 1
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON on line {line_num}")
    
    print(f"\n✅ Generated {examples_count:,} total examples")
    print(f"💾 Saved to: {output_file}")
    
    if sample_example:
        print(f"\n📝 Sample example:")
        print(f"   Corrupted: '{sample_example['corrupted']}'")
        print(f"   Clean: '{sample_example['clean']}'")
        print(f"   Complexity: {sample_example.get('complexity', 'unknown')}")
    
    return True

def create_balanced_dataset() -> bool:
    """Create balanced dataset with 50/50 punctuation split"""
    print(f"\n{'='*60}")
    print("⚖️  CREATING BALANCED DATASET")
    print(f"{'='*60}")
    
    cmd = ["python3", "t5/create_balanced_dataset.py"]
    return run_command(cmd, "Creating balanced punctuation dataset")

def validate_dataset() -> bool:
    """Validate the final dataset quality"""
    print(f"\n{'='*60}")
    print("🔍 VALIDATING DATASET QUALITY")
    print(f"{'='*60}")
    
    balanced_file = "data/enhanced_training_balanced.jsonl"
    
    if not os.path.exists(balanced_file):
        print(f"❌ Balanced dataset not found: {balanced_file}")
        return False
    
    examples = []
    with open(balanced_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError:
                print(f"⚠️  Invalid JSON on line {line_num}")
    
    if len(examples) < 10000:
        print(f"❌ Dataset too small: {len(examples)} examples (need at least 10,000)")
        return False
    
    # Quality checks
    with_punct = sum(1 for ex in examples if ex['clean'].endswith('.'))
    without_punct = len(examples) - with_punct
    
    has_corruption = sum(1 for ex in examples if ex['corrupted'] != ex['clean'])
    avg_length = sum(len(ex['clean']) for ex in examples) / len(examples)
    
    print(f"✅ Dataset validation passed!")
    print(f"   Total examples: {len(examples):,}")
    print(f"   With punctuation: {with_punct:,} ({with_punct/len(examples)*100:.1f}%)")
    print(f"   Without punctuation: {without_punct:,} ({without_punct/len(examples)*100:.1f}%)")
    print(f"   Examples with corruption: {has_corruption:,} ({has_corruption/len(examples)*100:.1f}%)")
    print(f"   Average length: {avg_length:.1f} characters")
    
    # Show sample examples
    print(f"\n📝 Sample Examples:")
    for i, ex in enumerate(examples[:3], 1):
        status = "✓" if ex['corrupted'] != ex['clean'] else "="
        print(f"   {i}. {status} '{ex['corrupted']}' → '{ex['clean']}'")
    
    return True

def cleanup_temp_files():
    """Clean up temporary files"""
    temp_patterns = ["data/temp_*.jsonl", "data/realistic_*.jsonl"]
    for pattern in temp_patterns:
        for file_path in Path(".").glob(pattern):
            try:
                file_path.unlink()
                print(f"🗑️  Cleaned up: {file_path}")
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="Generate complete ByT5 training dataset from scratch")
    parser.add_argument('--num-examples', type=int, default=100000,
                       help='Total number of examples to generate')
    parser.add_argument('--corruption-rate', type=float, default=0.15,
                       help='Rate of token corruption (0.0-1.0)')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip data generation (use existing enhanced_training_full.jsonl)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip final validation step')
    
    args = parser.parse_args()
    
    print("🚀 BYT5 COMPLETE DATASET CREATION")
    print("=" * 60)
    print(f"Target examples: {args.num_examples:,}")
    print(f"Corruption rate: {args.corruption_rate}")
    print(f"Expected training time: ~3-5 hours with this dataset")
    print()
    
    start_time = time.time()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        sys.exit(1)
    
    # Step 2: Generate training data (unless skipped)
    if not args.skip_generation:
        if not generate_training_data(args.num_examples, args.corruption_rate):
            print("❌ Data generation failed")
            sys.exit(1)
    else:
        print("⏭️  Skipping data generation (using existing data)")
    
    # Step 3: Create balanced dataset
    if not create_balanced_dataset():
        print("❌ Balanced dataset creation failed")
        sys.exit(1)
    
    # Step 4: Validate dataset (unless skipped)
    if not args.skip_validation:
        if not validate_dataset():
            print("❌ Dataset validation failed")
            sys.exit(1)
    else:
        print("⏭️  Skipping validation")
    
    # Step 5: Cleanup
    cleanup_temp_files()
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("🎉 DATASET CREATION COMPLETE!")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    print(f"📁 Output files:")
    print(f"   • data/enhanced_training_full.jsonl (raw dataset)")
    print(f"   • data/enhanced_training_balanced.jsonl (balanced for training)")
    print(f"   • balanced_dataset_info.json (metadata)")
    print()
    print("🎯 Next steps:")
    print("   1. Run: ./scripts/run_byt5_training_server.sh")
    print("   2. Model will auto-upload to HuggingFace when complete")
    print("   3. Test locally with: python3 scripts/test_byt5_on_server.py")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
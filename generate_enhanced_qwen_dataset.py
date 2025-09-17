#!/usr/bin/env python3
"""
Enhanced Qwen Training Dataset Generator

Combines the advanced T5 data generation improvements with Qwen-specific formatting:
- Advanced error pattern library with sophisticated corruptions
- Multi-domain source text diversification
- Punctuation balance (50/50 with/without ending punctuation)
- Complex multi-error sentence generation
- Contextual and phonetic error patterns
"""

import sys
import os
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent / "scripts"))
from advanced_training_data_generator import AdvancedTrainingDataGenerator, TrainingExample
from source_text_diversifier import SourceTextDiversifier
from error_pattern_library import AdvancedErrorPatterns

def create_punctuation_balanced_examples(examples: List[TrainingExample]) -> List[TrainingExample]:
    """Create balanced dataset with 50/50 punctuation split like T5 approach."""

    print("🔧 Creating punctuation-balanced dataset...")

    # Separate examples based on whether punctuation can be safely removed
    must_keep_punct = []  # Examples that must keep punctuation
    can_remove_punct = []  # Examples where punctuation can be removed

    for example in examples:
        clean_text = example.clean.strip()

        # Always keep punctuation for these cases
        if (clean_text.endswith('etc.') or
            clean_text.endswith('Mr.') or
            clean_text.endswith('Ms.') or
            clean_text.endswith('Dr.') or
            clean_text.endswith('Inc.') or
            clean_text.endswith('Ltd.') or
            '?' in clean_text or
            '!' in clean_text or
            clean_text.count('.') > 1):  # Multiple periods (abbreviations)
            must_keep_punct.append(example)
        else:
            can_remove_punct.append(example)

    print(f"📝 Must keep punctuation: {len(must_keep_punct):,}")
    print(f"📝 Can remove punctuation: {len(can_remove_punct):,}")

    # Create balanced dataset
    balanced_examples = []

    # Add all examples that must keep punctuation
    balanced_examples.extend(must_keep_punct)

    # For removable ones, create 50/50 split
    random.shuffle(can_remove_punct)
    half_point = len(can_remove_punct) // 2

    # First half: keep punctuation
    balanced_examples.extend(can_remove_punct[:half_point])

    # Second half: remove punctuation
    for example in can_remove_punct[half_point:]:
        # Create new example with punctuation removed
        new_corrupted = example.corrupted.rstrip('.')
        new_clean = example.clean.rstrip('.')

        # Create new training example
        new_example = TrainingExample(
            corrupted=new_corrupted,
            clean=new_clean,
            domain=example.domain,
            complexity=example.complexity,
            error_types=example.error_types,
            num_errors=example.num_errors,
            word_count=len(new_clean.split()),
            char_count=len(new_clean),
            difficulty_score=example.difficulty_score,
            source=example.source + "_no_punct"
        )
        balanced_examples.append(new_example)

    # Shuffle final dataset
    random.shuffle(balanced_examples)

    # Analyze results
    with_punct = sum(1 for ex in balanced_examples if ex.clean.endswith('.'))
    without_punct = len(balanced_examples) - with_punct

    print(f"📊 Balanced Results:")
    print(f"  Total examples: {len(balanced_examples):,}")
    print(f"  With punctuation: {with_punct:,} ({with_punct/len(balanced_examples)*100:.1f}%)")
    print(f"  Without punctuation: {without_punct:,} ({without_punct/len(balanced_examples)*100:.1f}%)")

    return balanced_examples

def format_for_qwen_training(examples: List[TrainingExample]) -> List[Dict]:
    """Format training examples for Qwen fine-tuning with instruction format."""

    print("🔄 Formatting examples for Qwen training...")

    formatted_examples = []

    for example in tqdm(examples, desc="Formatting"):
        # Use instruction-following format for Qwen
        formatted_example = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Correct the typos in this text: {example.corrupted}"
                },
                {
                    "role": "assistant",
                    "content": example.clean
                }
            ],
            # Include metadata for analysis
            "metadata": {
                "domain": example.domain,
                "complexity": example.complexity,
                "error_types": example.error_types,
                "num_errors": example.num_errors,
                "difficulty_score": example.difficulty_score,
                "source": example.source,
                "word_count": example.word_count,
                "char_count": example.char_count
            }
        }
        formatted_examples.append(formatted_example)

    return formatted_examples

def generate_enhanced_qwen_dataset(
    target_size: int = 100000,
    output_file: str = "data/enhanced_qwen_training.jsonl",
    source_sentences_file: str = "data/diverse_source_sentences.json",
    seed: int = 42
):
    """Generate enhanced training dataset for Qwen using T5 improvements."""

    print("🚀 Generating Enhanced Qwen Training Dataset")
    print("=" * 50)
    print(f"Target size: {target_size:,} examples")
    print(f"Output: {output_file}")
    print()

    # Set random seed for reproducibility
    random.seed(seed)

    # Step 1: Collect diverse source sentences
    print("📚 Step 1: Collecting diverse source sentences...")
    if not os.path.exists(source_sentences_file):
        print("  Collecting from datasets...")
        diversifier = SourceTextDiversifier()
        source_sentences = diversifier.collect_diverse_sentences(target_size // 2)
        diversifier.save_collected_sentences(source_sentences, source_sentences_file)
    else:
        print(f"  Loading from {source_sentences_file}...")
        with open(source_sentences_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        source_sentences = data.get("sentences_by_domain", {})

    print(f"✅ Loaded {sum(len(s) for s in source_sentences.values()):,} source sentences")
    print()

    # Step 2: Generate advanced training examples
    print("🔧 Step 2: Generating advanced training examples...")
    generator = AdvancedTrainingDataGenerator()
    examples = generator.generate_training_dataset(source_sentences, target_size)
    print()

    # Step 3: Create punctuation-balanced dataset
    print("⚖️ Step 3: Creating punctuation-balanced dataset...")
    balanced_examples = create_punctuation_balanced_examples(examples)
    print()

    # Step 4: Format for Qwen training
    print("📝 Step 4: Formatting for Qwen training...")
    formatted_examples = format_for_qwen_training(balanced_examples)
    print()

    # Step 5: Save dataset
    print("💾 Step 5: Saving dataset...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in formatted_examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Saved {len(formatted_examples):,} formatted examples to {output_file}")

    # Step 6: Generate statistics and metadata
    print("📊 Step 6: Generating statistics...")
    stats = generate_dataset_statistics(balanced_examples)

    # Save metadata
    metadata_file = output_file.replace('.jsonl', '_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"📊 Metadata saved to {metadata_file}")

    # Show final statistics
    print(f"\n📈 FINAL DATASET STATISTICS:")
    print(f"  Total Examples: {stats['total_examples']:,}")
    print(f"  Complexity Distribution:")
    for complexity, count in stats['complexity_distribution'].items():
        print(f"    {complexity:8}: {count:6,} ({count/stats['total_examples']:5.1%})")

    print(f"  Error Type Distribution:")
    for error_type, count in list(stats['error_type_distribution'].items())[:8]:
        print(f"    {error_type:12}: {count:6,}")

    print(f"  Average Difficulty: {stats['difficulty_stats']['mean']:.1f}")
    print(f"  Average Errors/Example: {stats['error_count_stats']['mean']:.1f}")

    # Show sample examples
    print(f"\n📝 Sample Examples:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line.strip())
            user_msg = data['messages'][0]['content']
            assistant_msg = data['messages'][1]['content']
            metadata = data['metadata']
            print(f"\n{i+1}. Domain: {metadata['domain']}, Complexity: {metadata['complexity']}")
            print(f"   Input:  '{user_msg}'")
            print(f"   Output: '{assistant_msg}'")

    print(f"\n✅ Enhanced Qwen training dataset generation complete!")
    return output_file, stats

def generate_dataset_statistics(examples: List[TrainingExample]) -> Dict:
    """Generate comprehensive statistics for the dataset."""
    from collections import Counter

    stats = {
        "total_examples": len(examples),
        "complexity_distribution": dict(Counter(ex.complexity for ex in examples)),
        "domain_distribution": dict(Counter(ex.domain for ex in examples)),
        "error_type_distribution": Counter(),
        "source_distribution": dict(Counter(ex.source for ex in examples)),
        "difficulty_stats": {
            "mean": sum(ex.difficulty_score for ex in examples) / len(examples),
            "min": min(ex.difficulty_score for ex in examples),
            "max": max(ex.difficulty_score for ex in examples),
        },
        "word_count_stats": {
            "mean": sum(ex.word_count for ex in examples) / len(examples),
            "min": min(ex.word_count for ex in examples),
            "max": max(ex.word_count for ex in examples),
        },
        "error_count_stats": {
            "mean": sum(ex.num_errors for ex in examples) / len(examples),
            "min": min(ex.num_errors for ex in examples),
            "max": max(ex.num_errors for ex in examples),
        }
    }

    # Count error types (examples can have multiple types)
    for example in examples:
        for error_type in example.error_types:
            stats["error_type_distribution"][error_type] += 1

    # Convert Counter to dict for JSON serialization
    stats["error_type_distribution"] = dict(stats["error_type_distribution"])

    return stats

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate enhanced Qwen training dataset with T5 improvements")
    parser.add_argument('--target-size', type=int, default=100000,
                       help='Target number of training examples')
    parser.add_argument('--output-file', default='data/enhanced_qwen_training.jsonl',
                       help='Output training file')
    parser.add_argument('--source-file', default='data/diverse_source_sentences.json',
                       help='Source sentences file (will be generated if missing)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Generate the enhanced dataset
    output_file, stats = generate_enhanced_qwen_dataset(
        target_size=args.target_size,
        output_file=args.output_file,
        source_sentences_file=args.source_file,
        seed=args.seed
    )

    print(f"\n🎉 SUCCESS!")
    print(f"📁 Enhanced Qwen training dataset: {output_file}")
    print(f"📊 Total examples: {stats['total_examples']:,}")
    print(f"📈 Ready for Qwen fine-tuning!")

if __name__ == "__main__":
    main()
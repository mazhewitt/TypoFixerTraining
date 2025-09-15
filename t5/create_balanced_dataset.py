#!/usr/bin/env python3
"""
Create balanced dataset with 50% examples ending with punctuation and 50% without.
This will make the model more robust to different input formats.
"""

import json
import random
import re
from pathlib import Path

def should_keep_punctuation(text):
    """Determine if punctuation should be kept based on content."""
    text = text.strip()
    
    # Always keep punctuation for these cases
    if (text.endswith('etc.') or 
        text.endswith('Mr.') or 
        text.endswith('Ms.') or
        text.endswith('Dr.') or
        text.endswith('Inc.') or
        text.endswith('Ltd.') or
        re.search(r'\b[A-Z]\.$', text) or  # Single letter abbreviations
        '?' in text or 
        '!' in text or
        text.count('.') > 1):  # Multiple periods (likely abbreviations)
        return True
    
    return False

def create_balanced_dataset():
    """Create balanced dataset with/without punctuation."""
    
    print("🔧 Creating balanced punctuation dataset...")
    
    original_examples = []
    
    # Load all examples
    with open('data/enhanced_training_full.jsonl', 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                original_examples.append(data)
            except json.JSONDecodeError:
                print(f"Skipping invalid line {line_num}")
                continue
    
    print(f"📊 Loaded {len(original_examples):,} original examples")
    
    # Separate examples into categories
    must_keep_punct = []  # Examples that must keep punctuation
    can_remove_punct = []  # Examples where punctuation can be removed
    
    for example in original_examples:
        clean_text = example['clean'].strip()
        
        if should_keep_punctuation(clean_text):
            must_keep_punct.append(example)
        else:
            can_remove_punct.append(example)
    
    print(f"📝 Must keep punctuation: {len(must_keep_punct):,}")
    print(f"📝 Can remove punctuation: {len(can_remove_punct):,}")
    
    # Create balanced dataset
    balanced_examples = []
    
    # Add all examples that must keep punctuation
    for example in must_keep_punct:
        balanced_examples.append(example.copy())
    
    # For the removable ones, create 50/50 split
    random.shuffle(can_remove_punct)
    
    half_point = len(can_remove_punct) // 2
    
    # First half: keep punctuation
    for example in can_remove_punct[:half_point]:
        balanced_examples.append(example.copy())
    
    # Second half: remove punctuation
    for example in can_remove_punct[half_point:]:
        new_example = example.copy()
        
        # Remove ending punctuation from both corrupted and clean
        if new_example['corrupted'].endswith('.'):
            new_example['corrupted'] = new_example['corrupted'][:-1]
        if new_example['clean'].endswith('.'):
            new_example['clean'] = new_example['clean'][:-1]
        
        # Update character counts
        new_example['char_count'] = len(new_example['clean'])
        
        # Add metadata
        new_example['punctuation_removed'] = True
        
        balanced_examples.append(new_example)
    
    # Shuffle the final dataset
    random.shuffle(balanced_examples)
    
    # Analyze the results
    with_punct = sum(1 for ex in balanced_examples if ex['clean'].endswith('.'))
    without_punct = len(balanced_examples) - with_punct
    
    print(f"\n📈 Balanced Dataset Results:")
    print(f"  Total examples: {len(balanced_examples):,}")
    print(f"  With punctuation: {with_punct:,} ({with_punct/len(balanced_examples)*100:.1f}%)")
    print(f"  Without punctuation: {without_punct:,} ({without_punct/len(balanced_examples)*100:.1f}%)")
    
    # Save the balanced dataset
    output_file = 'data/enhanced_training_balanced.jsonl'
    with open(output_file, 'w') as f:
        for example in balanced_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n💾 Balanced dataset saved to: {output_file}")
    
    # Show some examples
    print(f"\n📝 Sample Examples:")
    print("With punctuation:")
    with_punct_examples = [ex for ex in balanced_examples if ex['clean'].endswith('.')][:3]
    for i, ex in enumerate(with_punct_examples, 1):
        print(f"  {i}. '{ex['corrupted']}' → '{ex['clean']}'")
    
    print("Without punctuation:")
    without_punct_examples = [ex for ex in balanced_examples if not ex['clean'].endswith('.')][:3]
    for i, ex in enumerate(without_punct_examples, 1):
        print(f"  {i}. '{ex['corrupted']}' → '{ex['clean']}'")
    
    return {
        'total_examples': len(balanced_examples),
        'with_punctuation': with_punct,
        'without_punctuation': without_punct,
        'output_file': output_file
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    results = create_balanced_dataset()
    
    # Save metadata
    with open('balanced_dataset_info.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Balanced dataset creation complete!")
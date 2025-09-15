#!/usr/bin/env python3
"""
Analyze punctuation patterns in the training data.
"""

import json
import re
from collections import Counter

def analyze_punctuation_patterns():
    """Analyze punctuation in the training data."""
    
    print("📊 Analyzing punctuation patterns in training data...")
    
    corrupted_endings = Counter()
    clean_endings = Counter()
    
    has_period_corrupted = 0
    has_period_clean = 0
    total_examples = 0
    
    ending_punctuation = set(['.', '!', '?', ',', ';', ':'])
    
    with open('data/enhanced_training_full.jsonl', 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                corrupted = data['corrupted'].strip()
                clean = data['clean'].strip()
                
                # Check last character
                corrupted_last = corrupted[-1] if corrupted else ''
                clean_last = clean[-1] if clean else ''
                
                corrupted_endings[corrupted_last] += 1
                clean_endings[clean_last] += 1
                
                # Count periods specifically
                if corrupted.endswith('.'):
                    has_period_corrupted += 1
                if clean.endswith('.'):
                    has_period_clean += 1
                
                total_examples += 1
                
                # Show first few examples
                if line_num <= 10:
                    print(f"Example {line_num}:")
                    print(f"  Corrupted: '{corrupted}' (ends with '{corrupted_last}')")
                    print(f"  Clean: '{clean}' (ends with '{clean_last}')")
                    print()
                
            except json.JSONDecodeError:
                print(f"Skipping invalid line {line_num}")
                continue
    
    print(f"📈 Analysis Results ({total_examples:,} examples):")
    print(f"{'='*50}")
    
    print(f"\n🔤 Corrupted Text Endings:")
    for char, count in corrupted_endings.most_common(10):
        pct = count / total_examples * 100
        char_display = repr(char) if char in [' ', '\t', '\n'] else char
        print(f"  {char_display}: {count:,} ({pct:.1f}%)")
    
    print(f"\n🔤 Clean Text Endings:")
    for char, count in clean_endings.most_common(10):
        pct = count / total_examples * 100
        char_display = repr(char) if char in [' ', '\t', '\n'] else char
        print(f"  {char_display}: {count:,} ({pct:.1f}%)")
    
    print(f"\n📊 Period Analysis:")
    print(f"  Corrupted ending with period: {has_period_corrupted:,} ({has_period_corrupted/total_examples*100:.1f}%)")
    print(f"  Clean ending with period: {has_period_clean:,} ({has_period_clean/total_examples*100:.1f}%)")
    
    # Check for examples that could be made non-punctuated
    candidates_for_no_punct = 0
    with open('data/enhanced_training_full.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            corrupted = data['corrupted'].strip()
            clean = data['clean'].strip()
            
            # Count examples that end with period and could work without
            if (clean.endswith('.') and 
                not clean.endswith('etc.') and 
                not clean.endswith('Mr.') and
                not clean.endswith('Dr.') and
                not re.search(r'\b[A-Z]\.$', clean)):  # Not abbreviations
                candidates_for_no_punct += 1
    
    print(f"\n💡 Recommendation:")
    print(f"  Examples that could work without punctuation: {candidates_for_no_punct:,}")
    print(f"  Suggested approach: Create 50/50 split (with/without ending punctuation)")
    
    return {
        'total_examples': total_examples,
        'period_corrupted': has_period_corrupted,
        'period_clean': has_period_clean,
        'candidates_no_punct': candidates_for_no_punct,
        'corrupted_endings': dict(corrupted_endings.most_common()),
        'clean_endings': dict(clean_endings.most_common())
    }

if __name__ == "__main__":
    results = analyze_punctuation_patterns()
    
    with open('punctuation_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Analysis saved to punctuation_analysis.json")
#!/usr/bin/env python3
"""
Serial dataset generation - no threading, no parallel processing.
Designed to reliably generate 50K examples without crashes.
"""

import json
import random
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Disable all parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Simple sentences for generation (no external datasets to avoid threading)
SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "I believe this is the correct answer.",
    "She received her degree last year.",
    "The restaurant serves excellent food.",
    "He is studying for his final examination.",
    "We discussed the important details.",
    "The beginning of the story was exciting.",
    "I definitely need to improve my skills.",
    "The experience was challenging and rewarding.",
    "This document contains several errors.",
    "Please check your spelling carefully.",
    "The weather forecast predicts rain tomorrow.",
    "Students must complete their assignments on time.",
    "Technology has changed our daily lives significantly.",
    "The company announced new product developments.",
    "Environmental protection requires immediate attention.",
    "Communication skills are essential for success.",
    "Scientific research advances human knowledge.",
    "Cultural diversity enriches our communities.",
    "Educational opportunities should be accessible to everyone.",
    "Healthcare systems need continuous improvement.",
    "Transportation infrastructure requires regular maintenance.",
    "Economic policies affect global markets.",
    "Social media influences public opinion.",
    "Artificial intelligence transforms various industries.",
    "Climate change poses serious challenges.",
    "International cooperation promotes peace.",
    "Innovation drives technological progress.",
    "Democracy depends on citizen participation.",
    "Human rights must be protected universally.",
]

# Expand the sentence pool by creating variations
def create_sentence_variations():
    """Create variations of base sentences to increase diversity."""
    variations = []
    
    # Add original sentences
    variations.extend(SAMPLE_SENTENCES)
    
    # Create simple variations
    subjects = ["The team", "Our group", "This person", "Each student", "The manager"]
    verbs = ["completed", "finished", "started", "reviewed", "analyzed"]
    objects = ["the project", "their work", "the assignment", "the report", "the task"]
    
    for subject in subjects:
        for verb in verbs:
            for obj in objects:
                variations.append(f"{subject} {verb} {obj} successfully.")
    
    # Add more varied sentences
    templates = [
        "The {adj} {noun} {verb} {adverb}.",
        "{Name} {verb} {noun} {prep} {location}.",
        "We {verb} {adj} {noun} {time}.",
        "This {noun} {verb} {adverb} {prep} {location}.",
    ]
    
    adjectives = ["beautiful", "important", "difficult", "interesting", "successful"]
    nouns = ["project", "meeting", "document", "presentation", "system"]
    verbs = ["completed", "organized", "reviewed", "developed", "implemented"]
    adverbs = ["carefully", "quickly", "efficiently", "thoroughly", "successfully"]
    prepositions = ["in", "at", "for", "with", "during"]
    locations = ["the office", "the conference", "the building", "the facility", "the center"]
    names = ["Sarah", "Michael", "Jennifer", "David", "Lisa"]
    times = ["yesterday", "last week", "this morning", "recently", "today"]
    
    # Generate template-based sentences
    for template in templates:
        for i in range(20):  # Generate 20 per template
            sentence = template.format(
                adj=random.choice(adjectives),
                noun=random.choice(nouns),
                verb=random.choice(verbs),
                adverb=random.choice(adverbs),
                prep=random.choice(prepositions),
                location=random.choice(locations),
                Name=random.choice(names),
                time=random.choice(times)
            )
            variations.append(sentence)
    
    return variations

# Keyboard layout for typos
KEYBOARD_NEIGHBORS = {
    'q': ['w', 'a'], 'w': ['q', 'e', 'a', 's'], 'e': ['w', 'r', 's', 'd'],
    'r': ['e', 't', 'd', 'f'], 't': ['r', 'y', 'f', 'g'], 'y': ['t', 'u', 'g', 'h'],
    'u': ['y', 'i', 'h', 'j'], 'i': ['u', 'o', 'j', 'k'], 'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'], 'a': ['q', 'w', 's'], 's': ['w', 'e', 'a', 'd'],
    'd': ['e', 'r', 's', 'f'], 'f': ['r', 't', 'd', 'g'], 'g': ['t', 'y', 'f', 'h'],
    'h': ['y', 'u', 'g', 'j'], 'j': ['u', 'i', 'h', 'k'], 'k': ['i', 'o', 'j', 'l'],
    'l': ['o', 'p', 'k'], 'z': ['x'], 'x': ['z', 'c'], 'c': ['x', 'v'],
    'v': ['c', 'b'], 'b': ['v', 'n'], 'n': ['b', 'm'], 'm': ['n']
}

# Common typos
COMMON_TYPOS = {
    'their': 'there', 'there': 'their', 'believe': 'beleive',
    'receive': 'recieve', 'definitely': 'definately', 'separate': 'seperate',
    'restaurant': 'resturant', 'excellent': 'excelent', 'beginning': 'begining',
    'experience': 'experiance', 'challenging': 'chalenging', 'rewarding': 'rewardng',
    'studying': 'studyng', 'examination': 'examintion', 'discussed': 'dicussed',
    'important': 'importnt', 'exciting': 'excting', 'improve': 'imporve',
    'skills': 'skils', 'answer': 'answr', 'correct': 'correct', 'degree': 'degre'
}

def corrupt_word(word):
    """Apply corruption to a single word."""
    if not word or len(word) < 2:
        return word
    
    # 40% chance of no corruption
    if random.random() < 0.4:
        return word
    
    word_lower = word.lower().strip('.,!?;:"\'')
    
    # Use common typos if available
    if word_lower in COMMON_TYPOS and random.random() < 0.7:
        corrupted = COMMON_TYPOS[word_lower]
        # Preserve capitalization
        if word[0].isupper():
            corrupted = corrupted.capitalize()
        return word.replace(word_lower, corrupted)
    
    # Apply keyboard typos
    chars = list(word)
    if random.random() < 0.3 and len(chars) > 2:
        pos = random.randint(0, len(chars) - 1)
        char = chars[pos].lower()
        if char in KEYBOARD_NEIGHBORS:
            chars[pos] = random.choice(KEYBOARD_NEIGHBORS[char])
    
    # Character deletion (10% chance)
    if random.random() < 0.1 and len(chars) > 3:
        pos = random.randint(1, len(chars) - 2)
        chars.pop(pos)
    
    # Character duplication (10% chance)
    if random.random() < 0.1 and len(chars) < 15:
        pos = random.randint(0, len(chars) - 1)
        chars.insert(pos, chars[pos])
    
    return ''.join(chars)

def corrupt_sentence(sentence, corruption_rate=0.15):
    """Apply corruption to a sentence."""
    words = sentence.split()
    corrupted_words = []
    
    for word in words:
        if random.random() < corruption_rate:
            corrupted_words.append(corrupt_word(word))
        else:
            corrupted_words.append(word)
    
    return ' '.join(corrupted_words)

def generate_dataset_serial(output_file, num_examples=50000, corruption_rate=0.15):
    """Generate dataset serially - no threading."""
    print(f"🔄 Generating {num_examples:,} examples serially (no threading)...")
    print(f"📁 Output: {output_file}")
    print(f"⚠️ This will take 15-20 minutes but will be reliable!")
    
    # Create sentence pool
    print("📝 Creating sentence variations...")
    sentences = create_sentence_variations()
    print(f"✅ Created {len(sentences):,} sentence variations")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    start_time = time.time()
    generated = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_examples):
            # Progress reporting every 5000 examples
            if i % 5000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (num_examples - i) / rate / 60 if rate > 0 else 0
                print(f"📊 Progress: {i:,}/{num_examples:,} ({i/num_examples*100:.1f}%) - ETA: {eta:.1f} min")
            
            # Pick random sentence
            sentence = random.choice(sentences)
            
            # Apply corruption
            corrupted = corrupt_sentence(sentence, corruption_rate)
            
            # Only include if corruption occurred
            if corrupted != sentence:
                data = {
                    "corrupted": corrupted,
                    "clean": sentence,
                    "complexity": "generated",
                    "word_count": len(sentence.split()),
                    "char_count": len(sentence),
                    "source": "serial_generation"
                }
                
                f.write(json.dumps(data) + '\n')
                generated += 1
    
    elapsed = time.time() - start_time
    print(f"✅ Generated {generated:,} examples in {elapsed/60:.1f} minutes")
    print(f"💾 Saved to: {output_file}")
    
    return generated

def main():
    output_file = "data/enhanced_training_large.jsonl"
    
    print("🚀 Serial Dataset Generation (No Threading)")
    print("=" * 50)
    
    # Remove existing file
    if os.path.exists(output_file):
        print(f"🗑️ Removing existing file...")
        os.remove(output_file)
    
    try:
        generated = generate_dataset_serial(
            output_file=output_file,
            num_examples=50000,
            corruption_rate=0.15
        )
        
        if generated >= 20000:
            print(f"\n✅ SUCCESS! Generated {generated:,} examples")
            print(f"🎯 Ready for training - this should prevent overfitting!")
            print(f"\n📋 Next command:")
            print(f"python3 train_rtx5090.py --train_file {output_file} --output_dir models/qwen-typo-fixer-v2 --hf_repo mazhewitt/qwen-typo-fixer-v2")
            return True
        else:
            print(f"⚠️ Only generated {generated:,} examples - may not be enough")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Improved data generation for typo correction with targeted fixes.
Implements recommendations for 90%+ accuracy:
- Higher corruption rates (25% vs 15%)
- Character-level noise (adjacent swaps)  
- Missing/extra space corruptions
- Identity examples (no corruption)
- Better keyboard layouts
"""

import argparse
import json
import random
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedTypoGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Enhanced keyboard layouts for international users
        self.qwerty_neighbors = {
            'q': ['w', 'a'], 'w': ['q', 'e', 'a', 's'], 'e': ['w', 'r', 's', 'd'],
            'r': ['e', 't', 'd', 'f'], 't': ['r', 'y', 'f', 'g'], 'y': ['t', 'u', 'g', 'h'],
            'u': ['y', 'i', 'h', 'j'], 'i': ['u', 'o', 'j', 'k'], 'o': ['i', 'p', 'k', 'l'],
            'p': ['o', 'l'], 'a': ['q', 'w', 's', 'z'], 's': ['a', 'w', 'e', 'd', 'z', 'x'],
            'd': ['s', 'e', 'r', 'f', 'x', 'c'], 'f': ['d', 'r', 't', 'g', 'c', 'v'],
            'g': ['f', 't', 'y', 'h', 'v', 'b'], 'h': ['g', 'y', 'u', 'j', 'b', 'n'],
            'j': ['h', 'u', 'i', 'k', 'n', 'm'], 'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p', 'm'], 'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'],
            'c': ['x', 'd', 'f', 'v'], 'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'],
            'n': ['b', 'h', 'j', 'm'], 'm': ['n', 'j', 'k', 'l']
        }
        
        # AZERTY layout for European users
        self.azerty_neighbors = {
            'a': ['z', 'q'], 'z': ['a', 'e', 'q', 's'], 'e': ['z', 'r', 's', 'd'],
            'r': ['e', 't', 'd', 'f'], 't': ['r', 'y', 'f', 'g'], 'y': ['t', 'u', 'g', 'h'],
            # Add more AZERTY mappings as needed
        }
        
        # Common typo patterns with their corrections
        self.common_typos = {
            'teh': 'the', 'hte': 'the', 'taht': 'that', 'thier': 'their',
            'recieve': 'receive', 'beleive': 'believe', 'seperate': 'separate',
            'definately': 'definitely', 'occured': 'occurred', 'begining': 'beginning',
            'completly': 'completely', 'usualy': 'usually', 'realy': 'really',
            'woudl': 'would', 'coudl': 'could', 'shoudl': 'should',
            'youre': "you're", 'its': "it's", 'dont': "don't", 'cant': "can't",
            'wont': "won't", 'isnt': "isn't", 'wasnt': "wasn't", 'hasnt': "hasn't"
        }
        
        # Word endings that commonly lose letters
        self.truncation_patterns = [
            ('ing', ''), ('ed', ''), ('ly', ''), ('er', ''), ('est', ''),
            ('tion', 'tion'), ('sion', 'ion'), ('ness', 'nes'), ('ment', 'ent'),
            ('able', 'abl'), ('ible', 'ibl'), ('ful', 'ul'), ('less', 'les')
        ]
        
    def corrupt_text(self, text: str, corruption_level: str = "mixed") -> str:
        """Apply various corruption types based on level."""
        words = text.split()
        corrupted_words = []
        
        # Corruption rates by level
        if corruption_level == "easy":
            char_corruption_rate = 0.05  # 5%
            word_corruption_rate = 0.10  # 10%
            space_corruption_rate = 0.02  # 2%
        elif corruption_level == "mixed":
            char_corruption_rate = 0.15  # 15%
            word_corruption_rate = 0.25  # 25% - increased from 15%
            space_corruption_rate = 0.05  # 5%
        else:  # hard
            char_corruption_rate = 0.25  # 25%
            word_corruption_rate = 0.35  # 35%
            space_corruption_rate = 0.10  # 10%
        
        for i, word in enumerate(words):
            corrupted_word = word
            
            # Apply word-level corruptions
            if random.random() < word_corruption_rate:
                corrupted_word = self._corrupt_word(word)
            
            # Apply character-level corruptions (new addition)
            if random.random() < char_corruption_rate:
                corrupted_word = self._corrupt_characters(corrupted_word)
            
            corrupted_words.append(corrupted_word)
        
        # Join words back
        result = ' '.join(corrupted_words)
        
        # Apply space corruptions (missing/extra spaces)
        if random.random() < space_corruption_rate:
            result = self._corrupt_spaces(result)
            
        return result
    
    def _corrupt_word(self, word: str) -> str:
        """Apply word-level corruptions."""
        if len(word) <= 2:
            return word
            
        # Check for common typos first
        word_lower = word.lower()
        if word_lower in self.common_typos:
            # Sometimes apply the common typo
            if random.random() < 0.7:  # 70% chance
                correction = self.common_typos[word_lower]
                # Preserve original case
                if word[0].isupper():
                    correction = correction.capitalize()
                return correction
        
        corruption_type = random.choice([
            'keyboard_neighbor',  # 40%
            'character_swap',     # 20% 
            'missing_letter',     # 20%
            'extra_letter',       # 10%
            'truncation'          # 10%
        ])
        
        if corruption_type == 'keyboard_neighbor':
            return self._keyboard_neighbor_error(word)
        elif corruption_type == 'character_swap':
            return self._swap_adjacent_chars(word)
        elif corruption_type == 'missing_letter':
            return self._remove_random_char(word)
        elif corruption_type == 'extra_letter':
            return self._add_random_char(word)
        elif corruption_type == 'truncation':
            return self._truncate_word(word)
            
        return word
    
    def _corrupt_characters(self, word: str) -> str:
        """Apply character-level noise (1-2% of characters)."""
        if len(word) <= 3:
            return word
            
        chars = list(word)
        num_swaps = max(1, int(len(word) * 0.02))  # 2% of characters
        
        for _ in range(num_swaps):
            if len(chars) >= 2:
                # Swap adjacent characters
                pos = random.randint(0, len(chars) - 2)
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                
        return ''.join(chars)
    
    def _corrupt_spaces(self, text: str) -> str:
        """Add missing or extra spaces."""
        if random.random() < 0.5:  # Missing space
            # Remove a space (join two words)
            words = text.split()
            if len(words) >= 2:
                pos = random.randint(0, len(words) - 2)
                words[pos] = words[pos] + words[pos + 1]
                words.pop(pos + 1)
                return ' '.join(words)
        else:  # Extra space
            # Add space within a word
            words = text.split()
            if words:
                word_idx = random.randint(0, len(words) - 1)
                word = words[word_idx]
                if len(word) > 3:
                    pos = random.randint(1, len(word) - 2)
                    words[word_idx] = word[:pos] + ' ' + word[pos:]
                return ' '.join(words)
        
        return text
    
    def _keyboard_neighbor_error(self, word: str) -> str:
        """Replace a character with keyboard neighbor."""
        if len(word) <= 1:
            return word
            
        pos = random.randint(0, len(word) - 1)
        char = word[pos].lower()
        
        # Choose keyboard layout
        layout = self.qwerty_neighbors
        if random.random() < 0.1:  # 10% AZERTY for international
            layout = self.azerty_neighbors
            
        if char in layout and layout[char]:
            new_char = random.choice(layout[char])
            # Preserve case
            if word[pos].isupper():
                new_char = new_char.upper()
            return word[:pos] + new_char + word[pos+1:]
            
        return word
    
    def _swap_adjacent_chars(self, word: str) -> str:
        """Swap two adjacent characters."""
        if len(word) <= 2:
            return word
            
        pos = random.randint(0, len(word) - 2)
        chars = list(word)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return ''.join(chars)
    
    def _remove_random_char(self, word: str) -> str:
        """Remove a random character."""
        if len(word) <= 2:
            return word
            
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos+1:]
    
    def _add_random_char(self, word: str) -> str:
        """Add a random character."""
        pos = random.randint(0, len(word))
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:pos] + char + word[pos:]
    
    def _truncate_word(self, word: str) -> str:
        """Remove common word endings."""
        for ending, replacement in self.truncation_patterns:
            if word.lower().endswith(ending) and len(word) > len(ending) + 2:
                if random.random() < 0.3:  # 30% chance
                    return word[:-len(ending)] + replacement
        return word

def generate_curriculum_data(generator: ImprovedTypoGenerator, 
                           texts: List[str], 
                           num_examples: int,
                           corruption_level: str = "mixed",
                           identity_rate: float = 0.15) -> List[Dict]:
    """Generate data with curriculum learning and identity examples."""
    
    examples = []
    texts_cycle = iter(texts * ((num_examples // len(texts)) + 1))
    
    logger.info(f"Generating {num_examples} examples (level: {corruption_level}, identity rate: {identity_rate:.1%})")
    
    for i in tqdm(range(num_examples), desc=f"Generating {corruption_level} examples"):
        try:
            clean_text = next(texts_cycle).strip()
            
            # Skip very short or long texts
            if len(clean_text) < 10 or len(clean_text) > 500:
                continue
                
            # Identity examples (no corruption) - key addition!
            if random.random() < identity_rate:
                corrupted_text = clean_text  # No corruption
            else:
                corrupted_text = generator.corrupt_text(clean_text, corruption_level)
            
            examples.append({
                'clean': clean_text,
                'corrupted': corrupted_text,
                'level': corruption_level
            })
            
        except StopIteration:
            break
    
    return examples

def main():
    parser = argparse.ArgumentParser(description="Improved typo correction data generation")
    parser.add_argument('--num_examples', type=int, default=50000,
                       help='Total number of examples to generate')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL file path')
    parser.add_argument('--dataset', type=str, default='wikitext',
                       choices=['wikitext', 'openwebtext', 'news'],
                       help='Source dataset')
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning (easy→mixed→hard)')
    parser.add_argument('--identity_rate', type=float, default=0.15,
                       help='Proportion of identity examples (no corruption)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generator = ImprovedTypoGenerator(args.seed)
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    if args.dataset == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 20]
    elif args.dataset == 'news':
        # Use news dataset for shorter, cleaner sentences
        dataset = load_dataset('ag_news', split='train')
        texts = [item['text'] for item in dataset]
    else:
        dataset = load_dataset('openwebtext', split='train', streaming=True)
        texts = [item['text'] for item in dataset.take(100000)]
    
    logger.info(f"Loaded {len(texts):,} source texts")
    
    # Generate examples
    all_examples = []
    
    if args.curriculum:
        # Curriculum learning: easy → mixed → hard
        easy_examples = generate_curriculum_data(
            generator, texts, args.num_examples // 4, "easy", args.identity_rate * 0.5
        )
        mixed_examples = generate_curriculum_data(
            generator, texts, args.num_examples // 2, "mixed", args.identity_rate
        )
        hard_examples = generate_curriculum_data(
            generator, texts, args.num_examples // 4, "hard", args.identity_rate * 2
        )
        all_examples = easy_examples + mixed_examples + hard_examples
    else:
        # Standard mixed-level generation
        all_examples = generate_curriculum_data(
            generator, texts, args.num_examples, "mixed", args.identity_rate
        )
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Save to file
    logger.info(f"Saving {len(all_examples):,} examples to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for example in all_examples:
            json.dump({
                'clean': example['clean'],
                'corrupted': example['corrupted']
            }, f)
            f.write('\n')
    
    # Calculate statistics
    identity_count = sum(1 for ex in all_examples if ex['clean'] == ex['corrupted'])
    corruption_count = len(all_examples) - identity_count
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    
    logger.info("✅ Improved data generation completed!")
    logger.info(f"📊 Generated: {len(all_examples):,} examples")
    logger.info(f"📊 Identity examples: {identity_count:,} ({identity_count/len(all_examples)*100:.1f}%)")
    logger.info(f"📊 Corrupted examples: {corruption_count:,} ({corruption_count/len(all_examples)*100:.1f}%)")
    logger.info(f"📁 Saved to: {args.output}")
    logger.info(f"💾 File size: {file_size:.1f} MB")
    
    # Show sample examples
    logger.info("\n📝 Sample examples:")
    for i, example in enumerate(all_examples[:5], 1):
        logger.info(f"  {i}. '{example['corrupted']}' → '{example['clean']}'")
    
    logger.info(f"\n🚀 Recommended training command:")
    logger.info(f"   python src/train.py \\")
    logger.info(f"     --train_file {args.output} \\")
    logger.info(f"     --output_dir models/improved_typo_fixer \\")
    logger.info(f"     --num_train_epochs 6 \\")
    logger.info(f"     --per_device_train_batch_size 64 \\")
    logger.info(f"     --weight_decay 0.03 \\")
    logger.info(f"     --max_seq_len 96")

if __name__ == "__main__":
    main()
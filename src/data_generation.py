#!/usr/bin/env python3
"""
Synthetic typo generation for training DistilBERT typo correction model.
Applies keyboard-neighbor swaps, drops, doubles, transpositions, and space splits.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re
from datasets import load_dataset
from tqdm import tqdm

# QWERTY keyboard layout for neighbor-based typos
KEYBOARD_NEIGHBORS = {
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

# Common homophone and contextual word confusions
WORD_CONFUSIONS = {
    # their/there/they're
    'their': ['there', 'they\'re'],
    'there': ['their', 'they\'re'],
    'they\'re': ['their', 'there'],
    'theyre': ['their', 'there'],  # common contraction typo
    
    # your/you're
    'your': ['you\'re'],
    'you\'re': ['your'],
    'youre': ['your'],  # common contraction typo
    
    # its/it's
    'its': ['it\'s'],
    'it\'s': ['its'],
    'its\'': ['its'],  # over-apostrophe
    
    # hear/here
    'hear': ['here'],
    'here': ['hear'],
    
    # to/too/two
    'to': ['too', 'two'],
    'too': ['to', 'two'],
    'two': ['to', 'too'],
    
    # then/than
    'then': ['than'],
    'than': ['then'],
    
    # accept/except
    'accept': ['except'],
    'except': ['accept'],
    
    # affect/effect
    'affect': ['effect'],
    'effect': ['affect'],
    
    # loose/lose
    'loose': ['lose'],
    'lose': ['loose'],
    
    # brake/break
    'brake': ['break'],
    'break': ['brake'],
    
    # desert/dessert
    'desert': ['dessert'],
    'dessert': ['desert'],
    
    # principal/principle
    'principal': ['principle'],
    'principle': ['principal'],
    
    # complement/compliment
    'complement': ['compliment'],
    'compliment': ['complement'],
    
    # breath/breathe
    'breath': ['breathe'],
    'breathe': ['breath'],
    
    # advice/advise
    'advice': ['advise'],
    'advise': ['advice'],
}

def keyboard_neighbor_swap(word: str) -> str:
    """Replace a random character with a keyboard neighbor."""
    if len(word) < 2:
        return word
    
    chars = list(word.lower())
    idx = random.randint(0, len(chars) - 1)
    char = chars[idx]
    
    if char in KEYBOARD_NEIGHBORS and KEYBOARD_NEIGHBORS[char]:
        chars[idx] = random.choice(KEYBOARD_NEIGHBORS[char])
    
    return ''.join(chars)

def character_drop(word: str) -> str:
    """Drop a random character from the word."""
    if len(word) <= 1:
        return word
    
    chars = list(word)
    idx = random.randint(0, len(chars) - 1)
    chars.pop(idx)
    
    return ''.join(chars)

def character_double(word: str) -> str:
    """Double a random character in the word."""
    if len(word) < 1:
        return word
    
    chars = list(word)
    idx = random.randint(0, len(chars) - 1)
    chars.insert(idx, chars[idx])
    
    return ''.join(chars)

def character_transpose(word: str) -> str:
    """Swap two adjacent characters."""
    if len(word) < 2:
        return word
    
    chars = list(word)
    idx = random.randint(0, len(chars) - 2)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    
    return ''.join(chars)

def word_space_split(word: str) -> str:
    """Split word with a space at random position."""
    if len(word) <= 2:
        return word
    
    split_pos = random.randint(1, len(word) - 1)
    return word[:split_pos] + ' ' + word[split_pos:]

def word_confusion(word: str) -> str:
    """Replace word with commonly confused alternative."""
    word_lower = word.lower()
    
    if word_lower in WORD_CONFUSIONS:
        confused_word = random.choice(WORD_CONFUSIONS[word_lower])
        
        # Preserve original capitalization
        if word[0].isupper():
            confused_word = confused_word[0].upper() + confused_word[1:]
        if word.isupper():
            confused_word = confused_word.upper()
            
        return confused_word
    
    return word

def corrupt_sentence(sentence: str, corruption_rate: float = 0.15) -> str:
    """Apply random corruptions to a sentence."""
    words = sentence.split()
    corrupted_words = []
    
    # Character-level corruption functions
    char_corruption_functions = [
        keyboard_neighbor_swap,
        character_drop, 
        character_double,
        character_transpose,
        word_space_split
    ]
    
    for word in words:
        # Skip very short words and punctuation for character corruptions
        if len(word) <= 1 or not re.match(r'^[a-zA-Z\']+$', word):
            corrupted_words.append(word)
            continue
            
        if random.random() < corruption_rate:
            # 30% chance of word confusion (homophone/contextual errors)
            # 70% chance of character-level corruption
            if random.random() < 0.3:
                corrupted_word = word_confusion(word)
            else:
                corruption_func = random.choice(char_corruption_functions)
                corrupted_word = corruption_func(word)
            
            corrupted_words.append(corrupted_word)
        else:
            corrupted_words.append(word)
    
    return ' '.join(corrupted_words)

def load_english_text_data(dataset_split: str = "train[:2%]", max_sentences: int = 2_000_000):
    """Load sentences from available English text datasets."""
    print(f"Loading English text dataset: {dataset_split}")
    
    # Try different datasets that are commonly available
    dataset_options = [
        ("wikitext", "wikitext-2-raw-v1"),
        ("bookcorpus", None),
        ("imdb", None),
    ]
    
    dataset = None
    dataset_name = None
    
    for name, config in dataset_options:
        try:
            print(f"Trying dataset: {name}")
            if config:
                dataset = load_dataset(name, config, split=dataset_split)
            else:
                dataset = load_dataset(name, split=dataset_split)
            dataset_name = name
            print(f"Successfully loaded {name}")
            break
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
    
    if dataset is None:
        print("All datasets failed, creating sample sentences...")
        # Fallback to sample sentences
        return generate_sample_sentences(max_sentences)
    
    sentences = []
    print(f"Processing {dataset_name} entries...")
    
    for entry in tqdm(dataset):
        # Extract text based on dataset structure
        text = None
        if 'text' in entry:
            text = entry['text']
        elif 'sentence' in entry:
            text = entry['sentence']
        elif dataset_name == 'imdb' and 'review' in entry:
            text = entry['review']
        
        if text:
            # Split into sentences (simple approach)
            import re
            sentence_endings = re.split(r'[.!?]+', text)
            
            for sentence in sentence_endings:
                sentence = sentence.strip()
                
                # Filter sentences: reasonable length, contains letters, not too many special chars
                if (10 <= len(sentence) <= 200 and 
                    re.search(r'[a-zA-Z]', sentence) and 
                    len(re.findall(r'[^\w\s\'\-\.,\?!]', sentence)) < 5):  # Limit special characters
                    
                    # Clean up text
                    sentence = re.sub(r'\s+', ' ', sentence).strip()  # Normalize whitespace
                    
                    if len(sentence) >= 10:  # Recheck length after cleaning
                        sentences.append(sentence)
                        
                        if len(sentences) >= max_sentences:
                            break
        
        if len(sentences) >= max_sentences:
            break
    
    print(f"Collected {len(sentences)} clean sentences from {dataset_name}")
    return sentences

def generate_sample_sentences(max_sentences: int = 1000):
    """Generate sample sentences as fallback when no datasets are available."""
    print("Generating sample sentences as fallback...")
    
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "I went to the store to buy some milk and bread.",
        "The weather is really nice today, isn't it?",
        "Can you help me with this problem?",
        "They're going to their house over there.",
        "Your presentation was excellent, you're very talented.",
        "It's important to know its purpose before starting.",
        "I hear you went over here yesterday.",
        "We need to accept this proposal, except for the budget part.",
        "The effect of this change will affect everyone.",
        "I need some advice about how to advise my team.",
        "Take a deep breath before you breathe out slowly.",
        "The brake system needs to break less frequently.",
        "This compliment will complement our strategy perfectly.",
        "The desert sand was perfect for our dessert picnic.",
        "Our principal shared an important principle with us.",
        "There are too many people going to the two stores.",
        "Then we realized this was better than the alternative.",
        "I loose my keys more often than I lose my patience.",
    ] * (max_sentences // 20 + 1)  # Repeat to get enough sentences
    
    return sample_sentences[:max_sentences]

def process_sentences_from_dataset(output_file: Path, dataset_split: str = "train[:2%]", 
                                 max_sentences: int = 2_000_000, corruption_rate: float = 0.15):
    """Load English text data and generate corrupted/clean sentence pairs."""
    
    # Load sentences from available datasets
    sentences = load_english_text_data(dataset_split, max_sentences)
    
    if not sentences:
        print("No sentences loaded from dataset!")
        return
    
    print(f"Generating corrupted pairs with {corruption_rate:.1%} corruption rate...")
    
    # Generate corrupted pairs and write to JSONL
    pairs_written = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in tqdm(sentences, desc="Corrupting sentences"):
            corrupted = corrupt_sentence(sentence, corruption_rate)
            
            # Only include if corruption actually occurred
            if corrupted != sentence:
                data = {
                    "corrupted": corrupted,
                    "clean": sentence
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                pairs_written += 1
    
    print(f"Generated {pairs_written} corrupted/clean pairs")

def process_sentences_from_files(input_files: List[Path], output_file: Path, 
                               max_sentences: int = 2_000_000, corruption_rate: float = 0.15):
    """Process input text files and generate corrupted/clean sentence pairs."""
    
    sentences = []
    
    # Read sentences from input files
    for input_file in input_files:
        print(f"Processing {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                
                # Filter: reasonable length, contains letters
                if 10 <= len(sentence) <= 200 and re.search(r'[a-zA-Z]', sentence):
                    sentences.append(sentence)
                    
                if len(sentences) >= max_sentences:
                    break
        
        if len(sentences) >= max_sentences:
            break
    
    print(f"Collected {len(sentences)} sentences")
    
    # Generate corrupted pairs and write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            corrupted = corrupt_sentence(sentence, corruption_rate)
            
            # Only include if corruption actually occurred
            if corrupted != sentence:
                data = {
                    "corrupted": corrupted,
                    "clean": sentence
                }
                f.write(json.dumps(data) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic typo training data")
    parser.add_argument('--source', type=str, choices=['opensubtitles', 'files'], 
                       default='opensubtitles', help='Data source: opensubtitles dataset or local files')
    parser.add_argument('--input', type=str, 
                       help='Input directory containing text files (for --source files)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL file path')
    parser.add_argument('--dataset_split', type=str, default='train[:2%]',
                       help='Dataset split to use for OpenSubtitles (e.g., train[:2%], train[:100000])')
    parser.add_argument('--max_sentences', type=int, default=2_000_000,
                       help='Maximum number of sentences to process')
    parser.add_argument('--corruption_rate', type=float, default=0.15,
                       help='Fraction of words to corrupt')
    parser.add_argument('--validation_split', type=float, default=0.002,
                       help='Fraction of data to hold out for validation')
    
    args = parser.parse_args()
    
    output_file = Path(args.output)
    
    if args.source == 'opensubtitles':
        # Use available English text dataset
        print(f"Using English text dataset: {args.dataset_split}")
        process_sentences_from_dataset(
            output_file, 
            args.dataset_split, 
            args.max_sentences, 
            args.corruption_rate
        )
    
    elif args.source == 'files':
        # Use local text files
        if not args.input:
            print("Error: --input required when using --source files")
            return
        
        input_dir = Path(args.input)
        text_files = list(input_dir.glob('*.txt')) + list(input_dir.glob('*.tsv'))
        
        if not text_files:
            print(f"No .txt or .tsv files found in {input_dir}")
            return
        
        print(f"Found {len(text_files)} input files")
        process_sentences_from_files(text_files, output_file, args.max_sentences, args.corruption_rate)
    
    print(f"Training data written to {output_file}")
    
    # Generate validation split if requested
    if args.validation_split > 0:
        validation_file = output_file.parent / f"validation_{output_file.name}"
        print(f"Creating validation split: {validation_file}")
        
        # Simple approach: take last N% of generated pairs for validation
        create_validation_split(output_file, validation_file, args.validation_split)

def create_validation_split(input_file: Path, validation_file: Path, split_ratio: float):
    """Create validation split from training data."""
    print(f"Creating validation split ({split_ratio:.1%})...")
    
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    # Calculate split point
    total_lines = len(all_lines)
    val_size = int(total_lines * split_ratio)
    train_size = total_lines - val_size
    
    print(f"Total examples: {total_lines:,}")
    print(f"Training: {train_size:,}, Validation: {val_size:,}")
    
    # Write validation split
    with open(validation_file, 'w', encoding='utf-8') as f:
        f.writelines(all_lines[-val_size:])
    
    # Update training file (remove validation examples)
    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(all_lines[:train_size])
    
    print(f"Validation data written to {validation_file}")
    print(f"Training data updated: {input_file}")

if __name__ == "__main__":
    main()
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

def word_joining(words: List[str], word_idx: int) -> List[str]:
    """Join last letter of current word to next word (e.g., 'it works' -> 'i tworks')."""
    if word_idx >= len(words) - 1:  # Can't join if this is the last word
        return words
    
    current_word = words[word_idx]
    next_word = words[word_idx + 1]
    
    # Skip if either word is too short or contains punctuation
    if (len(current_word) <= 1 or len(next_word) < 1 or 
        not re.match(r'^[a-zA-Z\']+$', current_word) or 
        not re.match(r'^[a-zA-Z\']+$', next_word)):
        return words
    
    # Take last letter from current word and prepend to next word
    modified_words = words.copy()
    modified_words[word_idx] = current_word[:-1]  # Remove last letter
    modified_words[word_idx + 1] = current_word[-1] + next_word  # Add it to next word
    
    return modified_words

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
    corrupted_words = words.copy()
    
    # Track which words have been modified
    modified_indices = set()
    
    # First pass: apply word-level corruptions that affect multiple words
    for i in range(len(words)):
        if i in modified_indices:
            continue
            
        if random.random() < corruption_rate:
            # 30% chance of word confusion (homophone/contextual errors)
            # 35% chance of word joining (half of what used to be keyboard neighbor)
            # 35% chance of character-level corruption
            rand_val = random.random()
            
            if rand_val < 0.3:
                # Word confusion
                if len(words[i]) > 1 and re.match(r'^[a-zA-Z\']+$', words[i]):
                    corrupted_words[i] = word_confusion(words[i])
                    modified_indices.add(i)
            elif rand_val < 0.65:
                # Word joining (replaces half of keyboard neighbor swaps)
                if i < len(words) - 1:  # Can only join if not the last word
                    result_words = word_joining(corrupted_words, i)
                    if result_words != corrupted_words:  # Only if joining actually happened
                        corrupted_words = result_words
                        modified_indices.add(i)
                        modified_indices.add(i + 1)
            else:
                # Character-level corruption (remaining keyboard neighbors + others)
                if len(words[i]) > 1 and re.match(r'^[a-zA-Z\']+$', words[i]):
                    char_corruption_functions = [
                        keyboard_neighbor_swap,
                        character_drop, 
                        character_double,
                        character_transpose,
                        word_space_split
                    ]
                    corruption_func = random.choice(char_corruption_functions)
                    corrupted_words[i] = corruption_func(words[i])
                    modified_indices.add(i)
    
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
    parser = argparse.ArgumentParser(description="Generate synthetic typo data for training")
    parser.add_argument('--output', type=str, default='data/processed/train.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--num_examples', type=int, default=1000,
                       help='Number of examples to generate (default: 1000, use 100000+ for production)')
    parser.add_argument('--corruption_rate', type=float, default=0.15,
                       help='Rate of token corruption (0.0-1.0)')
    parser.add_argument('--dataset', type=str, default='wikitext',
                       choices=['wikitext', 'opensubtitles'], 
                       help='Source dataset to use')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sentence length in characters')
    parser.add_argument('--min_length', type=int, default=10,
                       help='Minimum sentence length in characters')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Generating synthetic typo data...")
    print(f"📁 Output file: {args.output}")
    print(f"📊 Target examples: {args.num_examples:,}")
    print(f"🔀 Corruption rate: {args.corruption_rate}")
    print(f"📚 Source dataset: {args.dataset}")
    
    try:
        # Load dataset based on user choice
        if args.dataset == 'wikitext':
            print("Loading WikiText dataset...")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            text_key = 'text'
        else:
            print("Loading OpenSubtitles dataset...")
            try:
                dataset = load_dataset("open_subtitles", lang1="en", split="train", streaming=True)
                text_key = 'translation'  # OpenSubtitles uses this key
            except Exception as e:
                print(f"⚠️ OpenSubtitles failed ({e}), falling back to WikiText...")
                dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
                text_key = 'text'
        
        generated_count = 0
        processed_count = 0
        skipped_count = 0
        
        # Progress tracking for large datasets
        progress_interval = max(1, args.num_examples // 100)  # Update every 1%
        
        with open(args.output, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc="Processing examples", unit="docs"):
                if generated_count >= args.num_examples:
                    break
                
                processed_count += 1
                
                # Extract text based on dataset format
                if text_key == 'translation':
                    # OpenSubtitles format
                    if 'en' in example[text_key]:
                        text = example[text_key]['en'].strip()
                    else:
                        skipped_count += 1
                        continue
                else:
                    # WikiText format
                    text = example[text_key].strip()
                
                # Skip empty lines, very short lines, or headers
                if len(text) < args.min_length or text.startswith('=') or not text.strip():
                    skipped_count += 1
                    continue
                
                # Process sentences (split on periods, exclamation marks, question marks)
                sentences = re.split(r'[.!?]+', text)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    
                    # Skip short or empty sentences
                    if len(sentence) < args.min_length or len(sentence) > args.max_length:
                        continue
                    
                    # Require minimum number of words
                    if len(sentence.split()) < 3:
                        continue
                    
                    # Skip sentences with too many special characters
                    if len(re.findall(r'[^a-zA-Z0-9\s\'\-]', sentence)) > len(sentence) * 0.1:
                        continue
                    
                    # Generate corrupted version
                    corrupted = corrupt_sentence(sentence, args.corruption_rate)
                    
                    # Skip if no corruption occurred
                    if corrupted == sentence:
                        continue
                    
                    # Create training example
                    example_data = {
                        'corrupted': corrupted,
                        'clean': sentence
                    }
                    
                    f.write(json.dumps(example_data, ensure_ascii=False) + '\n')
                    generated_count += 1
                    
                    # Progress reporting for large datasets
                    if generated_count % progress_interval == 0:
                        print(f"  Generated: {generated_count:,}/{args.num_examples:,} examples "
                              f"({100*generated_count/args.num_examples:.1f}%)")
                    
                    if generated_count >= args.num_examples:
                        break
                
                # Early termination for testing with small datasets
                if args.num_examples <= 10000 and processed_count > args.num_examples * 5:
                    print(f"⚠️ Processed {processed_count:,} documents, stopping early to avoid infinite loop")
                    break
        
        print(f"\n✅ Data generation completed!")
        print(f"📊 Generated: {generated_count:,} training examples")
        print(f"📚 Processed: {processed_count:,} documents")
        print(f"⏭️ Skipped: {skipped_count:,} documents")
        print(f"📁 Saved to: {args.output}")
        print(f"💾 File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Show sample examples
        print(f"\n📝 Sample examples:")
        with open(args.output, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Show first 5 examples
                    break
                data = json.loads(line.strip())
                print(f"  {i+1}. '{data['corrupted']}' → '{data['clean']}'")
                
        # Provide usage recommendations
        if generated_count < args.num_examples:
            print(f"\n⚠️ Warning: Only generated {generated_count:,} examples (requested {args.num_examples:,})")
            print(f"💡 Consider using a different dataset or increasing --max_length")
        
        if args.num_examples >= 50000:
            print(f"\n🚀 Production-scale dataset created! Recommended training:")
            print(f"   • Batch size: 64-128")
            print(f"   • Epochs: 3")
            print(f"   • Learning rate: 2e-5")
            print(f"   • Expected training time: 2-4 hours on RTX 4090")
    
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        raise

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
#!/usr/bin/env python3
"""
Fixed data generation script that avoids multiprocessing/GIL issues.
Generates 100k realistic typo examples using a safer approach.
"""

import json
import random
import re
import os
from pathlib import Path
from typing import List, Dict, Set
from tqdm import tqdm

# Avoid multiprocessing issues
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import datasets with error handling
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  datasets library not available, using fallback sentences")

# Corruption functions
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

COMMON_TYPOS = {
    'the': ['teh', 'hte'], 'and': ['adn', 'nad'], 'that': ['taht', 'htat'],
    'have': ['ahve', 'haev'], 'for': ['fro', 'ofr'], 'not': ['nto', 'ont'],
    'with': ['wtih', 'whit'], 'you': ['yuo', 'oyu'], 'this': ['tihs', 'htis'],
    'but': ['btu', 'ubt'], 'can': ['acn', 'cna'], 'had': ['ahd', 'hda'],
    'are': ['aer', 'rae'], 'what': ['waht', 'hwat'], 'your': ['yuor', 'yoru'],
    'when': ['wehn', 'hwen'], 'said': ['siad', 'asid'], 'there': ['tehre', 'htere'],
    'each': ['eahc', 'aech'], 'which': ['whcih', 'hwich'], 'will': ['wlil', 'iwll'],
    'about': ['abuot', 'baout'], 'could': ['coudl', 'ocudl'], 'time': ['tiem', 'itme'],
    'believe': ['beleive', 'belive'], 'receive': ['recieve', 'recive'],
    'definitely': ['definately', 'definatly'], 'separate': ['seperate', 'seprate'],
    'necessary': ['neccessary', 'necesary'], 'different': ['diferent', 'diffrent']
}

def keyboard_neighbor_swap(word: str) -> str:
    """Replace a character with keyboard neighbor."""
    if len(word) < 2:
        return word
    pos = random.randint(0, len(word) - 1)
    char = word[pos].lower()
    if char in KEYBOARD_NEIGHBORS and random.random() < 0.3:
        replacement = random.choice(KEYBOARD_NEIGHBORS[char])
        return word[:pos] + replacement + word[pos + 1:]
    return word

def character_drop(word: str) -> str:
    """Drop a character from word."""
    if len(word) <= 3:
        return word
    pos = random.randint(1, len(word) - 2)
    return word[:pos] + word[pos + 1:]

def character_swap(word: str) -> str:
    """Swap adjacent characters."""
    if len(word) < 3:
        return word
    pos = random.randint(0, len(word) - 2)
    chars = list(word)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return ''.join(chars)

def corrupt_word(word: str) -> str:
    """Apply realistic corruption to a word."""
    # Check common typos first
    lower_word = word.lower()
    if lower_word in COMMON_TYPOS and random.random() < 0.4:
        typo = random.choice(COMMON_TYPOS[lower_word])
        return typo if word.islower() else typo.capitalize()
    
    # Apply random corruption
    corruption_type = random.choice(['keyboard', 'drop', 'swap'])
    
    if corruption_type == 'keyboard':
        return keyboard_neighbor_swap(word)
    elif corruption_type == 'drop':
        return character_drop(word)
    else:
        return character_swap(word)

def corrupt_sentence(sentence: str, corruption_rate: float = 0.15) -> str:
    """Apply realistic corruption to a sentence."""
    words = sentence.split()
    corrupted_words = []
    
    for word in words:
        # Extract punctuation
        punct = ''
        clean_word = word
        if word and word[-1] in '.,!?;:':
            punct = word[-1]
            clean_word = word[:-1]
        
        # Corrupt word based on rate
        if (len(clean_word) > 2 and 
            random.random() < corruption_rate and
            not clean_word.isnumeric()):
            clean_word = corrupt_word(clean_word)
        
        corrupted_words.append(clean_word + punct)
    
    return ' '.join(corrupted_words)

def get_fallback_sentences() -> List[str]:
    """High-quality fallback sentences when datasets fail."""
    return [
        "The weather is beautiful today.",
        "I need to check my email before the meeting.",
        "She received an important message yesterday.",
        "The restaurant serves excellent food every evening.",
        "We should discuss this matter more carefully.",
        "The government announced new policies last week.",
        "Students are studying hard for their final examinations.",
        "The library has many interesting books available.",
        "Please send me the document as soon as possible.",
        "The meeting was successful and very informative.",
        "I believe this is the correct answer to the question.",
        "The quick brown fox jumps over the lazy dog.",
        "Technology has changed the way we communicate.",
        "The conference will be held next month in Chicago.",
        "She works as a software engineer at a tech company.",
        "The project deadline has been extended until Friday.",
        "We need to separate these items more carefully.",
        "The software update fixed many bugs and improved performance.",
        "According to the studies, this method is very effective.",
        "The team worked together to achieve their common goals.",
        "Education plays a crucial role in personal development.",
        "The hospital provides excellent medical care to patients.",
        "Climate change is one of the most pressing issues today.",
        "The concert was attended by thousands of music fans.",
        "She graduated from university with honors last year.",
        "The company plans to expand its operations internationally.",
        "Regular exercise is important for maintaining good health.",
        "The museum displays artifacts from ancient civilizations.",
        "Scientists are working on breakthrough medical treatments.",
        "The new policy will affect all employees starting next month.",
        "Online shopping has become increasingly popular recently.",
        "The research team published their findings in a scientific journal.",
        "Public transportation helps reduce traffic congestion in cities.",
        "The documentary explores the history of space exploration.",
        "Advanced technology enables more efficient business processes.",
        "Environmental protection requires cooperation from all countries.",
        "The university offers various programs for international students.",
        "Social media platforms have transformed how people share information.",
        "The construction project is expected to be completed by December.",
        "Renewable energy sources are becoming more cost-effective.",
        "The book contains valuable insights about leadership and management.",
        "Medical professionals recommend regular health checkups for everyone.",
        "The festival celebrates cultural diversity through music and dance.",
        "Financial planning is essential for achieving long-term goals.",
        "The laboratory conducts important research on genetic diseases.",
        "International trade agreements benefit multiple countries economically.",
        "The workshop provides practical training for professional development.",
        "Archaeological discoveries reveal fascinating details about ancient cultures.",
        "Digital transformation has revolutionized traditional business models.",
        "The scholarship program supports talented students from diverse backgrounds.",
        "Manufacturing companies are adopting sustainable production methods."
    ] * 20  # Repeat to get more variety through corruption

def load_sentences_safely(dataset_name: str, max_sentences: int = 50000) -> List[str]:
    """Safely load sentences from dataset with fallback."""
    print(f"📚 Loading sentences from {dataset_name}...")
    
    if not DATASETS_AVAILABLE:
        print("⚠️  Using fallback sentences (datasets library not available)")
        return get_fallback_sentences()
    
    try:
        # Load dataset with minimal processing to avoid multiprocessing issues
        if dataset_name == 'wikitext':
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=False)
            text_key = 'text'
        else:  # bookcorpus
            try:
                dataset = load_dataset('bookcorpus', split='train[:10000]', streaming=False)
                text_key = 'text'
            except:
                print("⚠️  BookCorpus not available, using WikiText only")
                dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=False)
                text_key = 'text'
        
        sentences = []
        seen_sentences = set()
        
        print(f"🔄 Extracting sentences from {len(dataset)} documents...")
        
        for i, example in enumerate(tqdm(dataset, desc="Processing", disable=True)):
            if len(sentences) >= max_sentences:
                break
            
            text = example[text_key].strip()
            if not text or len(text) < 20:
                continue
            
            # Simple sentence splitting
            potential_sentences = re.split(r'[.!?]+\s+', text)
            
            for raw_sentence in potential_sentences:
                if len(sentences) >= max_sentences:
                    break
                
                sentence = raw_sentence.strip()
                if sentence and not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                
                # Quality filters
                if (20 <= len(sentence) <= 200 and
                    5 <= len(sentence.split()) <= 25 and
                    sentence.count(' ') >= 4 and
                    not sentence.startswith(('==', '--', '*', '#')) and
                    sentence.lower() not in seen_sentences):
                    
                    seen_sentences.add(sentence.lower())
                    sentences.append(sentence)
        
        print(f"✅ Extracted {len(sentences)} sentences from {dataset_name}")
        return sentences
        
    except Exception as e:
        print(f"⚠️  Error loading {dataset_name}: {e}")
        print("Using fallback sentences...")
        return get_fallback_sentences()

def generate_dataset(num_examples: int, corruption_rate: float = 0.15) -> None:
    """Generate the complete dataset."""
    print(f"🚀 GENERATING {num_examples:,} TRAINING EXAMPLES")
    print(f"📊 Corruption rate: {corruption_rate}")
    print("=" * 60)
    
    # Load sentences from multiple sources
    all_sentences = []
    
    # Get WikiText sentences (60% of total)
    wiki_sentences = load_sentences_safely('wikitext', max_sentences=30000)
    all_sentences.extend(wiki_sentences)
    
    # Get BookCorpus sentences (40% of total)  
    book_sentences = load_sentences_safely('bookcorpus', max_sentences=20000)
    all_sentences.extend(book_sentences)
    
    # Add fallback sentences to ensure we have enough
    fallback_sentences = get_fallback_sentences()
    all_sentences.extend(fallback_sentences)
    
    # Remove duplicates and shuffle
    unique_sentences = list(set(all_sentences))
    random.shuffle(unique_sentences)
    
    print(f"📝 Total unique sentences available: {len(unique_sentences):,}")
    
    # Generate training examples
    os.makedirs("data", exist_ok=True)
    output_file = "data/enhanced_training_full.jsonl"
    
    generated = 0
    max_attempts = num_examples * 3  # Try up to 3x to account for rejections
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for attempt in tqdm(range(max_attempts), desc="Generating examples"):
            if generated >= num_examples:
                break
            
            # Pick random sentence
            sentence = random.choice(unique_sentences)
            
            # Apply corruption
            corrupted = corrupt_sentence(sentence, corruption_rate)
            
            # Only include if corruption actually occurred
            if corrupted != sentence:
                complexity = (
                    'simple' if len(sentence.split()) <= 8 else
                    'complex' if len(sentence.split()) >= 15 else
                    'medium'
                )
                
                data = {
                    "corrupted": corrupted,
                    "clean": sentence,
                    "complexity": complexity,
                    "word_count": len(sentence.split()),
                    "char_count": len(sentence),
                    "source": "mixed_safe"
                }
                
                f.write(json.dumps(data) + '\n')
                generated += 1
    
    print(f"\n✅ Generated {generated:,} examples")
    print(f"💾 Saved to: {output_file}")
    
    # Quick quality check
    if generated > 0:
        with open(output_file, 'r') as f:
            sample = json.loads(f.readline())
        print(f"📝 Sample: '{sample['corrupted']}' → '{sample['clean']}'")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 100k dataset safely")
    parser.add_argument('--num-examples', type=int, default=100000,
                       help='Number of examples to generate')
    parser.add_argument('--corruption-rate', type=float, default=0.15,
                       help='Rate of corruption (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    generate_dataset(args.num_examples, args.corruption_rate)

if __name__ == "__main__":
    main()
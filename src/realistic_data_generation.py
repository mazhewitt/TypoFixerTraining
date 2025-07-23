#!/usr/bin/env python3
"""
Realistic single-sentence typo generation for Qwen training.
Extracts natural sentences from datasets and applies realistic corruptions.
Focuses on authentic, real-world sentence patterns rather than artificial concatenations.
"""

import json
import random
import argparse
import re
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datasets import load_dataset
from tqdm import tqdm

# Corruption functions and data
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

WORD_CONFUSIONS = {
    'their': 'there', 'there': 'their', 'they\'re': 'their',
    'your': 'you\'re', 'you\'re': 'your', 'its': 'it\'s', 'it\'s': 'its',
    'too': 'to', 'to': 'too', 'than': 'then', 'then': 'than',
    'affect': 'effect', 'effect': 'affect', 'lose': 'loose', 'loose': 'lose'
}

# Cache for external datasets
_norvig_misspellings = None
_holbrook_pairs = None

def load_norvig_misspellings() -> Dict[str, str]:
    """Load Norvig's 20k misspellings from Peter Norvig's spelling corrector."""
    global _norvig_misspellings
    if _norvig_misspellings is not None:
        return _norvig_misspellings
    
    print("📥 Loading Norvig's 20k misspellings...")
    _norvig_misspellings = {}
    
    try:
        # Download from Peter Norvig's website
        url = "https://norvig.com/big.txt"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract misspelling patterns from text
        text = response.text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        
        # Create common misspelling patterns based on frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top frequent words to create realistic misspellings
        frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:1000]
        
        for word, _ in frequent_words:
            if len(word) >= 4:  # Focus on longer words
                # Generate common misspelling patterns
                misspellings = generate_word_misspellings(word)
                for misspelled in misspellings:
                    _norvig_misspellings[misspelled] = word
                    
    except Exception as e:
        print(f"⚠️ Could not load Norvig data ({e}), using built-in patterns")
        # Fallback to common patterns
        _norvig_misspellings = {
            'accomodate': 'accommodate', 'occured': 'occurred', 'recieve': 'receive',
            'seperate': 'separate', 'definately': 'definitely', 'neccessary': 'necessary',
            'begining': 'beginning', 'comming': 'coming', 'untill': 'until',
            'sucessful': 'successful', 'occassion': 'occasion', 'priviledge': 'privilege',
            'embarass': 'embarrass', 'millionaire': 'millionaire', 'dissapear': 'disappear',
            'tommorrow': 'tomorrow', 'reccomend': 'recommend', 'fourty': 'forty',
            'wierd': 'weird', 'freind': 'friend', 'beleive': 'believe',
            'acheive': 'achieve', 'peice': 'piece', 'recieved': 'received'
        }
    
    print(f"✅ Loaded {len(_norvig_misspellings)} Norvig misspelling patterns")
    return _norvig_misspellings

def generate_word_misspellings(word: str) -> List[str]:
    """Generate realistic misspellings for a word using multiple techniques."""
    misspellings = []
    
    # Double letters
    for i in range(len(word) - 1):
        if word[i] == word[i + 1]:  # Already doubled
            # Remove one
            misspelled = word[:i] + word[i + 1:]
            misspellings.append(misspelled)
        else:
            # Double a consonant
            if word[i] not in 'aeiou':
                misspelled = word[:i + 1] + word[i] + word[i + 1:]
                misspellings.append(misspelled)
    
    # ie/ei swaps
    if 'ie' in word:
        misspellings.append(word.replace('ie', 'ei'))
    if 'ei' in word:
        misspellings.append(word.replace('ei', 'ie'))
    
    # Drop silent letters
    if word.endswith('e') and len(word) > 3:
        misspellings.append(word[:-1])
    
    return misspellings[:3]  # Limit to avoid explosion

def load_holbrook_pairs() -> List[Tuple[str, str]]:
    """Load Holbrook/Birkbeck academic typo correction pairs."""
    global _holbrook_pairs
    if _holbrook_pairs is not None:
        return _holbrook_pairs
    
    print("📥 Loading Holbrook/Birkbeck academic typo pairs...")
    _holbrook_pairs = []
    
    # Built-in academic correction pairs from literature
    academic_pairs = [
        ("The qucik brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog."),
        ("I beleive this is teh correct answr.", "I believe this is the correct answer."),
        ("She recieved her degre last year.", "She received her degree last year."),
        ("The meeting was sucessful and very informativ.", "The meeting was successful and very informative."),
        ("We need to seperate these items carfully.", "We need to separate these items carefully."),
        ("The resturant serves excelent food.", "The restaurant serves excellent food."),
        ("He is studyng for his final examintion.", "He is studying for his final examination."),
        ("The goverment anounced new policies today.", "The government announced new policies today."),
        ("I wrte a leter to my frend yesterday.", "I wrote a letter to my friend yesterday."),
        ("The libary has many intresting books.", "The library has many interesting books."),
        ("We dicussed the importnt details.", "We discussed the important details."),
        ("The temperture is extrmely high today.", "The temperature is extremely high today."),
        ("I acidentally deleted the wrong file.", "I accidentally deleted the wrong file."),
        ("The comittee made an anouncement.", "The committee made an announcement."),
        ("She perfomed exceptionaly well in the test.", "She performed exceptionally well in the test."),
        ("The begining of the story was very excting.", "The beginning of the story was very exciting."),
        ("We reccommend this aproach for best results.", "We recommend this approach for best results."),
        ("The ocassion was truely memorable.", "The occasion was truly memorable."),
        ("I definately need to imporve my skils.", "I definitely need to improve my skills."),
        ("The experiance was both chalenging and rewardng.", "The experience was both challenging and rewarding.")
    ]
    
    _holbrook_pairs = academic_pairs
    print(f"✅ Loaded {len(_holbrook_pairs)} Holbrook/Birkbeck correction pairs")
    return _holbrook_pairs

def apply_norvig_corruption(words: List[str]) -> List[str]:
    """Apply Norvig misspellings to words."""
    misspellings = load_norvig_misspellings()
    corrupted = []
    
    for word in words:
        word_clean = re.sub(r'[^\w]', '', word.lower())
        if word_clean in misspellings and random.random() < 0.3:
            # Apply the misspelling while preserving case and punctuation
            correct_word = misspellings[word_clean]
            if word.isupper():
                corrupted.append(word.replace(word_clean.upper(), correct_word.upper()))
            elif word[0].isupper():
                corrupted.append(word.replace(word_clean.capitalize(), correct_word.capitalize()))
            else:
                corrupted.append(word.replace(word_clean, correct_word))
        else:
            corrupted.append(word)
    
    return corrupted

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
    pos = random.randint(1, len(word) - 2)  # Don't drop first/last
    if random.random() < 0.2:
        return word[:pos] + word[pos + 1:]
    return word

def character_double(word: str) -> str:
    """Double a character in word."""
    if len(word) < 2:
        return word
    pos = random.randint(0, len(word) - 1)
    char = word[pos]
    if random.random() < 0.15:
        return word[:pos + 1] + char + word[pos + 1:]
    return word

def character_transpose(word: str) -> str:
    """Swap adjacent characters."""
    if len(word) < 3:
        return word
    pos = random.randint(0, len(word) - 2)
    if random.random() < 0.2:
        chars = list(word)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return ''.join(chars)
    return word

def word_space_split(word: str) -> str:
    """Split word with space - common pattern is last character splits off."""
    if len(word) < 4:
        return word
    
    if random.random() < 0.15:  # Increased probability
        # Most common: last character splits off to next word
        if random.random() < 0.7:
            return word[:-1] + ' ' + word[-1]
        # Sometimes: first character splits off 
        elif random.random() < 0.5:
            return word[0] + ' ' + word[1:]
        # Random middle split (less common)
        else:
            pos = random.randint(2, len(word) - 2)
            return word[:pos] + ' ' + word[pos:]
    return word

def scramble_small_words(words: list) -> list:
    """Scramble sequences of small words (2-4 characters)."""
    result = []
    i = 0
    
    while i < len(words):
        word = words[i]
        
        # Find sequence of small words (2-4 chars, letters only)
        if (len(word) >= 2 and len(word) <= 4 and 
            word.isalpha() and random.random() < 0.1):
            
            # Collect small words in sequence
            small_sequence = [word]
            j = i + 1
            while (j < len(words) and j < i + 4 and  # Max 4 words
                   len(words[j]) >= 2 and len(words[j]) <= 4 and
                   words[j].isalpha()):
                small_sequence.append(words[j])
                j += 1
            
            # Only scramble if we have at least 2 small words
            if len(small_sequence) >= 2:
                # Scramble the sequence
                scrambled = small_sequence.copy()
                random.shuffle(scrambled)
                result.extend(scrambled)
                i = j  # Skip processed words
            else:
                result.append(word)
                i += 1
        else:
            result.append(word)
            i += 1
    
    return result

def character_merge_next(word: str, next_word: str = None) -> tuple:
    """Merge last character of word with next word."""
    if len(word) < 3 or not next_word or len(next_word) < 2:
        return word, next_word
    
    if random.random() < 0.12:  # 12% chance
        # Move last character to start of next word
        merged_current = word[:-1]
        merged_next = word[-1] + next_word
        return merged_current, merged_next
    
    return word, next_word

def word_confusion(word: str) -> str:
    """Replace with commonly confused word."""
    word_lower = word.lower()
    if word_lower in WORD_CONFUSIONS and random.random() < 0.25:
        replacement = WORD_CONFUSIONS[word_lower]
        # Preserve original case
        if word.isupper():
            return replacement.upper()
        elif word[0].isupper():
            return replacement.capitalize()
        else:
            return replacement
    return word

def corrupt_sentence(sentence: str, corruption_rate: float = 0.15, use_external_sources: bool = True) -> str:
    """Apply various corruptions to a sentence with realistic patterns from multiple sources."""
    words = sentence.split()
    
    # First try Norvig misspellings (20% chance for any word)
    if use_external_sources and random.random() < 0.2:
        words = apply_norvig_corruption(words)
    
    corrupted_words = []
    
    # Apply word-level corruptions and character merging
    i = 0
    while i < len(words):
        word = words[i]
        next_word = words[i + 1] if i + 1 < len(words) else None
        
        # Skip punctuation-only words
        if not re.search(r'[a-zA-Z]', word):
            corrupted_words.append(word)
            i += 1
            continue
        
        # Apply character merging between words (realistic pattern)
        if next_word and re.search(r'[a-zA-Z]', next_word):
            merged_current, merged_next = character_merge_next(word, next_word)
            if merged_current != word:  # Merging happened
                corrupted_words.append(merged_current)
                corrupted_words.append(merged_next)
                i += 2  # Skip next word since we processed it
                continue
        
        # Apply regular corruptions
        if random.random() < corruption_rate:
            # Enhanced corruption type selection with external sources
            corruption_choice = random.random()
            
            if corruption_choice < 0.25:  # 25% - space splitting (very common)
                corrupted_word = word_space_split(word)
            elif corruption_choice < 0.4:  # 15% - keyboard neighbors (enhanced)
                corrupted_word = enhanced_keyboard_corruption(word)
            elif corruption_choice < 0.55:  # 15% - character transpose
                corrupted_word = character_transpose(word)
            elif corruption_choice < 0.7:  # 15% - character drop
                corrupted_word = character_drop(word)
            elif corruption_choice < 0.8:  # 10% - word confusion
                corrupted_word = word_confusion(word)
            elif corruption_choice < 0.9:  # 10% - character double
                corrupted_word = character_double(word)
            else:  # 10% - academic misspellings
                corrupted_word = apply_academic_misspelling(word)
            
            corrupted_words.append(corrupted_word)
        else:
            corrupted_words.append(word)
        
        i += 1
    
    # Second pass: Apply small word scrambling to the result
    if random.random() < 0.08:  # 8% chance for sentence-level scrambling
        corrupted_words = scramble_small_words(corrupted_words)
    
    return ' '.join(corrupted_words)

def enhanced_keyboard_corruption(word: str) -> str:
    """Enhanced keyboard error simulation with multiple error types."""
    if len(word) < 2:
        return word
    
    corruption_type = random.random()
    
    if corruption_type < 0.4:  # Adjacent key substitution
        return keyboard_neighbor_swap(word)
    elif corruption_type < 0.6:  # Key insertion
        pos = random.randint(0, len(word))
        if pos < len(word):
            char = word[pos].lower()
            if char in KEYBOARD_NEIGHBORS:
                neighbor = random.choice(KEYBOARD_NEIGHBORS[char])
                return word[:pos] + neighbor + word[pos:]
    elif corruption_type < 0.8:  # Key deletion
        return character_drop(word)
    else:  # Key swap (transposition)
        return character_transpose(word)
    
    return word

def apply_academic_misspelling(word: str) -> str:
    """Apply academic-style misspellings from research literature."""
    word_lower = word.lower()
    
    # Common academic misspelling patterns
    academic_patterns = {
        'receive': 'recieve', 'believe': 'beleive', 'achieve': 'acheive',
        'separate': 'seperate', 'definitely': 'definately', 'necessary': 'neccessary',
        'occurred': 'occured', 'beginning': 'begining', 'successful': 'sucessful',
        'recommend': 'reccommend', 'accommodate': 'accomodate', 'privilege': 'priviledge',
        'tomorrow': 'tommorrow', 'until': 'untill', 'coming': 'comming',
        'friend': 'freind', 'piece': 'peice', 'weird': 'wierd',
        'government': 'goverment', 'restaurant': 'resturant', 'library': 'libary'
    }
    
    if word_lower in academic_patterns and random.random() < 0.4:
        misspelled = academic_patterns[word_lower]
        # Preserve case
        if word.isupper():
            return misspelled.upper()
        elif word[0].isupper():
            return misspelled.capitalize()
        else:
            return misspelled
    
    return word

def is_good_sentence(sentence: str) -> bool:
    """
    Filter for good natural sentences suitable for typo correction training.
    """
    sentence = sentence.strip()
    
    # Length checks - realistic sentence lengths
    if len(sentence) < 10 or len(sentence) > 150:
        return False
    
    # Word count - typical sentences
    words = sentence.split()
    if len(words) < 3 or len(words) > 25:
        return False
    
    # Must contain letters
    if not re.search(r'[a-zA-Z]', sentence):
        return False
    
    # Reject if mostly numbers or special characters
    letter_count = sum(1 for c in sentence if c.isalpha())
    if letter_count / len(sentence) < 0.6:
        return False
    
    # Reject sentences with too many special characters
    special_chars = len(re.findall(r'[^\w\s\'\-\.,\?!\(\):]', sentence))
    if special_chars > 3:
        return False
    
    # Reject if it looks like a header, list item, or metadata
    if (sentence.startswith('=') or 
        sentence.startswith('#') or
        sentence.startswith('*') or
        sentence.startswith('-') or
        sentence.startswith('•') or
        sentence.upper() == sentence and len(sentence) > 10):  # ALL CAPS
        return False
    
    # Reject URLs, emails, code
    if ('http' in sentence.lower() or 
        '@' in sentence or 
        'www.' in sentence.lower() or
        '```' in sentence or
        '{' in sentence or '}' in sentence):
        return False
    
    # Must end with proper punctuation
    if not sentence.endswith(('.', '!', '?', ';')):
        return False
    
    # Check for reasonable sentence structure (starts with capital)
    if not sentence[0].isupper():
        return False
    
    return True

def extract_natural_sentences(dataset_name: str = "wikitext", max_sentences: int = 50000) -> List[str]:
    """
    Extract natural, well-formed sentences from datasets.
    Focus on authentic sentence patterns for realistic training.
    """
    
    print(f"🔍 Extracting natural sentences from {dataset_name}...")
    
    sentences = []
    seen_sentences = set()  # Avoid duplicates
    
    try:
        if dataset_name == "wikitext":
            print("📚 Loading WikiText dataset...")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            text_key = 'text'
        elif dataset_name == "bookcorpus":
            print("📚 Loading BookCorpus dataset...")
            dataset = load_dataset("bookcorpus", split="train", streaming=True) 
            text_key = 'text'
        else:
            # Fallback to WikiText
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            text_key = 'text'
            
    except Exception as e:
        print(f"⚠️ Dataset loading failed ({e}), using sample sentences...")
        return get_sample_sentences()
    
    print(f"🔄 Processing dataset to extract natural sentences...")
    processed_docs = 0
    
    for example in tqdm(dataset, desc="Processing documents"):
        if len(sentences) >= max_sentences:
            break
            
        text = example[text_key].strip()
        processed_docs += 1
        
        if not text or len(text) < 20:
            continue
        
        # Split into sentences using multiple delimiters
        # Handle common sentence endings
        potential_sentences = re.split(r'[.!?]+\s+', text)
        
        for raw_sentence in potential_sentences:
            if len(sentences) >= max_sentences:
                break
                
            # Clean up the sentence
            sentence = raw_sentence.strip()
            
            # Add back period if it was split off
            if sentence and not sentence.endswith(('.', '!', '?', ';')):
                sentence += '.'
            
            # Apply quality filters
            if is_good_sentence(sentence):
                # Check for duplicates (case insensitive)
                sentence_lower = sentence.lower()
                if sentence_lower not in seen_sentences:
                    seen_sentences.add(sentence_lower)
                    sentences.append(sentence)
        
        # Progress update
        if processed_docs % 1000 == 0:
            print(f"   Processed {processed_docs} documents, found {len(sentences)} good sentences")
    
    print(f"✅ Extracted {len(sentences)} high-quality natural sentences")
    return sentences

def get_sample_sentences() -> List[str]:
    """
    Fallback high-quality sample sentences for when datasets fail.
    These are realistic, natural sentences for typo correction.
    """
    return [
        "The weather is beautiful today.",
        "I need to go to the grocery store later.",
        "She finished her homework before dinner.",
        "The meeting has been scheduled for tomorrow morning.",
        "Can you help me with this project?",
        "The book was very interesting and well written.",
        "We should probably leave soon to avoid traffic.",
        "His presentation was clear and informative.",
        "The restaurant serves excellent Italian food.",
        "I forgot to bring my umbrella this morning.",
        "The movie starts at seven thirty tonight.",
        "She works as a software engineer in the city.",
        "The train arrives at the station every hour.",
        "My brother is studying medicine at the university.",
        "The garden looks lovely in the spring.",
        "We need to finish this report by Friday.",
        "The coffee shop opens early every morning.",
        "He enjoys reading mystery novels in his free time.",
        "The conference was both educational and inspiring.",
        "I should call my parents this weekend.",
        "The new policy will take effect next month.",
        "She has been working on this project for weeks.",
        "The children played in the park all afternoon.",
        "We ordered pizza for dinner tonight.",
        "The lecture covered important topics in history.",
        "I need to renew my library books tomorrow.",
        "The team worked together to solve the problem.",
        "She graduated with honors from college.",
        "The store closes at nine o'clock.",
        "We should visit the museum while we're in town.",
        "The doctor recommended getting more exercise.",
        "I enjoy listening to music while I work.",
        "The flight was delayed due to bad weather.",
        "She teaches English at the local high school.",
        "We need to buy groceries for the week.",
        "The concert was absolutely fantastic.",
        "He drives to work every morning.",
        "The library has an excellent selection of books.",
        "We should plan our vacation for next summer.",
        "The presentation went very well.",
        "I need to update my resume soon.",
        "She volunteers at the animal shelter.",
        "The new restaurant downtown is very popular.",
        "We discussed the proposal during the meeting.",
        "The course covers both theory and practice.",
        "I should exercise more regularly.",
        "The weather forecast predicts rain tomorrow.",
        "She received a promotion at work.",
        "We need to clean the house this weekend.",
        "The museum has fascinating historical exhibits."
    ]

def classify_sentence_complexity(sentence: str) -> str:
    """Classify sentences by complexity for balanced training."""
    words = sentence.split()
    word_count = len(words)
    
    # Check for complex structures
    has_conjunctions = any(word.lower() in ['and', 'but', 'however', 'although', 'because'] for word in words)
    has_commas = ',' in sentence
    has_subclauses = sentence.count(',') >= 2 or ' which ' in sentence or ' that ' in sentence
    
    if word_count <= 8 and not has_commas:
        return "simple"
    elif word_count <= 15 and (has_conjunctions or has_commas) and not has_subclauses:
        return "medium"
    else:
        return "complex"

def generate_realistic_training_data(
    output_file: str,
    num_examples: int = 10000,
    corruption_rate: float = 0.15,
    dataset_name: str = "wikitext",
    use_external_sources: bool = True
):
    """
    Generate realistic typo correction training data using multiple high-quality sources.
    """
    
    print(f"🚀 Generating realistic typo correction training data with enhanced sources...")
    print(f"📁 Output: {output_file}")
    print(f"📊 Target examples: {num_examples:,}")
    print(f"🔀 Corruption rate: {corruption_rate}")
    print(f"📚 Source dataset: {dataset_name}")
    print(f"🔬 External sources: {'Enabled' if use_external_sources else 'Disabled'}")
    print()
    
    # Load external datasets if enabled
    holbrook_examples = []
    if use_external_sources:
        load_norvig_misspellings()  # Pre-load for caching
        holbrook_pairs = load_holbrook_pairs()
        
        # Add Holbrook pairs as training examples (high quality)
        for corrupted, clean in holbrook_pairs:
            holbrook_examples.append({
                "corrupted": corrupted,
                "clean": clean,
                "complexity": classify_sentence_complexity(clean),
                "word_count": len(clean.split()),
                "char_count": len(clean),
                "source": "holbrook"
            })
        
        print(f"🎓 Added {len(holbrook_examples)} Holbrook academic examples")
    
    # Extract natural sentences
    wiki_target = max(100, num_examples - len(holbrook_examples))  # Leave room for Holbrook
    sentences = extract_natural_sentences(dataset_name, max_sentences=wiki_target * 2)
    
    if not sentences:
        print("❌ No sentences extracted, using fallback samples")
        sentences = get_sample_sentences()
    
    # Classify sentences by complexity for balanced dataset
    sentence_groups = {"simple": [], "medium": [], "complex": []}
    
    for sentence in sentences:
        complexity = classify_sentence_complexity(sentence)
        sentence_groups[complexity].append(sentence)
    
    print(f"📊 WikiText sentence distribution:")
    print(f"   Simple: {len(sentence_groups['simple'])} sentences")
    print(f"   Medium: {len(sentence_groups['medium'])} sentences") 
    print(f"   Complex: {len(sentence_groups['complex'])} sentences")
    print()
    
    # Generate balanced training examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generated = 0
    
    # Reserve space for high-quality academic examples
    academic_reserve = len(holbrook_examples)
    wiki_examples_target = num_examples - academic_reserve
    
    complexity_targets = {
        "simple": int(wiki_examples_target * 0.4),    # 40% simple sentences
        "medium": int(wiki_examples_target * 0.4),    # 40% medium sentences  
        "complex": int(wiki_examples_target * 0.2)    # 20% complex sentences
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # First, write high-quality academic examples
        print(f"🎓 Writing {len(holbrook_examples)} academic examples...")
        for example in holbrook_examples:
            f.write(json.dumps(example) + '\n')
            generated += 1
        
        # Then generate WikiText examples with enhanced corruption
        for complexity, target_count in complexity_targets.items():
            available_sentences = sentence_groups[complexity]
            
            if not available_sentences:
                print(f"⚠️ No {complexity} sentences available, skipping...")
                continue
            
            print(f"🔄 Generating {target_count} {complexity} WikiText examples...")
            
            for _ in tqdm(range(target_count), desc=f"Processing {complexity}"):
                if generated >= num_examples:
                    break
                
                # Pick a random sentence from this complexity group
                sentence = random.choice(available_sentences)
                
                # Apply enhanced corruption with external sources
                corrupted = corrupt_sentence(sentence, corruption_rate, use_external_sources)
                
                # Only include if corruption actually occurred
                if corrupted != sentence:
                    data = {
                        "corrupted": corrupted,
                        "clean": sentence,
                        "complexity": complexity,
                        "word_count": len(sentence.split()),
                        "char_count": len(sentence),
                        "source": "wikitext"
                    }
                    
                    f.write(json.dumps(data) + '\n')
                    generated += 1
    
    print(f"✅ Generated {generated:,} realistic training examples")
    print(f"   🎓 Academic (Holbrook): {len(holbrook_examples)} examples")
    print(f"   📚 WikiText (enhanced): {generated - len(holbrook_examples)} examples")
    print(f"💾 Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate realistic single-sentence typo correction data with enhanced sources")
    parser.add_argument('--output', type=str, default='data/realistic/train.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--num_examples', type=int, default=10000,
                       help='Number of examples to generate')
    parser.add_argument('--corruption_rate', type=float, default=0.15,
                       help='Rate of token corruption (0.0-1.0)')
    parser.add_argument('--dataset', type=str, default='wikitext',
                       choices=['wikitext', 'bookcorpus'], 
                       help='Source dataset to use')
    parser.add_argument('--no_external_sources', action='store_true',
                       help='Disable external typo sources (Norvig, Holbrook)')
    
    args = parser.parse_args()
    
    generate_realistic_training_data(
        output_file=args.output,
        num_examples=args.num_examples,
        corruption_rate=args.corruption_rate,
        dataset_name=args.dataset,
        use_external_sources=not args.no_external_sources
    )
    
    print("✅ Realistic typo correction data generation complete!")
    
    # Show some examples
    print("\n📝 Sample generated examples:")
    try:
        with open(args.output, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 examples
                    break
                data = json.loads(line.strip())
                print(f"\n{i+1}. Complexity: {data['complexity']} ({data['word_count']} words)")
                print(f"   Corrupted:  \"{data['corrupted']}\"")
                print(f"   Clean:      \"{data['clean']}\"")
    except Exception as e:
        print(f"⚠️ Could not show examples: {e}")

if __name__ == "__main__":
    main()
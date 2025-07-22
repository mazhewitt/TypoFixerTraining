#!/usr/bin/env python3
"""
Enhanced synthetic typo generation for Qwen models.
Leverages Qwen's 32K context length to generate longer, more complex examples.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re
from datasets import load_dataset
from tqdm import tqdm

# Import corruption functions from existing data generation
from data_generation import (
    KEYBOARD_NEIGHBORS, WORD_CONFUSIONS,
    keyboard_neighbor_swap, character_drop, character_double, 
    character_transpose, word_space_split, word_confusion,
    corrupt_sentence
)

def create_multi_sentence_context(sentences: List[str], target_tokens: int = 512) -> str:
    """
    Combine multiple sentences into longer contexts for Qwen.
    Target token count is approximate (roughly 1 token per 4 characters in English).
    """
    context = ""
    current_length = 0
    target_chars = target_tokens * 4  # Rough estimation
    
    random.shuffle(sentences)
    
    for sentence in sentences:
        # Add sentence with proper spacing
        if context:
            context += " " + sentence.strip()
        else:
            context = sentence.strip()
            
        current_length = len(context)
        
        # Stop when we reach target length, but ensure we end with complete sentence
        if current_length >= target_chars:
            break
    
    return context

def create_paragraph_corruption(text: str, corruption_rate: float = 0.15) -> str:
    """
    Apply corruption to longer text while maintaining paragraph structure.
    """
    # Split into sentences for better corruption control
    sentences = re.split(r'([.!?]+)', text)
    corrupted_parts = []
    
    for part in sentences:
        if re.search(r'[a-zA-Z]', part):  # Only corrupt text parts, not punctuation
            corrupted_part = corrupt_sentence(part.strip(), corruption_rate)
            corrupted_parts.append(corrupted_part)
        else:
            corrupted_parts.append(part)  # Keep punctuation as-is
    
    return ''.join(corrupted_parts)

def add_paragraph_level_corruptions(text: str) -> str:
    """
    Add paragraph-level corruptions that require longer context understanding.
    """
    corruptions = []
    
    # Random chance to apply each type
    if random.random() < 0.1:  # 10% chance
        # Duplicate a sentence (common in rough drafts)
        sentences = re.split(r'[.!?]+', text)
        text_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(text_sentences) > 1:
            dup_sentence = random.choice(text_sentences)
            # Insert duplicate near original
            text = text.replace(dup_sentence, f"{dup_sentence}. {dup_sentence}", 1)
            corruptions.append("sentence_duplication")
    
    if random.random() < 0.05:  # 5% chance
        # Missing sentence ending punctuation
        text = re.sub(r'\.(\s+[A-Z])', r'\1', text, count=1)
        corruptions.append("missing_punctuation")
    
    if random.random() < 0.1:  # 10% chance
        # Word repetition across sentence boundaries
        words = text.split()
        if len(words) > 10:
            idx = random.randint(5, len(words) - 5)
            words.insert(idx, words[idx])  # Duplicate a word
            text = ' '.join(words)
            corruptions.append("word_repetition")
    
    return text

def generate_qwen_training_data(
    output_file: str,
    num_examples: int = 10000,
    corruption_rate: float = 0.15,
    context_lengths: List[int] = [256, 512, 1024],
    dataset_name: str = "wikitext"
):
    """
    Generate training data optimized for Qwen's capabilities.
    Creates examples of varying lengths to utilize Qwen's long context.
    """
    
    print(f"🚀 Generating Qwen-optimized training data...")
    print(f"📁 Output: {output_file}")
    print(f"📊 Target examples: {num_examples:,}")
    print(f"📏 Context lengths: {context_lengths}")
    
    # Load source dataset
    try:
        print(f"Loading {dataset_name} dataset...")
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            text_key = 'text'
        else:
            # Fallback to wikitext if other datasets fail
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            text_key = 'text'
    except Exception as e:
        print(f"⚠️ Dataset loading failed ({e}), using sample data...")
        return generate_sample_qwen_data(output_file, num_examples, corruption_rate)
    
    # Collect sentences from dataset
    print("📚 Collecting sentences...")
    sentences = []
    sentence_count = num_examples * 3  # Collect more sentences than needed
    
    for example in tqdm(dataset, desc="Collecting sentences", total=sentence_count):
        if len(sentences) >= sentence_count:
            break
            
        text = example[text_key].strip()
        
        if len(text) < 10 or text.startswith('=') or not text.strip():
            continue
            
        # Split into sentences
        text_sentences = re.split(r'[.!?]+', text)
        
        for sentence in text_sentences:
            sentence = sentence.strip()
            
            # Filter: reasonable length, contains letters
            if 20 <= len(sentence) <= 300 and re.search(r'[a-zA-Z]', sentence):
                # Clean up
                sentence = re.sub(r'\s+', ' ', sentence).strip()
                if len(sentence) >= 20:
                    sentences.append(sentence)
                    
                if len(sentences) >= sentence_count:
                    break
    
    print(f"📊 Collected {len(sentences)} sentences")
    
    # Generate training examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generated = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        with tqdm(total=num_examples, desc="Generating examples") as pbar:
            while generated < num_examples:
                # Choose random context length
                target_tokens = random.choice(context_lengths)
                
                # Create multi-sentence context
                context = create_multi_sentence_context(sentences, target_tokens)
                
                if len(context) < 50:  # Skip very short contexts
                    continue
                
                # Apply standard corruption
                corrupted = create_paragraph_corruption(context, corruption_rate)
                
                # Add paragraph-level corruptions occasionally
                if random.random() < 0.3:  # 30% chance
                    corrupted = add_paragraph_level_corruptions(corrupted)
                
                # Only include if corruption occurred
                if corrupted != context:
                    data = {
                        "corrupted": corrupted,
                        "clean": context,
                        "context_length": len(context.split()),  # Approximate token count
                        "corruption_types": "multi_sentence"
                    }
                    
                    f.write(json.dumps(data) + '\n')
                    generated += 1
                    pbar.update(1)

def generate_sample_qwen_data(output_file: str, num_examples: int, corruption_rate: float):
    """Generate sample data when datasets aren't available."""
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet. It's commonly used for testing fonts and keyboards. The fox represents speed and agility, while the dog represents laziness and comfort.",
        
        "Climate change is one of the most pressing issues of our time. Rising temperatures affect weather patterns globally. Scientists have been studying this phenomenon for decades. The greenhouse effect traps heat in our atmosphere, causing temperatures to rise.",
        
        "Artificial intelligence is transforming how we work and live. Machine learning algorithms can recognize patterns in vast datasets. Natural language processing enables computers to understand human speech. These technologies will continue to evolve rapidly.",
        
        "The history of human civilization spans thousands of years. Ancient civilizations built monuments that still stand today. The pyramids of Egypt, the Great Wall of China, and Stonehenge are examples. These structures show human ingenuity and determination.",
        
        "Cooking is both an art and a science. Understanding chemical reactions helps create better dishes. Heat transforms ingredients in predictable ways. Salt enhances flavors while acids add brightness to meals.",
    ]
    
    print(f"📝 Generating {num_examples} sample examples...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_examples):
            # Combine 2-4 sample texts
            num_texts = random.randint(2, 4)
            selected_texts = random.sample(sample_texts, min(num_texts, len(sample_texts)))
            context = " ".join(selected_texts)
            
            # Apply corruption
            corrupted = create_paragraph_corruption(context, corruption_rate)
            
            if corrupted != context:
                data = {
                    "corrupted": corrupted,
                    "clean": context,
                    "context_length": len(context.split()),
                    "corruption_types": "sample_multi_sentence"
                }
                
                f.write(json.dumps(data) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate Qwen-optimized typo correction data")
    parser.add_argument('--output', type=str, default='data/qwen/train.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--num_examples', type=int, default=10000,
                       help='Number of examples to generate')
    parser.add_argument('--corruption_rate', type=float, default=0.15,
                       help='Rate of token corruption (0.0-1.0)')
    parser.add_argument('--context_lengths', type=int, nargs='+', default=[256, 512, 1024],
                       help='Target context lengths in tokens')
    parser.add_argument('--dataset', type=str, default='wikitext',
                       help='Source dataset')
    
    args = parser.parse_args()
    
    generate_qwen_training_data(
        output_file=args.output,
        num_examples=args.num_examples,
        corruption_rate=args.corruption_rate,
        context_lengths=args.context_lengths,
        dataset_name=args.dataset
    )
    
    print("✅ Qwen training data generation complete!")

if __name__ == "__main__":
    main()
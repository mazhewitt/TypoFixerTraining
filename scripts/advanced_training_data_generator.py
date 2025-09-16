#!/usr/bin/env python3
"""
Advanced Training Data Generation Engine

Combines the sophisticated error pattern library with diversified source text
to create realistic training data for typo correction.
"""

import random
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from enum import Enum
from tqdm import tqdm
import sys

# Import our custom modules
sys.path.append(str(Path(__file__).parent))
from error_pattern_library import AdvancedErrorPatterns, ErrorType
from source_text_diversifier import SourceTextDiversifier


@dataclass
class TrainingExample:
    """Structured training example with metadata"""
    corrupted: str
    clean: str
    domain: str
    complexity: str
    error_types: List[str]
    num_errors: int
    word_count: int
    char_count: int
    difficulty_score: float
    source: str


class ComplexityLevel(Enum):
    SIMPLE = "simple"      # 1-2 errors, common words
    MEDIUM = "medium"      # 2-4 errors, mixed difficulty  
    COMPLEX = "complex"    # 4+ errors, challenging patterns


class AdvancedTrainingDataGenerator:
    """Advanced training data generator with sophisticated error patterns"""
    
    def __init__(self):
        self.error_patterns = AdvancedErrorPatterns()
        self.diversifier = SourceTextDiversifier()
        self.generation_stats = defaultdict(int)
        
        # Complexity distribution targets
        self.complexity_targets = {
            ComplexityLevel.SIMPLE: 0.30,   # 30% simple cases
            ComplexityLevel.MEDIUM: 0.50,   # 50% medium cases
            ComplexityLevel.COMPLEX: 0.20,  # 20% complex cases
        }
        
        # Error pattern distribution targets  
        self.error_distribution_targets = {
            "multi_error_sentences": 0.40,      # 40% - Multiple errors per sentence
            "contextual_phonetic": 0.20,        # 20% - Contextual and phonetic errors
            "advanced_typography": 0.15,        # 15% - Typography and formatting
            "keyboard_patterns": 0.15,          # 15% - Keyboard-based errors
            "simple_single": 0.10,              # 10% - Simple single-word errors
        }
    
    def calculate_difficulty_score(self, original: str, corrupted: str, 
                                 error_types: List[str], num_errors: int) -> float:
        """Calculate difficulty score for the training example"""
        
        score = 0.0
        word_count = len(original.split())
        
        # Base score from error count relative to sentence length
        score += min(num_errors / word_count, 0.5) * 40
        
        # Error type difficulty weights
        error_weights = {
            "keyboard": 0.6,     # Easier - adjacent keys
            "spelling": 0.8,     # Medium - character operations
            "phonetic": 1.0,     # Harder - sound-based confusions
            "grammar": 1.2,      # Harder - grammatical knowledge
            "punctuation": 0.7,  # Medium - contraction issues
            "spacing": 0.5,      # Easier - compound word issues
        }
        
        # Add weighted error type difficulty
        for error_type in error_types:
            score += error_weights.get(error_type, 0.8) * 15
        
        # Sentence length factor (longer = harder)
        score += min(word_count / 20, 1.0) * 10
        
        # Character change ratio
        char_changes = sum(1 for a, b in zip(original, corrupted) if a != b)
        change_ratio = char_changes / max(len(original), 1)
        score += change_ratio * 20
        
        # Normalize to 0-100 scale
        return min(score, 100.0)
    
    def determine_complexity_level(self, sentence: str, num_errors: int, 
                                 error_types: List[str]) -> ComplexityLevel:
        """Determine complexity level based on sentence and error characteristics"""
        
        word_count = len(sentence.split())
        
        # Simple cases: 1-2 errors, short sentences, basic error types
        if (num_errors <= 2 and 
            word_count <= 10 and
            all(et in ["keyboard", "spelling", "spacing"] for et in error_types)):
            return ComplexityLevel.SIMPLE
        
        # Complex cases: 4+ errors, long sentences, advanced error types
        elif (num_errors >= 4 or
              word_count >= 18 or
              any(et in ["phonetic", "grammar"] for et in error_types) or
              len(set(error_types)) >= 3):
            return ComplexityLevel.COMPLEX
        
        # Everything else is medium
        else:
            return ComplexityLevel.MEDIUM
    
    def generate_multi_error_sentence(self, sentence: str, domain: str) -> Optional[TrainingExample]:
        """Generate sentence with multiple realistic errors"""
        
        words = sentence.strip().split()
        if len(words) < 6:  # Need sufficient words for multiple errors
            return None
        
        # Determine number of errors (2-6 based on sentence length)
        min_errors = 2
        max_errors = min(6, max(2, len(words) // 3))
        num_errors = random.randint(min_errors, max_errors)
        
        # Apply complex corruption
        corrupted = self.error_patterns.corrupt_sentence(
            sentence, 
            complexity="complex", 
            num_errors=num_errors
        )
        
        # Ensure we actually made meaningful changes
        if corrupted == sentence:
            return None
        
        # Count actual changes
        actual_errors = sum(1 for w1, w2 in zip(sentence.split(), corrupted.split()) if w1 != w2)
        if actual_errors < 2:
            return None
        
        # Determine error types present
        error_types = self.analyze_error_types(sentence, corrupted)
        
        # Calculate complexity and difficulty
        complexity = self.determine_complexity_level(sentence, actual_errors, error_types)
        difficulty_score = self.calculate_difficulty_score(sentence, corrupted, error_types, actual_errors)
        
        return TrainingExample(
            corrupted=corrupted,
            clean=sentence,
            domain=domain,
            complexity=complexity.value,
            error_types=error_types,
            num_errors=actual_errors,
            word_count=len(words),
            char_count=len(sentence),
            difficulty_score=difficulty_score,
            source="multi_error_advanced"
        )
    
    def generate_contextual_phonetic_sentence(self, sentence: str, domain: str) -> Optional[TrainingExample]:
        """Generate sentence with contextual and phonetic errors"""
        
        words = sentence.strip().split()
        if len(words) < 4:
            return None
        
        # Apply phonetic corruption with contextual awareness
        corrupted_words = words.copy()
        error_count = 0
        applied_errors = []
        
        # Apply contextual errors first
        corrupted_words = self.error_patterns.apply_contextual_errors(corrupted_words)
        
        # Apply phonetic errors to 1-3 words
        num_phonetic_errors = random.randint(1, min(3, len(words) // 3))
        indices_to_corrupt = random.sample(range(len(words)), num_phonetic_errors)
        
        for idx in indices_to_corrupt:
            original_word = corrupted_words[idx]
            phonetic_word = self.error_patterns.apply_phonetic_error(original_word)
            
            if phonetic_word != original_word:
                corrupted_words[idx] = phonetic_word
                error_count += 1
                applied_errors.append("phonetic")
        
        # Apply contraction errors occasionally
        if random.random() < 0.3:
            for idx in range(len(corrupted_words)):
                original_word = corrupted_words[idx]
                contraction_word = self.error_patterns.apply_contraction_error(original_word)
                if contraction_word != original_word:
                    corrupted_words[idx] = contraction_word
                    error_count += 1
                    applied_errors.append("punctuation")
        
        corrupted = ' '.join(corrupted_words)
        
        # Ensure meaningful changes occurred
        if corrupted == sentence or error_count == 0:
            return None
        
        # Determine error types and complexity
        error_types = list(set(applied_errors + ["grammar"]))  # Contextual errors are grammatical
        complexity = self.determine_complexity_level(sentence, error_count, error_types)
        difficulty_score = self.calculate_difficulty_score(sentence, corrupted, error_types, error_count)
        
        return TrainingExample(
            corrupted=corrupted,
            clean=sentence,
            domain=domain,
            complexity=complexity.value,
            error_types=error_types,
            num_errors=error_count,
            word_count=len(words),
            char_count=len(sentence),
            difficulty_score=difficulty_score,
            source="contextual_phonetic"
        )
    
    def generate_advanced_typography_sentence(self, sentence: str, domain: str) -> Optional[TrainingExample]:
        """Generate sentence with advanced typography errors"""
        
        words = sentence.strip().split()
        if len(words) < 3:
            return None
        
        corrupted = sentence
        error_count = 0
        applied_errors = []
        
        # Apply double letter errors
        if random.random() < 0.6:
            for _ in range(random.randint(1, 2)):
                word_idx = random.randint(0, len(words) - 1)
                original_word = words[word_idx]
                double_letter_word = self.error_patterns.apply_double_letter_error(original_word)
                if double_letter_word != original_word:
                    corrupted = corrupted.replace(original_word, double_letter_word, 1)
                    error_count += 1
                    applied_errors.append("spelling")
        
        # Apply compound word errors
        if random.random() < 0.4:
            for word_idx in range(len(words)):
                original_word = words[word_idx]
                compound_word = self.error_patterns.apply_compound_word_error(original_word)
                if compound_word != original_word:
                    corrupted = corrupted.replace(original_word, compound_word, 1)
                    error_count += 1
                    applied_errors.append("spacing")
        
        # Apply character operation errors
        if error_count < 2:  # Ensure we have at least some errors
            words_corrupted = corrupted.split()
            for _ in range(random.randint(1, 3)):
                if not words_corrupted:
                    break
                word_idx = random.randint(0, len(words_corrupted) - 1)
                original_word = words_corrupted[word_idx]
                char_op_word = self.error_patterns.apply_character_operation_error(original_word)
                if char_op_word != original_word:
                    words_corrupted[word_idx] = char_op_word
                    error_count += 1
                    applied_errors.append("spelling")
            corrupted = ' '.join(words_corrupted)
        
        # Ensure meaningful changes occurred
        if corrupted == sentence or error_count == 0:
            return None
        
        error_types = list(set(applied_errors))
        complexity = self.determine_complexity_level(sentence, error_count, error_types)
        difficulty_score = self.calculate_difficulty_score(sentence, corrupted, error_types, error_count)
        
        return TrainingExample(
            corrupted=corrupted,
            clean=sentence,
            domain=domain,
            complexity=complexity.value,
            error_types=error_types,
            num_errors=error_count,
            word_count=len(words),
            char_count=len(sentence),
            difficulty_score=difficulty_score,
            source="advanced_typography"
        )
    
    def generate_keyboard_pattern_sentence(self, sentence: str, domain: str) -> Optional[TrainingExample]:
        """Generate sentence with keyboard pattern errors"""
        
        words = sentence.strip().split()
        if len(words) < 3:
            return None
        
        # Apply keyboard errors to 1-3 words
        num_keyboard_errors = random.randint(1, min(3, max(1, len(words) // 4)))
        indices_to_corrupt = random.sample(range(len(words)), num_keyboard_errors)
        
        corrupted_words = words.copy()
        error_count = 0
        
        for idx in indices_to_corrupt:
            original_word = corrupted_words[idx]
            keyboard_word = self.error_patterns.apply_keyboard_error(original_word)
            if keyboard_word != original_word:
                corrupted_words[idx] = keyboard_word
                error_count += 1
        
        # Add some character operations for variety
        if random.random() < 0.4:
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, len(corrupted_words) - 1)
                original_word = corrupted_words[idx]
                char_op_word = self.error_patterns.apply_character_operation_error(original_word)
                if char_op_word != original_word:
                    corrupted_words[idx] = char_op_word
                    error_count += 1
        
        corrupted = ' '.join(corrupted_words)
        
        # Ensure meaningful changes occurred
        if corrupted == sentence or error_count == 0:
            return None
        
        error_types = ["keyboard", "spelling"]
        complexity = self.determine_complexity_level(sentence, error_count, error_types)
        difficulty_score = self.calculate_difficulty_score(sentence, corrupted, error_types, error_count)
        
        return TrainingExample(
            corrupted=corrupted,
            clean=sentence,
            domain=domain,
            complexity=complexity.value,
            error_types=error_types,
            num_errors=error_count,
            word_count=len(words),
            char_count=len(sentence),
            difficulty_score=difficulty_score,
            source="keyboard_patterns"
        )
    
    def generate_simple_single_sentence(self, sentence: str, domain: str) -> Optional[TrainingExample]:
        """Generate sentence with simple single-word errors"""
        
        words = sentence.strip().split()
        if len(words) < 3:
            return None
        
        # Apply simple corruption to 1-2 words max
        num_errors = random.randint(1, min(2, max(1, len(words) // 6)))
        
        corrupted = self.error_patterns.corrupt_sentence(
            sentence,
            complexity="simple",
            num_errors=num_errors
        )
        
        # Ensure we made changes
        if corrupted == sentence:
            return None
        
        # Count actual errors
        actual_errors = sum(1 for w1, w2 in zip(sentence.split(), corrupted.split()) if w1 != w2)
        if actual_errors == 0:
            return None
        
        error_types = ["spelling", "keyboard"]  # Simple error types
        complexity = ComplexityLevel.SIMPLE
        difficulty_score = self.calculate_difficulty_score(sentence, corrupted, error_types, actual_errors)
        
        return TrainingExample(
            corrupted=corrupted,
            clean=sentence,
            domain=domain,
            complexity=complexity.value,
            error_types=error_types,
            num_errors=actual_errors,
            word_count=len(words),
            char_count=len(sentence),
            difficulty_score=difficulty_score,
            source="simple_single"
        )
    
    def analyze_error_types(self, original: str, corrupted: str) -> List[str]:
        """Analyze what types of errors are present in the corrupted text"""
        
        error_types = set()
        orig_words = original.split()
        corr_words = corrupted.split()
        
        # Compare words
        for orig_word, corr_word in zip(orig_words, corr_words):
            if orig_word != corr_word:
                # Check for keyboard errors (adjacent key substitutions)
                if self.looks_like_keyboard_error(orig_word, corr_word):
                    error_types.add("keyboard")
                
                # Check for phonetic errors
                if self.looks_like_phonetic_error(orig_word, corr_word):
                    error_types.add("phonetic")
                
                # Check for spacing/punctuation
                if "'" in orig_word or "'" in corr_word or " " in corr_word:
                    error_types.add("punctuation")
                
                # Default to spelling for character-level changes
                if len(error_types) == 0:
                    error_types.add("spelling")
        
        # Check for grammar/context errors
        if len(orig_words) == len(corr_words):
            for i, (orig, corr) in enumerate(zip(orig_words, corr_words)):
                if orig.lower() != corr.lower():
                    # Common grammar confusions
                    if (orig.lower(), corr.lower()) in [("then", "than"), ("than", "then"), 
                                                       ("their", "there"), ("there", "their"),
                                                       ("your", "you're"), ("you're", "your")]:
                        error_types.add("grammar")
        
        return list(error_types) if error_types else ["spelling"]
    
    def looks_like_keyboard_error(self, orig: str, corr: str) -> bool:
        """Check if error looks like keyboard mistake"""
        if len(orig) != len(corr):
            return False
        
        keyboard_layout = self.error_patterns.keyboard_layout
        diff_count = sum(1 for a, b in zip(orig.lower(), corr.lower()) if a != b)
        
        # Single character difference that could be keyboard neighbor
        if diff_count == 1:
            for a, b in zip(orig.lower(), corr.lower()):
                if a != b and a in keyboard_layout and b in keyboard_layout[a]:
                    return True
        
        return False
    
    def looks_like_phonetic_error(self, orig: str, corr: str) -> bool:
        """Check if error looks like phonetic mistake"""
        # Common phonetic patterns
        phonetic_pairs = [
            ("ph", "f"), ("ough", "uff"), ("eigh", "ay"), ("tion", "sion"),
            ("c", "s"), ("s", "c"), ("f", "ph"), ("i", "y"), ("y", "i")
        ]
        
        orig_lower = orig.lower()
        corr_lower = corr.lower()
        
        for pattern, replacement in phonetic_pairs:
            if pattern in orig_lower and replacement in corr_lower:
                return True
            if replacement in orig_lower and pattern in corr_lower:
                return True
        
        return False
    
    def generate_training_dataset(self, source_sentences: Dict[str, List[str]], 
                                target_size: int) -> List[TrainingExample]:
        """Generate complete training dataset with balanced error patterns"""
        
        print(f"🎯 Generating {target_size:,} training examples with advanced error patterns")
        print("=" * 70)
        
        examples = []
        generation_methods = [
            ("multi_error_sentences", self.generate_multi_error_sentence),
            ("contextual_phonetic", self.generate_contextual_phonetic_sentence),
            ("advanced_typography", self.generate_advanced_typography_sentence),
            ("keyboard_patterns", self.generate_keyboard_pattern_sentence),
            ("simple_single", self.generate_simple_single_sentence),
        ]
        
        # Calculate targets for each method
        method_targets = {}
        for method_name, _ in generation_methods:
            target_ratio = self.error_distribution_targets[method_name]
            method_targets[method_name] = int(target_size * target_ratio)
            print(f"  {method_name:20}: {method_targets[method_name]:5,} examples ({target_ratio:5.1%})")
        
        print()
        
        # Prepare sentence pools by domain
        all_sentences_by_domain = source_sentences
        total_source_sentences = sum(len(sentences) for sentences in all_sentences_by_domain.values())
        print(f"📚 Source sentences: {total_source_sentences:,} across {len(all_sentences_by_domain)} domains")
        print()
        
        # Generate examples for each method
        for method_name, generation_method in generation_methods:
            target_count = method_targets[method_name]
            method_examples = []
            
            print(f"🔄 Generating {target_count:,} examples using {method_name}...")
            
            max_attempts = target_count * 3  # Allow multiple attempts
            attempts = 0
            
            with tqdm(total=target_count, desc=f"  {method_name}") as pbar:
                while len(method_examples) < target_count and attempts < max_attempts:
                    attempts += 1
                    
                    # Select random domain and sentence
                    domain = random.choice(list(all_sentences_by_domain.keys()))
                    if not all_sentences_by_domain[domain]:
                        continue
                    
                    sentence = random.choice(all_sentences_by_domain[domain])
                    
                    # Generate example using the specific method
                    example = generation_method(sentence, domain)
                    
                    if example:
                        method_examples.append(example)
                        pbar.update(1)
                        self.generation_stats[method_name] += 1
            
            examples.extend(method_examples)
            print(f"  ✅ Generated {len(method_examples):,} examples ({len(method_examples)/target_count:.1%} success rate)")
        
        print(f"\n📊 Total examples generated: {len(examples):,}")
        
        # Shuffle final dataset
        random.shuffle(examples)
        
        return examples
    
    def save_training_dataset(self, examples: List[TrainingExample], 
                            output_file: str = "data/advanced_training_dataset.jsonl"):
        """Save training dataset with comprehensive metadata"""
        
        os.makedirs("data", exist_ok=True)
        
        # Calculate statistics
        stats = self.calculate_dataset_statistics(examples)
        
        # Save examples in JSONL format
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(asdict(example)) + '\n')
        
        # Save metadata separately
        metadata_file = output_file.replace('.jsonl', '_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(examples):,} training examples to {output_file}")
        print(f"📊 Metadata saved to {metadata_file}")
        
        return stats
    
    def calculate_dataset_statistics(self, examples: List[TrainingExample]) -> Dict:
        """Calculate comprehensive dataset statistics"""
        
        stats = {
            "total_examples": len(examples),
            "generation_stats": dict(self.generation_stats),
            "complexity_distribution": Counter(ex.complexity for ex in examples),
            "domain_distribution": Counter(ex.domain for ex in examples),
            "error_type_distribution": Counter(),
            "source_distribution": Counter(ex.source for ex in examples),
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
        
        return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate advanced training dataset")
    parser.add_argument('--source-file', 
                       default='data/diverse_source_sentences.json',
                       help='Source sentences file')
    parser.add_argument('--target-size', type=int, default=100000,
                       help='Target number of training examples')
    parser.add_argument('--output-file', 
                       default='data/advanced_training_dataset.jsonl',
                       help='Output training file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load source sentences
    print(f"📚 Loading source sentences from {args.source_file}...")
    
    if os.path.exists(args.source_file):
        with open(args.source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        source_sentences = source_data.get("sentences_by_domain", {})
        print(f"✅ Loaded {sum(len(s) for s in source_sentences.values()):,} source sentences")
    else:
        print("⚠️  Source file not found, collecting sentences first...")
        diversifier = SourceTextDiversifier()
        source_sentences = diversifier.collect_diverse_sentences(50000)
        print("✅ Collected source sentences")
    
    # Generate training dataset
    generator = AdvancedTrainingDataGenerator()
    examples = generator.generate_training_dataset(source_sentences, args.target_size)
    
    # Save dataset
    stats = generator.save_training_dataset(examples, args.output_file)
    
    # Print final statistics
    print(f"\n📈 DATASET STATISTICS:")
    print(f"  Total Examples: {stats['total_examples']:,}")
    print(f"  Complexity Distribution:")
    for complexity, count in stats['complexity_distribution'].items():
        print(f"    {complexity:8}: {count:6,} ({count/stats['total_examples']:5.1%})")
    
    print(f"  Error Type Distribution:")
    for error_type, count in stats['error_type_distribution'].most_common():
        print(f"    {error_type:12}: {count:6,}")
    
    print(f"  Average Difficulty: {stats['difficulty_stats']['mean']:.1f}")
    print(f"  Average Errors/Example: {stats['error_count_stats']['mean']:.1f}")
    print(f"\n✅ Advanced training dataset generation complete!")


if __name__ == "__main__":
    main()
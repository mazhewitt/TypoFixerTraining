#!/usr/bin/env python3
"""
Quality Validator for Training Data

Validates that generated training examples meet quality standards for
effective typo correction model training.
"""

import json
import re
import statistics
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    PASS = "pass"
    WARNING = "warning" 
    FAIL = "fail"


@dataclass
class ValidationResult:
    level: ValidationLevel
    category: str
    message: str
    example_id: Optional[int] = None
    example_text: Optional[str] = None


class QualityValidator:
    """Comprehensive quality validator for training data"""
    
    def __init__(self):
        self.setup_validation_rules()
        self.validation_results = []
        self.example_issues = defaultdict(list)
    
    def setup_validation_rules(self):
        """Define validation thresholds and patterns"""
        
        # Quality thresholds
        self.thresholds = {
            "min_word_count": 4,
            "max_word_count": 30,
            "min_char_count": 20,
            "max_char_count": 200,
            "min_change_ratio": 0.05,  # At least 5% character change
            "max_change_ratio": 0.50,  # No more than 50% character change
            "min_word_change_ratio": 0.10,  # At least 10% word change
            "max_word_change_ratio": 0.60,  # No more than 60% word change
            "max_consecutive_errors": 3,   # Avoid too many consecutive errors
            "min_readability": 0.3,        # Corrupted text should still be readable
        }
        
        # Complexity distribution targets (with tolerance)
        self.complexity_targets = {
            "simple": (0.25, 0.35),    # 25-35%
            "medium": (0.45, 0.55),    # 45-55%
            "complex": (0.15, 0.25),   # 15-25%
        }
        
        # Error type targets (minimum percentages)
        self.error_type_minimums = {
            "spelling": 0.40,      # At least 40% should have spelling errors
            "phonetic": 0.15,      # At least 15% should have phonetic errors
            "keyboard": 0.20,      # At least 20% should have keyboard errors
            "punctuation": 0.10,   # At least 10% should have punctuation errors
        }
        
        # Problematic patterns to avoid
        self.avoid_patterns = [
            r'^[A-Z\s]+$',              # ALL CAPS sentences
            r'\d{4,}',                  # Long numbers
            r'http|www|@|\.com',        # URLs/emails
            r'[<>{}[\]\\|]',           # Markup characters
            r'\.{3,}|\?{2,}|!{2,}',    # Multiple punctuation
            r'\b[bcdfghjklmnpqrstvwxz]{4,}\b',  # Consonant clusters
            r'(.)\1{3,}',              # Repeated characters (4+)
            r'\b\w{20,}\b',            # Very long words
        ]
    
    def validate_individual_example(self, example, index: int) -> List[ValidationResult]:
        """Validate a single training example"""
        
        results = []
        
        # Handle both dict and dataclass objects
        if hasattr(example, 'corrupted'):  # dataclass
            corrupted = example.corrupted
            clean = example.clean
        else:  # dict
            corrupted = example.get("corrupted", "")
            clean = example.get("clean", "")
        
        # Basic structure validation
        if not corrupted or not clean:
            results.append(ValidationResult(
                ValidationLevel.FAIL,
                "structure",
                "Missing corrupted or clean text",
                index,
                f"corrupted: '{corrupted[:50]}...'"
            ))
            return results
        
        # Word count validation
        corrupted_words = corrupted.split()
        clean_words = clean.split()
        
        if len(clean_words) < self.thresholds["min_word_count"]:
            results.append(ValidationResult(
                ValidationLevel.FAIL,
                "word_count",
                f"Too few words: {len(clean_words)} < {self.thresholds['min_word_count']}",
                index,
                clean[:80]
            ))
        
        if len(clean_words) > self.thresholds["max_word_count"]:
            results.append(ValidationResult(
                ValidationLevel.WARNING,
                "word_count",
                f"Many words: {len(clean_words)} > {self.thresholds['max_word_count']}",
                index,
                clean[:80]
            ))
        
        # Character count validation
        if len(clean) < self.thresholds["min_char_count"]:
            results.append(ValidationResult(
                ValidationLevel.FAIL,
                "char_count",
                f"Too short: {len(clean)} chars < {self.thresholds['min_char_count']}",
                index,
                clean
            ))
        
        # Change ratio validation
        char_changes = sum(1 for a, b in zip(clean, corrupted) if a != b)
        char_change_ratio = char_changes / max(len(clean), 1)
        
        if char_change_ratio < self.thresholds["min_change_ratio"]:
            results.append(ValidationResult(
                ValidationLevel.FAIL,
                "insufficient_change",
                f"Too few changes: {char_change_ratio:.2%} < {self.thresholds['min_change_ratio']:.2%}",
                index,
                f"'{corrupted}' vs '{clean}'"
            ))
        
        if char_change_ratio > self.thresholds["max_change_ratio"]:
            results.append(ValidationResult(
                ValidationLevel.FAIL,
                "excessive_change",
                f"Too many changes: {char_change_ratio:.2%} > {self.thresholds['max_change_ratio']:.2%}",
                index,
                f"'{corrupted}' vs '{clean}'"
            ))
        
        # Word-level change validation
        word_changes = sum(1 for w1, w2 in zip(corrupted_words, clean_words) if w1 != w2)
        word_change_ratio = word_changes / max(len(clean_words), 1)
        
        if word_change_ratio < self.thresholds["min_word_change_ratio"]:
            results.append(ValidationResult(
                ValidationLevel.WARNING,
                "low_word_changes",
                f"Few word changes: {word_change_ratio:.2%}",
                index
            ))
        
        # Pattern validation
        for pattern in self.avoid_patterns:
            if re.search(pattern, corrupted) or re.search(pattern, clean):
                results.append(ValidationResult(
                    ValidationLevel.WARNING,
                    "problematic_pattern",
                    f"Contains problematic pattern: {pattern}",
                    index,
                    corrupted[:80]
                ))
        
        # Readability check
        readability_score = self.calculate_readability(corrupted)
        if readability_score < self.thresholds["min_readability"]:
            results.append(ValidationResult(
                ValidationLevel.WARNING,
                "low_readability",
                f"Low readability: {readability_score:.2f}",
                index,
                corrupted
            ))
        
        # Consecutive error check
        consecutive_errors = self.count_consecutive_errors(clean_words, corrupted_words)
        if consecutive_errors > self.thresholds["max_consecutive_errors"]:
            results.append(ValidationResult(
                ValidationLevel.WARNING,
                "consecutive_errors",
                f"Too many consecutive errors: {consecutive_errors}",
                index,
                corrupted
            ))
        
        # Meaningfulness check
        if self.is_corrupted_meaningless(corrupted):
            results.append(ValidationResult(
                ValidationLevel.FAIL,
                "meaningless",
                "Corrupted text is not meaningful",
                index,
                corrupted
            ))
        
        return results
    
    def calculate_readability(self, text: str) -> float:
        """Calculate simple readability score"""
        
        words = text.split()
        if not words:
            return 0.0
        
        score = 1.0
        
        # Penalize for very short words (likely corrupted)
        very_short = sum(1 for w in words if len(w) <= 2 and w.isalpha())
        score -= (very_short / len(words)) * 0.3
        
        # Penalize for nonsense sequences
        consonant_clusters = len(re.findall(r'[bcdfghjklmnpqrstvwxz]{3,}', text.lower()))
        score -= min(consonant_clusters * 0.1, 0.5)
        
        # Penalize for excessive repeated characters
        repeated_chars = len(re.findall(r'(.)\1{2,}', text))
        score -= min(repeated_chars * 0.1, 0.3)
        
        return max(0.0, score)
    
    def count_consecutive_errors(self, clean_words: List[str], corrupted_words: List[str]) -> int:
        """Count maximum consecutive errors in a row"""
        
        if len(clean_words) != len(corrupted_words):
            return len(clean_words)  # All considered errors if different lengths
        
        max_consecutive = 0
        current_consecutive = 0
        
        for clean_word, corrupted_word in zip(clean_words, corrupted_words):
            if clean_word != corrupted_word:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def is_corrupted_meaningless(self, corrupted: str) -> bool:
        """Check if corrupted text has become meaningless"""
        
        words = corrupted.split()
        if not words:
            return True
        
        # Check for excessive nonsense words
        nonsense_patterns = [
            r'^[bcdfghjklmnpqrstvwxz]{4,}$',  # All consonants
            r'^[aeiou]{4,}$',                 # All vowels
            r'^(.)\1{3,}$',                   # Repeated character
            r'^\w{1,2}$',                     # Very short non-function words
        ]
        
        nonsense_count = 0
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if any(re.match(pattern, word_clean) for pattern in nonsense_patterns):
                nonsense_count += 1
        
        # Fail if more than 30% of words are nonsense
        return nonsense_count / len(words) > 0.3
    
    def validate_dataset_distribution(self, examples: List) -> List[ValidationResult]:
        """Validate overall dataset distribution"""
        
        results = []
        
        if not examples:
            results.append(ValidationResult(
                ValidationLevel.FAIL,
                "dataset_empty",
                "Dataset is empty"
            ))
            return results
        
        # Helper function to get attribute from mixed types
        def get_attr(ex, attr, default="unknown"):
            if hasattr(ex, attr):
                return getattr(ex, attr)
            elif hasattr(ex, 'get'):
                return ex.get(attr, default)
            else:
                return default
        
        # Complexity distribution
        complexity_counts = Counter(get_attr(ex, "complexity") for ex in examples)
        total_examples = len(examples)
        
        for complexity, (min_target, max_target) in self.complexity_targets.items():
            actual_ratio = complexity_counts.get(complexity, 0) / total_examples
            
            if actual_ratio < min_target:
                results.append(ValidationResult(
                    ValidationLevel.WARNING,
                    "complexity_distribution",
                    f"{complexity} examples too low: {actual_ratio:.1%} < {min_target:.1%}"
                ))
            elif actual_ratio > max_target:
                results.append(ValidationResult(
                    ValidationLevel.WARNING,
                    "complexity_distribution", 
                    f"{complexity} examples too high: {actual_ratio:.1%} > {max_target:.1%}"
                ))
        
        # Error type distribution
        error_type_counts = Counter()
        for example in examples:
            error_types = get_attr(example, "error_types", [])
            if isinstance(error_types, list):
                for error_type in error_types:
                    error_type_counts[error_type] += 1
        
        for error_type, min_ratio in self.error_type_minimums.items():
            actual_ratio = error_type_counts.get(error_type, 0) / total_examples
            if actual_ratio < min_ratio:
                results.append(ValidationResult(
                    ValidationLevel.WARNING,
                    "error_type_distribution",
                    f"{error_type} errors too low: {actual_ratio:.1%} < {min_ratio:.1%}"
                ))
        
        # Domain distribution (should be reasonably balanced)
        domain_counts = Counter(get_attr(ex, "domain") for ex in examples)
        domain_ratios = {domain: count/total_examples for domain, count in domain_counts.items()}
        
        # Check for domain imbalance (no domain should be >60% or <5%)
        for domain, ratio in domain_ratios.items():
            if ratio > 0.60:
                results.append(ValidationResult(
                    ValidationLevel.WARNING,
                    "domain_imbalance",
                    f"{domain} domain too dominant: {ratio:.1%}"
                ))
            elif ratio < 0.05 and total_examples > 100:  # Only warn for small domains in large datasets
                results.append(ValidationResult(
                    ValidationLevel.WARNING,
                    "domain_imbalance",
                    f"{domain} domain too small: {ratio:.1%}"
                ))
        
        # Source distribution
        source_counts = Counter(get_attr(ex, "source") for ex in examples)
        if len(source_counts) < 3:  # Should have multiple generation sources
            results.append(ValidationResult(
                ValidationLevel.WARNING,
                "source_diversity",
                f"Only {len(source_counts)} generation sources used"
            ))
        
        return results
    
    def validate_dataset_quality(self, examples: List[Dict]) -> Dict:
        """Comprehensive dataset quality validation"""
        
        print(f"🔍 Validating quality of {len(examples):,} training examples...")
        print("=" * 60)
        
        # Reset validation results
        self.validation_results = []
        self.example_issues = defaultdict(list)
        
        # Validate individual examples
        print("📝 Validating individual examples...")
        failed_examples = []
        warning_examples = []
        
        for i, example in enumerate(examples):
            example_results = self.validate_individual_example(example, i)
            self.validation_results.extend(example_results)
            
            # Track examples with issues
            has_failure = any(r.level == ValidationLevel.FAIL for r in example_results)
            has_warning = any(r.level == ValidationLevel.WARNING for r in example_results)
            
            if has_failure:
                failed_examples.append(i)
            elif has_warning:
                warning_examples.append(i)
            
            for result in example_results:
                self.example_issues[i].append(result)
        
        # Validate dataset distribution
        print("📊 Validating dataset distribution...")
        distribution_results = self.validate_dataset_distribution(examples)
        self.validation_results.extend(distribution_results)
        
        # Compile results
        validation_stats = self.compile_validation_stats()
        
        print(f"\n✅ Validation complete!")
        print(f"   Total Examples: {len(examples):,}")
        print(f"   Failed Examples: {len(failed_examples):,} ({len(failed_examples)/len(examples):.1%})")
        print(f"   Warning Examples: {len(warning_examples):,} ({len(warning_examples)/len(examples):.1%})")
        print(f"   Clean Examples: {len(examples) - len(failed_examples) - len(warning_examples):,}")
        
        return {
            "validation_stats": validation_stats,
            "failed_examples": failed_examples,
            "warning_examples": warning_examples,
            "all_results": self.validation_results,
            "recommendations": self.generate_recommendations()
        }
    
    def compile_validation_stats(self) -> Dict:
        """Compile validation statistics"""
        
        stats = {
            "total_issues": len(self.validation_results),
            "by_level": Counter(r.level.value for r in self.validation_results),
            "by_category": Counter(r.category for r in self.validation_results),
            "examples_with_issues": len(self.example_issues),
            "most_common_issues": []
        }
        
        # Find most common issue types
        issue_counter = Counter(f"{r.category}: {r.message}" for r in self.validation_results)
        stats["most_common_issues"] = issue_counter.most_common(10)
        
        return stats
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Count issues by category
        category_counts = Counter(r.category for r in self.validation_results)
        fail_counts = Counter(r.category for r in self.validation_results if r.level == ValidationLevel.FAIL)
        
        # Generate specific recommendations
        if fail_counts["insufficient_change"] > 0:
            recommendations.append(
                "🔧 Increase corruption intensity to ensure meaningful changes in examples"
            )
        
        if fail_counts["excessive_change"] > 0:
            recommendations.append(
                "🔧 Reduce corruption intensity to maintain text readability"
            )
        
        if category_counts["consecutive_errors"] > 10:
            recommendations.append(
                "🔧 Distribute errors more evenly across sentences to improve naturalness"
            )
        
        if category_counts["low_readability"] > len(self.validation_results) * 0.1:
            recommendations.append(
                "🔧 Improve error patterns to maintain text readability"
            )
        
        if category_counts["complexity_distribution"] > 0:
            recommendations.append(
                "📊 Adjust generation targets to better balance complexity levels"
            )
        
        if category_counts["error_type_distribution"] > 0:
            recommendations.append(
                "📊 Increase generation of underrepresented error types"
            )
        
        if not recommendations:
            recommendations.append("✅ Dataset quality looks good! No major issues detected.")
        
        return recommendations
    
    def save_validation_report(self, validation_results: Dict, output_file: str):
        """Save detailed validation report"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📝 Validation report saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate training dataset quality")
    parser.add_argument('--dataset-file', required=True,
                       help='Path to training dataset (JSONL format)')
    parser.add_argument('--report-file', 
                       help='Output validation report file')
    parser.add_argument('--sample-size', type=int,
                       help='Validate only a sample of examples')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"📚 Loading dataset from {args.dataset_file}...")
    examples = []
    
    try:
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        print(f"✅ Loaded {len(examples):,} examples")
        
        # Sample if requested
        if args.sample_size and args.sample_size < len(examples):
            import random
            examples = random.sample(examples, args.sample_size)
            print(f"📊 Using sample of {len(examples):,} examples")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Validate dataset
    validator = QualityValidator()
    results = validator.validate_dataset_quality(examples)
    
    # Save report
    report_file = args.report_file or args.dataset_file.replace('.jsonl', '_validation_report.json')
    validator.save_validation_report(results, report_file)
    
    # Print summary
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"  Total Issues: {results['validation_stats']['total_issues']:,}")
    print(f"  Failures: {results['validation_stats']['by_level'].get('fail', 0):,}")
    print(f"  Warnings: {results['validation_stats']['by_level'].get('warning', 0):,}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()
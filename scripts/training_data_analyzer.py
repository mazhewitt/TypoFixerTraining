#!/usr/bin/env python3
"""
Training Data Analyzer for Distribution Validation

Provides comprehensive analysis and visualization of training data distributions
to ensure balanced and effective typo correction training datasets.
"""

import json
import re
import statistics
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class TrainingDataAnalyzer:
    """Comprehensive training data analyzer"""
    
    def __init__(self):
        self.examples = []
        self.analysis_results = {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile without numpy"""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100)
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    def load_dataset(self, dataset_file: str) -> int:
        """Load training dataset from JSONL file"""
        
        print(f"📚 Loading dataset from {dataset_file}...")
        self.examples = []
        
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
            
            print(f"✅ Loaded {len(self.examples):,} training examples")
            return len(self.examples)
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return 0
    
    def analyze_basic_statistics(self) -> Dict:
        """Analyze basic dataset statistics"""
        
        print("📊 Analyzing basic statistics...")
        
        if not self.examples:
            return {}
        
        # Extract basic metrics
        word_counts = [ex.get("word_count", len(ex.get("clean", "").split())) for ex in self.examples]
        char_counts = [ex.get("char_count", len(ex.get("clean", ""))) for ex in self.examples]
        num_errors = [ex.get("num_errors", 0) for ex in self.examples]
        difficulty_scores = [ex.get("difficulty_score", 0) for ex in self.examples if ex.get("difficulty_score")]
        
        stats = {
            "total_examples": len(self.examples),
            "word_count_stats": {
                "mean": statistics.mean(word_counts),
                "median": statistics.median(word_counts),
                "std": statistics.stdev(word_counts) if len(word_counts) > 1 else 0,
                "min": min(word_counts),
                "max": max(word_counts),
                "percentiles": {
                    "25th": np.percentile(word_counts, 25),
                    "75th": np.percentile(word_counts, 75),
                    "90th": np.percentile(word_counts, 90),
                    "95th": np.percentile(word_counts, 95)
                }
            },
            "char_count_stats": {
                "mean": statistics.mean(char_counts),
                "median": statistics.median(char_counts),
                "std": statistics.stdev(char_counts) if len(char_counts) > 1 else 0,
                "min": min(char_counts),
                "max": max(char_counts)
            },
            "error_count_stats": {
                "mean": statistics.mean(num_errors),
                "median": statistics.median(num_errors),
                "std": statistics.stdev(num_errors) if len(num_errors) > 1 else 0,
                "min": min(num_errors),
                "max": max(num_errors),
                "distribution": Counter(num_errors)
            }
        }
        
        if difficulty_scores:
            stats["difficulty_stats"] = {
                "mean": statistics.mean(difficulty_scores),
                "median": statistics.median(difficulty_scores),
                "std": statistics.stdev(difficulty_scores) if len(difficulty_scores) > 1 else 0,
                "min": min(difficulty_scores),
                "max": max(difficulty_scores)
            }
        
        return stats
    
    def analyze_distributions(self) -> Dict:
        """Analyze key dataset distributions"""
        
        print("📈 Analyzing distributions...")
        
        distributions = {
            "complexity": Counter(ex.get("complexity", "unknown") for ex in self.examples),
            "domain": Counter(ex.get("domain", "unknown") for ex in self.examples),
            "source": Counter(ex.get("source", "unknown") for ex in self.examples),
            "error_types": Counter(),
            "word_count_bins": Counter(),
            "error_count_bins": Counter()
        }
        
        # Count error types (examples can have multiple)
        for example in self.examples:
            error_types = example.get("error_types", [])
            for error_type in error_types:
                distributions["error_types"][error_type] += 1
        
        # Create word count bins
        for example in self.examples:
            word_count = example.get("word_count", len(example.get("clean", "").split()))
            if word_count <= 5:
                bin_name = "≤5 words"
            elif word_count <= 10:
                bin_name = "6-10 words"
            elif word_count <= 15:
                bin_name = "11-15 words"
            elif word_count <= 20:
                bin_name = "16-20 words"
            else:
                bin_name = ">20 words"
            distributions["word_count_bins"][bin_name] += 1
        
        # Create error count bins
        for example in self.examples:
            error_count = example.get("num_errors", 0)
            if error_count == 1:
                bin_name = "1 error"
            elif error_count == 2:
                bin_name = "2 errors"
            elif error_count <= 4:
                bin_name = "3-4 errors"
            elif error_count <= 6:
                bin_name = "5-6 errors"
            else:
                bin_name = ">6 errors"
            distributions["error_count_bins"][bin_name] += 1
        
        return distributions
    
    def analyze_error_patterns(self) -> Dict:
        """Analyze error patterns and corruption quality"""
        
        print("🔍 Analyzing error patterns...")
        
        patterns = {
            "common_corruptions": Counter(),
            "corruption_types": Counter(),
            "word_corruption_ratio": [],
            "char_corruption_ratio": [],
            "readability_issues": 0,
            "meaningless_corruptions": 0
        }
        
        for example in self.examples:
            clean = example.get("clean", "")
            corrupted = example.get("corrupted", "")
            
            if not clean or not corrupted:
                continue
            
            clean_words = clean.split()
            corrupted_words = corrupted.split()
            
            # Word-level corruption analysis
            if len(clean_words) == len(corrupted_words):
                word_changes = sum(1 for c, w in zip(clean_words, corrupted_words) if c != w)
                word_corruption_ratio = word_changes / len(clean_words) if clean_words else 0
                patterns["word_corruption_ratio"].append(word_corruption_ratio)
                
                # Identify specific corruption patterns
                for clean_word, corr_word in zip(clean_words, corrupted_words):
                    if clean_word != corr_word:
                        corruption_key = f"{clean_word} → {corr_word}"
                        patterns["common_corruptions"][corruption_key] += 1
                        
                        # Categorize corruption type
                        if self.is_keyboard_error(clean_word, corr_word):
                            patterns["corruption_types"]["keyboard"] += 1
                        elif self.is_phonetic_error(clean_word, corr_word):
                            patterns["corruption_types"]["phonetic"] += 1
                        elif self.is_character_operation(clean_word, corr_word):
                            patterns["corruption_types"]["character_operation"] += 1
                        else:
                            patterns["corruption_types"]["other"] += 1
            
            # Character-level corruption analysis
            char_changes = sum(1 for c, w in zip(clean, corrupted) if c != w)
            char_corruption_ratio = char_changes / len(clean) if clean else 0
            patterns["char_corruption_ratio"].append(char_corruption_ratio)
            
            # Quality checks
            if self.has_readability_issues(corrupted):
                patterns["readability_issues"] += 1
            
            if self.is_meaningless_corruption(corrupted):
                patterns["meaningless_corruptions"] += 1
        
        # Calculate statistics for ratios
        if patterns["word_corruption_ratio"]:
            patterns["word_corruption_stats"] = {
                "mean": statistics.mean(patterns["word_corruption_ratio"]),
                "median": statistics.median(patterns["word_corruption_ratio"]),
                "std": statistics.stdev(patterns["word_corruption_ratio"]) if len(patterns["word_corruption_ratio"]) > 1 else 0
            }
        
        if patterns["char_corruption_ratio"]:
            patterns["char_corruption_stats"] = {
                "mean": statistics.mean(patterns["char_corruption_ratio"]),
                "median": statistics.median(patterns["char_corruption_ratio"]),
                "std": statistics.stdev(patterns["char_corruption_ratio"]) if len(patterns["char_corruption_ratio"]) > 1 else 0
            }
        
        return patterns
    
    def is_keyboard_error(self, clean_word: str, corrupted_word: str) -> bool:
        """Check if corruption looks like keyboard error"""
        if len(clean_word) != len(corrupted_word):
            return False
        
        # Simple keyboard layout check
        keyboard_neighbors = {
            'q': 'wa', 'w': 'qeas', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg', 
            'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
            'a': 'qws', 's': 'awde', 'd': 'serf', 'f': 'drtg', 'g': 'ftyh',
            'h': 'gyuj', 'j': 'huik', 'k': 'jiol', 'l': 'kop',
            'z': 'ax', 'x': 'zsc', 'c': 'xdf', 'v': 'cfb', 'b': 'vgn',
            'n': 'bhm', 'm': 'nj'
        }
        
        diff_count = sum(1 for c, w in zip(clean_word.lower(), corrupted_word.lower()) if c != w)
        if diff_count == 1:
            for c, w in zip(clean_word.lower(), corrupted_word.lower()):
                if c != w and c in keyboard_neighbors and w in keyboard_neighbors[c]:
                    return True
        
        return False
    
    def is_phonetic_error(self, clean_word: str, corrupted_word: str) -> bool:
        """Check if corruption looks like phonetic error"""
        # Common phonetic substitutions
        phonetic_pairs = [
            ('ph', 'f'), ('f', 'ph'), ('c', 's'), ('s', 'c'),
            ('i', 'y'), ('y', 'i'), ('tion', 'sion'), ('sion', 'tion')
        ]
        
        clean_lower = clean_word.lower()
        corr_lower = corrupted_word.lower()
        
        for pattern, replacement in phonetic_pairs:
            if pattern in clean_lower and replacement in corr_lower:
                return True
            
        return False
    
    def is_character_operation(self, clean_word: str, corrupted_word: str) -> bool:
        """Check if corruption is character insertion/deletion/substitution"""
        len_diff = abs(len(clean_word) - len(corrupted_word))
        return len_diff <= 2  # Allow for insertions/deletions
    
    def has_readability_issues(self, text: str) -> bool:
        """Check if text has readability issues"""
        # Check for excessive consonant clusters
        consonant_clusters = len(re.findall(r'[bcdfghjklmnpqrstvwxz]{4,}', text.lower()))
        
        # Check for repeated characters
        repeated_chars = len(re.findall(r'(.)\1{3,}', text))
        
        return consonant_clusters > 0 or repeated_chars > 0
    
    def is_meaningless_corruption(self, text: str) -> bool:
        """Check if corruption has made text meaningless"""
        words = text.split()
        if not words:
            return True
        
        # Count very short words that might be corrupted beyond recognition
        very_short_words = sum(1 for w in words if len(re.sub(r'[^\w]', '', w)) <= 2 and w.isalpha())
        
        return very_short_words / len(words) > 0.4
    
    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """Create comprehensive visualizations"""
            
        print(f"📊 Creating visualizations in {output_dir}/...")
        Path(output_dir).mkdir(exist_ok=True)
        
        # Basic statistics distributions
        basic_stats = self.analysis_results.get("basic_statistics", {})
        distributions = self.analysis_results.get("distributions", {})
        
        # 1. Word count distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        word_counts = [ex.get("word_count", len(ex.get("clean", "").split())) for ex in self.examples]
        plt.hist(word_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Word Count Distribution')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        
        # 2. Error count distribution
        plt.subplot(2, 3, 2)
        error_counts = [ex.get("num_errors", 0) for ex in self.examples]
        plt.hist(error_counts, bins=max(error_counts)+1, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Error Count Distribution')
        plt.xlabel('Number of Errors')
        plt.ylabel('Frequency')
        
        # 3. Complexity distribution
        plt.subplot(2, 3, 3)
        complexity_data = distributions.get("complexity", Counter())
        plt.pie(complexity_data.values(), labels=complexity_data.keys(), autopct='%1.1f%%')
        plt.title('Complexity Distribution')
        
        # 4. Domain distribution
        plt.subplot(2, 3, 4)
        domain_data = distributions.get("domain", Counter())
        plt.bar(domain_data.keys(), domain_data.values(), color='lightgreen', alpha=0.7)
        plt.title('Domain Distribution')
        plt.xlabel('Domain')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 5. Error type distribution
        plt.subplot(2, 3, 5)
        error_type_data = distributions.get("error_types", Counter())
        plt.bar(error_type_data.keys(), error_type_data.values(), color='orange', alpha=0.7)
        plt.title('Error Type Distribution')
        plt.xlabel('Error Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 6. Source distribution
        plt.subplot(2, 3, 6)
        source_data = distributions.get("source", Counter())
        plt.bar(source_data.keys(), source_data.values(), color='purple', alpha=0.7)
        plt.title('Source Distribution')
        plt.xlabel('Source')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distribution_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Difficulty score distribution (if available)
        difficulty_scores = [ex.get("difficulty_score", 0) for ex in self.examples if ex.get("difficulty_score")]
        if difficulty_scores:
            plt.figure(figsize=(10, 6))
            plt.hist(difficulty_scores, bins=30, alpha=0.7, color='gold', edgecolor='black')
            plt.title('Difficulty Score Distribution')
            plt.xlabel('Difficulty Score')
            plt.ylabel('Frequency')
            plt.axvline(statistics.mean(difficulty_scores), color='red', linestyle='--', 
                       label=f'Mean: {statistics.mean(difficulty_scores):.1f}')
            plt.legend()
            plt.savefig(f"{output_dir}/difficulty_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Corruption ratio analysis
        error_patterns = self.analysis_results.get("error_patterns", {})
        if "word_corruption_ratio" in error_patterns:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(error_patterns["word_corruption_ratio"], bins=20, alpha=0.7, 
                    color='lightblue', edgecolor='black')
            plt.title('Word Corruption Ratio Distribution')
            plt.xlabel('Ratio of Words Changed')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            plt.hist(error_patterns["char_corruption_ratio"], bins=20, alpha=0.7,
                    color='lightcoral', edgecolor='black')
            plt.title('Character Corruption Ratio Distribution')
            plt.xlabel('Ratio of Characters Changed')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/corruption_ratios.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualizations saved to {output_dir}/")
    
    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        
        print("📝 Generating comprehensive analysis report...")
        
        # Perform all analyses
        basic_stats = self.analyze_basic_statistics()
        distributions = self.analyze_distributions()
        error_patterns = self.analyze_error_patterns()
        
        # Store results
        self.analysis_results = {
            "basic_statistics": basic_stats,
            "distributions": distributions,
            "error_patterns": error_patterns,
            "quality_metrics": self.calculate_quality_metrics()
        }
        
        return self.analysis_results
    
    def calculate_quality_metrics(self) -> Dict:
        """Calculate overall dataset quality metrics"""
        
        metrics = {
            "dataset_size": len(self.examples),
            "balance_scores": {},
            "diversity_scores": {},
            "quality_flags": []
        }
        
        # Calculate balance scores for key distributions
        distributions = self.analysis_results.get("distributions", {})
        
        for dist_name, dist_data in distributions.items():
            if isinstance(dist_data, Counter) and dist_data:
                # Calculate entropy as diversity measure
                total = sum(dist_data.values())
                probabilities = [count/total for count in dist_data.values()]
                entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                max_entropy = np.log2(len(dist_data))
                
                balance_score = entropy / max_entropy if max_entropy > 0 else 0
                metrics["balance_scores"][dist_name] = balance_score
                
                # Diversity score (number of unique categories)
                metrics["diversity_scores"][dist_name] = len(dist_data)
        
        # Quality flags
        error_patterns = self.analysis_results.get("error_patterns", {})
        
        if error_patterns.get("readability_issues", 0) > len(self.examples) * 0.05:
            metrics["quality_flags"].append("High number of readability issues")
        
        if error_patterns.get("meaningless_corruptions", 0) > len(self.examples) * 0.02:
            metrics["quality_flags"].append("High number of meaningless corruptions")
        
        basic_stats = self.analysis_results.get("basic_statistics", {})
        error_stats = basic_stats.get("error_count_stats", {})
        
        if error_stats.get("mean", 0) < 1.5:
            metrics["quality_flags"].append("Low average error count per example")
        
        if error_stats.get("mean", 0) > 5:
            metrics["quality_flags"].append("High average error count per example")
        
        return metrics
    
    def save_analysis_report(self, output_file: str):
        """Save comprehensive analysis report to file"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 Analysis report saved to {output_file}")
    
    def print_summary_report(self):
        """Print summary of analysis results"""
        
        if not self.analysis_results:
            print("❌ No analysis results available. Run generate_analysis_report() first.")
            return
        
        print("\n" + "="*60)
        print("📊 TRAINING DATA ANALYSIS SUMMARY")
        print("="*60)
        
        basic_stats = self.analysis_results["basic_statistics"]
        distributions = self.analysis_results["distributions"]
        quality_metrics = self.analysis_results["quality_metrics"]
        
        # Basic statistics
        print(f"\n📈 Basic Statistics:")
        print(f"  Total Examples: {basic_stats['total_examples']:,}")
        print(f"  Average Words: {basic_stats['word_count_stats']['mean']:.1f}")
        print(f"  Average Errors: {basic_stats['error_count_stats']['mean']:.1f}")
        
        if "difficulty_stats" in basic_stats:
            print(f"  Average Difficulty: {basic_stats['difficulty_stats']['mean']:.1f}")
        
        # Distribution balance
        print(f"\n⚖️  Distribution Balance Scores (0.0-1.0):")
        for dist_name, score in quality_metrics["balance_scores"].items():
            print(f"  {dist_name:15}: {score:.3f}")
        
        # Top categories
        print(f"\n🏆 Top Categories:")
        for dist_name, dist_data in distributions.items():
            if isinstance(dist_data, Counter):
                top_item = dist_data.most_common(1)[0] if dist_data else ("None", 0)
                total = sum(dist_data.values())
                print(f"  {dist_name:15}: {top_item[0]} ({top_item[1]:,}, {top_item[1]/total:.1%})")
        
        # Quality flags
        if quality_metrics["quality_flags"]:
            print(f"\n⚠️  Quality Concerns:")
            for flag in quality_metrics["quality_flags"]:
                print(f"  • {flag}")
        else:
            print(f"\n✅ No major quality concerns detected!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze training dataset comprehensively")
    parser.add_argument('--dataset-file', required=True,
                       help='Path to training dataset (JSONL format)')
    parser.add_argument('--output-dir', default='analysis_output',
                       help='Output directory for reports and plots')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer and load data
    analyzer = TrainingDataAnalyzer()
    
    if not analyzer.load_dataset(args.dataset_file):
        return
    
    # Generate analysis
    results = analyzer.generate_analysis_report()
    
    # Print summary
    analyzer.print_summary_report()
    
    # Save detailed report
    report_file = output_dir / "analysis_report.json"
    analyzer.save_analysis_report(str(report_file))
    
    # Create visualizations if requested
    if args.create_plots:
        plot_dir = output_dir / "plots"
        analyzer.create_visualizations(str(plot_dir))
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
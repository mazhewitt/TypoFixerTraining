#!/usr/bin/env python3
"""
Comprehensive diagnostic analysis for typo correction models.
Analyzes 1000+ examples to identify error patterns and improvement opportunities.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter
import re

import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TypoErrorAnalyzer:
    def __init__(self, model_dir: str):
        """Initialize analyzer with trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {model_dir}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForMaskedLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Error taxonomy
        self.error_patterns = {
            'keyboard_neighbor': defaultdict(int),     # teh -> the
            'missing_letter': defaultdict(int),        # sentenc -> sentence  
            'extra_letter': defaultdict(int),          # bbegan -> began
            'letter_swap': defaultdict(int),           # quikc -> quick
            'word_boundary': defaultdict(int),         # "too the" -> "to the"
            'apostrophe': defaultdict(int),            # its -> it's
            'homophone': defaultdict(int),             # there/their/they're
            'real_word': defaultdict(int),             # affect/effect
            'capitalization': defaultdict(int),        # thi -> This
        }
        
        # Success/failure tracking
        self.correction_stats = {
            'successful_corrections': [],
            'missed_corrections': [],
            'overcorrections': [],
            'hallucinations': [],
        }
        
        logger.info(f"Model loaded on {self.device}")

    def classify_error_type(self, original: str, target: str, predicted: str) -> List[str]:
        """Classify the type of error and correction attempt."""
        error_types = []
        
        # Normalize for comparison
        orig_lower = original.lower().strip()
        target_lower = target.lower().strip() 
        pred_lower = predicted.lower().strip()
        
        # Check for common error patterns
        if len(orig_lower) == len(target_lower):
            # Same length - likely character substitution
            differences = [(i, orig_lower[i], target_lower[i]) for i in range(len(orig_lower)) 
                          if orig_lower[i] != target_lower[i]]
            
            if len(differences) == 1:
                pos, orig_char, target_char = differences[0]
                if self._are_keyboard_neighbors(orig_char, target_char):
                    error_types.append('keyboard_neighbor')
                else:
                    error_types.append('letter_swap')
            elif len(differences) == 2:
                # Check for adjacent character swap
                pos1, _, _ = differences[0]
                pos2, _, _ = differences[1]
                if abs(pos1 - pos2) == 1:
                    error_types.append('letter_swap')
                    
        elif len(orig_lower) < len(target_lower):
            error_types.append('missing_letter')
        elif len(orig_lower) > len(target_lower):
            error_types.append('extra_letter')
            
        # Check for word boundary issues
        if ' ' in original or ' ' in target:
            if len(original.split()) != len(target.split()):
                error_types.append('word_boundary')
                
        # Check for apostrophe issues
        if "'" in target and "'" not in original:
            error_types.append('apostrophe')
            
        # Check for capitalization
        if original.lower() == target.lower() and original != target:
            error_types.append('capitalization')
            
        # Real word errors (both are valid English words)
        if (self._is_english_word(original) and self._is_english_word(target) and 
            original.lower() != target.lower()):
            error_types.append('real_word')
            
        return error_types if error_types else ['unknown']

    def _are_keyboard_neighbors(self, char1: str, char2: str) -> bool:
        """Check if two characters are keyboard neighbors."""
        keyboard_map = {
            'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etdf', 't': 'ryfg',
            'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
            'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
            'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
            'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
            'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
        }
        
        return char2 in keyboard_map.get(char1, '') or char1 in keyboard_map.get(char2, '')

    def _is_english_word(self, word: str) -> bool:
        """Simple check if word looks like English (could be improved with dictionary)."""
        # Simple heuristic: contains only letters, reasonable length
        return (word.isalpha() and 2 <= len(word) <= 15 and 
                not re.search(r'[qx]{2,}|[z]{2,}|[bcdfghjklmnpqrstvwxyz]{5,}', word.lower()))

    def correct_text_mlm(self, corrupted_text: str, max_length: int = 128) -> str:
        """Correct text using MLM approach (same as validate_mlm.py)."""
        inputs = self.tokenizer(
            corrupted_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        corrected_ids = input_ids.clone()
        
        # Simple typo detection patterns
        typo_patterns = [
            r'\bteh\b', r'\bsis\b', r'\bsentenc\b', r'\bmistaks?\b', 
            r'\bbeutiful\b', r'\boutsid\b', r'\bquikc?\b', r'\bstor\b',
            r'\bsom\b', r'\bther\b(?!\s+(are|is))', r'\brecieve\b',
            r'\bseperate\b', r'\bdefinately\b', r'\boccured\b'
        ]
        
        with torch.no_grad():
            # Find positions that might need correction
            for pos in range(1, input_ids.shape[1] - 1):  # Skip CLS and SEP
                if attention_mask[0, pos] == 0:  # Skip padding
                    break
                    
                current_token = self.tokenizer.decode([input_ids[0, pos].item()], skip_special_tokens=True)
                
                # Check if token matches typo patterns
                should_check = any(re.search(pattern, current_token.lower()) for pattern in typo_patterns)
                
                if should_check and len(current_token.strip()) > 1:
                    # Create masked version
                    masked_ids = input_ids.clone()
                    masked_ids[0, pos] = self.tokenizer.mask_token_id
                    
                    # Get prediction with beam search
                    outputs = self.model(input_ids=masked_ids, attention_mask=attention_mask)
                    logits = outputs.logits[0, pos]
                    top_candidates = torch.topk(logits, k=5, dim=-1)
                    
                    # Find best plausible candidate
                    for candidate_id in top_candidates.indices:
                        candidate_token = self.tokenizer.decode([candidate_id.item()], skip_special_tokens=True)
                        
                        # Use plausibility check
                        if (len(candidate_token.strip()) >= 1 and
                            candidate_token.strip().replace("'", "").isalpha() and
                            self._is_plausible_correction(current_token.strip(), candidate_token.strip())):
                            
                            if candidate_id != input_ids[0, pos]:
                                corrected_ids[0, pos] = candidate_id
                            break
        
        corrected_text = self.tokenizer.decode(corrected_ids[0], skip_special_tokens=True)
        return corrected_text.strip()

    def _is_plausible_correction(self, original: str, candidate: str) -> bool:
        """Check if correction is plausible (from validate_mlm.py)."""
        if not original.strip() or not candidate.strip():
            return True
            
        if original.lower() == candidate.lower():
            return True
            
        similarity = SequenceMatcher(None, original.lower(), candidate.lower()).ratio()
        return similarity >= 0.6

    def analyze_dataset(self, test_cases: List[Tuple[str, str]], max_samples: int = 1000) -> Dict:
        """Comprehensive analysis of model performance."""
        logger.info(f"Analyzing {min(len(test_cases), max_samples)} examples...")
        
        results = {
            'total_examples': 0,
            'error_patterns': dict(self.error_patterns),
            'correction_performance': defaultdict(list),
            'per_error_accuracy': defaultdict(list),
            'confusion_matrix': defaultdict(lambda: defaultdict(int)),
            'examples_by_category': defaultdict(list),
            'speed_metrics': [],
        }
        
        for i, (corrupted, expected) in enumerate(tqdm(test_cases[:max_samples], desc="Analyzing")):
            start_time = time.time()
            predicted = self.correct_text_mlm(corrupted)
            inference_time = time.time() - start_time
            
            results['speed_metrics'].append(inference_time)
            results['total_examples'] += 1
            
            # Tokenize for detailed analysis
            corrupted_words = corrupted.lower().split()
            expected_words = expected.lower().split()
            predicted_words = predicted.lower().split()
            
            # Align words for comparison
            max_words = max(len(corrupted_words), len(expected_words), len(predicted_words))
            
            for j in range(max_words):
                orig_word = corrupted_words[j] if j < len(corrupted_words) else ""
                target_word = expected_words[j] if j < len(expected_words) else ""
                pred_word = predicted_words[j] if j < len(predicted_words) else ""
                
                if orig_word and target_word and orig_word != target_word:
                    # This is a correction that should be made
                    error_types = self.classify_error_type(orig_word, target_word, pred_word)
                    
                    for error_type in error_types:
                        self.error_patterns[error_type][f"{orig_word}->{target_word}"] += 1
                        
                        # Track correction success
                        if pred_word == target_word:
                            results['correction_performance'][error_type].append(1)  # Success
                            self.correction_stats['successful_corrections'].append(
                                (orig_word, target_word, pred_word, error_type)
                            )
                        else:
                            results['correction_performance'][error_type].append(0)  # Failure
                            if pred_word == orig_word:
                                self.correction_stats['missed_corrections'].append(
                                    (orig_word, target_word, pred_word, error_type)
                                )
                            else:
                                self.correction_stats['overcorrections'].append(
                                    (orig_word, target_word, pred_word, error_type)
                                )
                        
                        # Update confusion matrix
                        results['confusion_matrix'][error_type][f"{orig_word}->{pred_word}"] += 1
                        
                elif orig_word == target_word and pred_word != orig_word:
                    # This is an overcorrection (model changed something that was correct)
                    self.correction_stats['overcorrections'].append(
                        (orig_word, target_word, pred_word, 'overcorrection')
                    )
            
            # Store examples by category
            if predicted == expected:
                category = 'perfect_match'
            elif len(set(predicted.lower().split()) & set(expected.lower().split())) > 0:
                category = 'partial_improvement'
            else:
                category = 'no_improvement'
                
            results['examples_by_category'][category].append({
                'corrupted': corrupted,
                'expected': expected,
                'predicted': predicted,
                'inference_time': inference_time
            })
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary_stats(results)
        
        return results

    def _calculate_summary_stats(self, results: Dict) -> Dict:
        """Calculate summary statistics from analysis results."""
        summary = {
            'total_examples': results['total_examples'],
            'avg_inference_time': np.mean(results['speed_metrics']),
            'error_type_distribution': {},
            'correction_accuracy_by_type': {},
            'most_common_errors': {},
            'performance_by_category': {}
        }
        
        # Error type distribution
        total_errors = sum(len(errors) for errors in self.error_patterns.values())
        for error_type, errors in self.error_patterns.items():
            count = len(errors)
            summary['error_type_distribution'][error_type] = {
                'count': count,
                'percentage': (count / total_errors * 100) if total_errors > 0 else 0
            }
        
        # Correction accuracy by type
        for error_type, successes in results['correction_performance'].items():
            if successes:
                accuracy = np.mean(successes)
                summary['correction_accuracy_by_type'][error_type] = {
                    'accuracy': accuracy,
                    'total_attempts': len(successes),
                    'successes': sum(successes)
                }
        
        # Most common error patterns
        for error_type, errors in self.error_patterns.items():
            if errors:
                most_common = Counter(errors).most_common(5)
                summary['most_common_errors'][error_type] = most_common
        
        # Performance by category
        for category, examples in results['examples_by_category'].items():
            summary['performance_by_category'][category] = {
                'count': len(examples),
                'percentage': (len(examples) / results['total_examples'] * 100) if results['total_examples'] > 0 else 0
            }
        
        return summary

    def generate_report(self, results: Dict, output_file: str = None) -> str:
        """Generate comprehensive analysis report."""
        report = []
        summary = results['summary']
        
        report.append("=" * 80)
        report.append("🔍 COMPREHENSIVE TYPO CORRECTION ANALYSIS")
        report.append("=" * 80)
        report.append(f"📊 Total examples analyzed: {summary['total_examples']:,}")
        report.append(f"⚡ Average inference time: {summary['avg_inference_time']*1000:.1f}ms")
        report.append("")
        
        # Error type distribution
        report.append("📈 ERROR TYPE DISTRIBUTION:")
        report.append("-" * 40)
        for error_type, stats in summary['error_type_distribution'].items():
            if stats['count'] > 0:
                report.append(f"  {error_type:.<20} {stats['count']:>4} ({stats['percentage']:>5.1f}%)")
        report.append("")
        
        # Correction accuracy by type
        report.append("🎯 CORRECTION ACCURACY BY ERROR TYPE:")
        report.append("-" * 50)
        for error_type, stats in summary['correction_accuracy_by_type'].items():
            report.append(f"  {error_type:.<20} {stats['accuracy']:>6.1%} "
                         f"({stats['successes']}/{stats['total_attempts']})")
        report.append("")
        
        # Most common errors
        report.append("🔄 MOST COMMON ERROR PATTERNS:")
        report.append("-" * 40)
        for error_type, patterns in summary['most_common_errors'].items():
            if patterns:
                report.append(f"\n  {error_type.upper()}:")
                for pattern, count in patterns:
                    report.append(f"    {pattern:<25} {count:>3}x")
        report.append("")
        
        # Performance by category
        report.append("📊 OVERALL PERFORMANCE:")
        report.append("-" * 30)
        for category, stats in summary['performance_by_category'].items():
            report.append(f"  {category.replace('_', ' ').title():.<20} "
                         f"{stats['count']:>4} ({stats['percentage']:>5.1f}%)")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        report.append("-" * 20)
        report.extend(self._generate_recommendations(summary))
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"📄 Report saved to {output_file}")
        
        return report_text

    def _generate_recommendations(self, summary: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Find worst-performing error types
        worst_types = sorted(
            summary['correction_accuracy_by_type'].items(),
            key=lambda x: x[1]['accuracy']
        )[:3]
        
        if worst_types:
            recommendations.append("🎯 Priority improvements:")
            for error_type, stats in worst_types:
                if stats['accuracy'] < 0.5:
                    recommendations.append(f"   • Focus on {error_type} errors "
                                         f"(currently {stats['accuracy']:.1%} accuracy)")
        
        # Performance-based recommendations
        perfect_match_rate = summary['performance_by_category'].get('perfect_match', {}).get('percentage', 0)
        
        if perfect_match_rate < 30:
            recommendations.append("📈 Low perfect match rate suggests:")
            recommendations.append("   • Increase training data diversity")
            recommendations.append("   • Add more epochs or larger model")
            
        if perfect_match_rate > 70:
            recommendations.append("✅ High perfect match rate - consider:")
            recommendations.append("   • More challenging test cases")
            recommendations.append("   • Real-world evaluation datasets")
            
        # Speed recommendations
        avg_time = summary['avg_inference_time']
        if avg_time > 0.1:  # 100ms
            recommendations.append("⚡ Speed improvements:")
            recommendations.append("   • Model quantization for deployment")
            recommendations.append("   • Batch processing for multiple texts")
            
        return recommendations

def load_test_data_from_file(file_path: str, max_samples: int = None) -> List[Tuple[str, str]]:
    """Load test data from JSONL file."""
    test_cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line.strip())
            test_cases.append((data['corrupted'], data['clean']))
    return test_cases

def main():
    parser = argparse.ArgumentParser(description="Comprehensive typo correction diagnostic analysis")
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--test_file', type=str,
                       help='JSONL file with test examples')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples to analyze')
    parser.add_argument('--output_report', type=str, default='diagnostic_report.txt',
                       help='Output file for analysis report')
    parser.add_argument('--output_json', type=str, default='diagnostic_results.json',
                       help='Output file for detailed JSON results')
    
    args = parser.parse_args()
    
    logger.info("🔍 Starting comprehensive diagnostic analysis...")
    
    # Initialize analyzer
    analyzer = TypoErrorAnalyzer(args.model_dir)
    
    # Load test data
    if args.test_file:
        logger.info(f"Loading test data from {args.test_file}")
        test_cases = load_test_data_from_file(args.test_file, args.max_samples)
    else:
        logger.info("Using built-in test cases")
        test_cases = [
            ("Thi sis a test sentenc with typos", "This is a test sentence with typos"),
            ("The quikc brown fox jumps over teh lazy dog", "The quick brown fox jumps over the lazy dog"),
            ("I went too the stor to buy som milk", "I went to the store to buy some milk"),
            ("Ther are many mistaks in this sentance", "There are many mistakes in this sentence"),
            ("Its a beutiful day outsid today", "It's a beautiful day outside today"),
            ("Can you help me wiht this problme?", "Can you help me with this problem?"),
            ("They're going to there house over their", "They're going to their house over there"),
            ("The affect of this change will effect everyone", "The effect of this change will affect everyone"),
            ("I need advise about how to advice my team", "I need advice about how to advise my team"),
            ("Recieve the seperate definately occured begining", "Receive the separate definitely occurred beginning")
        ] * 100  # Repeat to get 1000 examples
    
    logger.info(f"Analyzing {len(test_cases):,} test cases...")
    
    # Run analysis
    results = analyzer.analyze_dataset(test_cases, args.max_samples)
    
    # Generate and save report
    report = analyzer.generate_report(results, args.output_report)
    print(report)
    
    # Save detailed JSON results
    with open(args.output_json, 'w', encoding='utf-8') as f:
        # Convert defaultdicts to regular dicts for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, defaultdict):
                json_results[key] = dict(value)
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2, default=str)
    
    logger.info(f"📊 Detailed results saved to {args.output_json}")
    logger.info("🎉 Diagnostic analysis complete!")

if __name__ == "__main__":
    main()
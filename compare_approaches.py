#!/usr/bin/env python3
"""
Compare the different correction approaches on the validation dataset.
"""

import logging
import sys
import os
sys.path.append('src')

from simple_two_stage import SimpleTypoCorrector
from typo_correction import load_pretrained_corrector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_correction_approaches():
    """Compare different correction approaches."""
    
    print("🔍 COMPARISON: Different Typo Correction Approaches")
    print("="*80)
    
    # Test cases from validation
    test_cases = [
        ("Thi sis a test sentenc with typos", "This is a test sentence with typos"),
        ("The quikc brown fox jumps over teh lazy dog", "The quick brown fox jumps over the lazy dog"),
        ("I went too the stor to buy som milk", "I went to the store to buy some milk"),
        ("Ther are many mistaks in this sentance", "There are many mistakes in this sentence"),
        ("Its a beutiful day outsid today", "It's a beautiful day outside today"),
        ("Can you help me wiht this problme?", "Can you help me with this problem?"),
        ("They're going to there house over their", "They're going to their house over there"),
        ("The affect of this change will effect everyone", "The effect of this change will affect everyone"),
    ]
    
    # Initialize correctors
    print("Initializing correctors...")
    
    # 1. Two-stage approach (spell checker + MLM)
    two_stage = SimpleTypoCorrector("distilbert-base-uncased")
    
    # 2. Single-stage approach (base DistilBERT)
    single_stage = load_pretrained_corrector("distilbert-base-uncased", 
                                           low_prob_threshold=-3.5, 
                                           edit_penalty_lambda=1.0)
    
    print("\nRunning comparisons...")
    print("-" * 80)
    
    two_stage_stats = {
        'total_cases': len(test_cases),
        'corrections_made': 0,
        'exact_matches': 0,
        'partial_improvements': 0,
        'total_tokens': 0,
        'correct_tokens': 0
    }
    
    single_stage_stats = {
        'total_cases': len(test_cases),
        'corrections_made': 0,
        'exact_matches': 0,
        'partial_improvements': 0,
        'total_tokens': 0,
        'correct_tokens': 0
    }
    
    for i, (corrupted, expected) in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Testing: '{corrupted}'")
        print(f"    Expected: '{expected}'")
        
        # Two-stage correction
        two_stage_result, two_stage_info = two_stage.correct_text(corrupted)
        print(f"    Two-stage: '{two_stage_result}' ({two_stage_info['total_corrections']} corrections)")
        
        # Single-stage correction  
        single_stage_result, single_stage_info = single_stage.correct_typos(corrupted)
        print(f"    Single-stage: '{single_stage_result}' ({single_stage_info['total_corrections']} corrections)")
        
        # Update two-stage stats
        if two_stage_info['total_corrections'] > 0:
            two_stage_stats['corrections_made'] += 1
        
        if two_stage_result.lower().strip() == expected.lower().strip():
            two_stage_stats['exact_matches'] += 1
        
        # Check partial improvements for two-stage
        orig_tokens = corrupted.lower().split()
        pred_tokens = two_stage_result.lower().split()
        exp_tokens = expected.lower().split()
        
        improvements = 0
        for j in range(min(len(pred_tokens), len(exp_tokens), len(orig_tokens))):
            if pred_tokens[j] == exp_tokens[j] and orig_tokens[j] != exp_tokens[j]:
                improvements += 1
        
        if improvements > 0:
            two_stage_stats['partial_improvements'] += 1
        
        # Token accuracy for two-stage
        max_len = max(len(pred_tokens), len(exp_tokens))
        for j in range(max_len):
            two_stage_stats['total_tokens'] += 1
            pred_token = pred_tokens[j] if j < len(pred_tokens) else ""
            exp_token = exp_tokens[j] if j < len(exp_tokens) else ""
            if pred_token == exp_token:
                two_stage_stats['correct_tokens'] += 1
        
        # Update single-stage stats
        if single_stage_info['total_corrections'] > 0:
            single_stage_stats['corrections_made'] += 1
        
        if single_stage_result.lower().strip() == expected.lower().strip():
            single_stage_stats['exact_matches'] += 1
        
        # Check partial improvements for single-stage
        single_pred_tokens = single_stage_result.lower().split()
        
        improvements = 0
        for j in range(min(len(single_pred_tokens), len(exp_tokens), len(orig_tokens))):
            if single_pred_tokens[j] == exp_tokens[j] and orig_tokens[j] != exp_tokens[j]:
                improvements += 1
        
        if improvements > 0:
            single_stage_stats['partial_improvements'] += 1
        
        # Token accuracy for single-stage
        max_len = max(len(single_pred_tokens), len(exp_tokens))
        for j in range(max_len):
            single_stage_stats['total_tokens'] += 1
            pred_token = single_pred_tokens[j] if j < len(single_pred_tokens) else ""
            exp_token = exp_tokens[j] if j < len(exp_tokens) else ""
            if pred_token == exp_token:
                single_stage_stats['correct_tokens'] += 1
    
    # Calculate final metrics
    two_stage_stats['token_accuracy'] = two_stage_stats['correct_tokens'] / two_stage_stats['total_tokens']
    two_stage_stats['exact_match_rate'] = two_stage_stats['exact_matches'] / two_stage_stats['total_cases']
    two_stage_stats['partial_improvement_rate'] = two_stage_stats['partial_improvements'] / two_stage_stats['total_cases']
    two_stage_stats['correction_rate'] = two_stage_stats['corrections_made'] / two_stage_stats['total_cases']
    
    single_stage_stats['token_accuracy'] = single_stage_stats['correct_tokens'] / single_stage_stats['total_tokens']
    single_stage_stats['exact_match_rate'] = single_stage_stats['exact_matches'] / single_stage_stats['total_cases']
    single_stage_stats['partial_improvement_rate'] = single_stage_stats['partial_improvements'] / single_stage_stats['total_cases']
    single_stage_stats['correction_rate'] = single_stage_stats['corrections_made'] / single_stage_stats['total_cases']
    
    # Print comparison
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON RESULTS")
    print("="*80)
    
    print(f"{'Metric':<25} {'Two-Stage':<15} {'Single-Stage':<15} {'Improvement':<15}")
    print("-" * 70)
    
    metrics = [
        ('Token Accuracy', 'token_accuracy', '%'),
        ('Exact Match Rate', 'exact_match_rate', '%'),
        ('Partial Improvements', 'partial_improvement_rate', '%'),
        ('Correction Attempts', 'correction_rate', '%'),
    ]
    
    for name, key, unit in metrics:
        two_val = two_stage_stats[key] * 100
        single_val = single_stage_stats[key] * 100
        improvement = two_val - single_val
        
        print(f"{name:<25} {two_val:>10.1f}{unit:<4} {single_val:>10.1f}{unit:<4} {improvement:>+10.1f}{unit}")
    
    print("\n🎯 Summary:")
    if two_stage_stats['token_accuracy'] > single_stage_stats['token_accuracy']:
        print("✅ Two-stage approach shows better token accuracy")
    else:
        print("❌ Single-stage approach shows better token accuracy")
    
    if two_stage_stats['partial_improvement_rate'] > single_stage_stats['partial_improvement_rate']:
        print("✅ Two-stage approach makes more helpful corrections")
    else:
        print("❌ Single-stage approach makes more helpful corrections")
    
    print(f"\n💡 Two-stage detected {sum(len(two_stage.detect_typos(corrupted)) for corrupted, _ in test_cases)} total typos across all test cases")

if __name__ == "__main__":
    compare_correction_approaches()
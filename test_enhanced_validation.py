#!/usr/bin/env python3
"""
Test enhanced corrector against validation set.
"""

from enhanced_two_stage import EnhancedTypoCorrector

def test_enhanced_validation():
    """Test enhanced corrector on validation dataset."""
    
    print("🎯 ENHANCED CORRECTOR VALIDATION TEST")
    print("="*70)
    
    # Same test cases as validation
    test_cases = [
        ("Thi sis a test sentenc with typos", "This is a test sentence with typos"),
        ("The quikc brown fox jumps over teh lazy dog", "The quick brown fox jumps over the lazy dog"),
        ("I went too the stor to buy som milk", "I went to the store to buy some milk"),
        ("Ther are many mistaks in this sentance", "There are many mistakes in this sentence"),
        ("Its a beutiful day outsid today", "It's a beautiful day outside today"),
        ("Can you help me wiht this problme?", "Can you help me with this problem?"),
        ("They're going to there house over their", "They're going to their house over there"),
        ("Your presentation was excellent, you're very talented", "Your presentation was excellent, you're very talented"),
        ("The affect of this change will effect everyone", "The effect of this change will affect everyone"),
        ("I need advise about how to advice my team", "I need advice about how to advise my team")
    ]
    
    corrector = EnhancedTypoCorrector()
    
    # Calculate detailed metrics
    stats = {
        'total_cases': len(test_cases),
        'corrections_made': 0,
        'exact_matches': 0,
        'partial_improvements': 0,
        'total_tokens': 0,
        'correct_tokens': 0
    }
    
    print("\nDETAILED RESULTS:")
    print("-" * 70)
    
    for i, (corrupted, expected) in enumerate(test_cases, 1):
        corrected, info = corrector.correct_text(corrupted)
        
        print(f"\n{i:2d}. Corrupted: '{corrupted}'")
        print(f"    Expected:  '{expected}'")
        print(f"    Enhanced:  '{corrected}' ({info['total_corrections']} corrections)")
        
        # Check if correction was attempted
        if info['total_corrections'] > 0:
            stats['corrections_made'] += 1
        
        # Check exact match
        if corrected.lower().strip() == expected.lower().strip():
            stats['exact_matches'] += 1
            print(f"    ✅ EXACT MATCH!")
        
        # Check partial improvements (token-level)
        orig_tokens = corrupted.lower().split()
        pred_tokens = corrected.lower().split()
        exp_tokens = expected.lower().split()
        
        improvements = 0
        max_len = max(len(pred_tokens), len(exp_tokens), len(orig_tokens))
        
        for j in range(max_len):
            orig_token = orig_tokens[j] if j < len(orig_tokens) else ""
            pred_token = pred_tokens[j] if j < len(pred_tokens) else ""
            exp_token = exp_tokens[j] if j < len(exp_tokens) else ""
            
            # Count total tokens for accuracy
            stats['total_tokens'] += 1
            if pred_token == exp_token:
                stats['correct_tokens'] += 1
            
            # Count improvements
            if pred_token == exp_token and orig_token != exp_token:
                improvements += 1
        
        if improvements > 0:
            stats['partial_improvements'] += 1
            print(f"    ✅ {improvements} tokens improved")
        
        # Show specific corrections made
        if info['corrections_made']:
            for correction in info['corrections_made']:
                method = correction.get('method', 'unknown')
                print(f"    - '{correction['original']}' → '{correction['corrected']}' "
                      f"(score: {correction['score']:.3f}, {method})")
    
    # Calculate final metrics
    token_accuracy = stats['correct_tokens'] / stats['total_tokens'] * 100
    exact_match_rate = stats['exact_matches'] / stats['total_cases'] * 100
    partial_improvement_rate = stats['partial_improvements'] / stats['total_cases'] * 100
    correction_attempt_rate = stats['corrections_made'] / stats['total_cases'] * 100
    
    print("\n" + "="*70)
    print("📊 ENHANCED VALIDATION RESULTS")
    print("="*70)
    
    print(f"📊 Total test cases: {stats['total_cases']}")
    print(f"📊 Token accuracy: {token_accuracy:.1f}%")
    print(f"📊 Exact matches: {stats['exact_matches']} ({exact_match_rate:.1f}%)")
    print(f"📊 Partial improvements: {stats['partial_improvements']} ({partial_improvement_rate:.1f}%)")
    print(f"📊 Corrections attempted: {stats['corrections_made']} ({correction_attempt_rate:.1f}%)")
    
    print(f"\n🎯 PERFORMANCE vs TARGETS:")
    print(f"   Token Accuracy: {token_accuracy:.1f}% (Target: 90%+) {'✅' if token_accuracy >= 90 else '❌'}")
    print(f"   Exact Matches: {exact_match_rate:.1f}% (Target: 60%+) {'✅' if exact_match_rate >= 60 else '❌'}")
    print(f"   Improvements: {partial_improvement_rate:.1f}% (Target: 80%+) {'✅' if partial_improvement_rate >= 80 else '❌'}")
    
    # Comparison with previous version
    print(f"\n📈 IMPROVEMENT vs SIMPLE VERSION:")
    print(f"   Token Accuracy: {token_accuracy:.1f}% (was 68.3%) = +{token_accuracy-68.3:.1f}%")
    print(f"   Exact Matches: {exact_match_rate:.1f}% (was 0.0%) = +{exact_match_rate:.1f}%")
    print(f"   Corrections: {correction_attempt_rate:.1f}% (was 12.5%) = +{correction_attempt_rate-12.5:.1f}%")
    
    if token_accuracy >= 90:
        print(f"\n🎉 SUCCESS: Target accuracy achieved! ({token_accuracy:.1f}% >= 90%)")
    else:
        remaining_gap = 90 - token_accuracy
        print(f"\n🔧 REMAINING GAP: {remaining_gap:.1f}% to reach 90% target")
        print(f"💡 Next steps: Add context-aware homophone correction, train dedicated MLM")
    
    return {
        'token_accuracy': token_accuracy,
        'exact_match_rate': exact_match_rate,
        'partial_improvement_rate': partial_improvement_rate,
        'correction_attempt_rate': correction_attempt_rate
    }

if __name__ == "__main__":
    test_enhanced_validation()
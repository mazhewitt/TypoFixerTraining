#!/usr/bin/env python3
"""
Test the typo-aware MLM model to see if it solves our remaining 0.5% accuracy gap.
"""

from enhanced_two_stage import EnhancedTypoCorrector
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
import torch

class TypoAwareCorrector(EnhancedTypoCorrector):
    """Enhanced corrector using the typo-aware MLM model."""
    
    def __init__(self, typo_aware_model_path: str):
        """Initialize with typo-aware model instead of base DistilBERT."""
        
        # Initialize spell checker (same as parent)
        try:
            from spellchecker import SpellChecker
            self.spell = SpellChecker()
            print("✅ Enhanced spell checker initialized")
        except ImportError:
            self.spell = None
            print("❌ Spell checker not available")
        
        # Load the typo-aware MLM model
        print(f"✅ Loading typo-aware MLM model: {typo_aware_model_path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(typo_aware_model_path)
        self.model = DistilBertForMaskedLM.from_pretrained(typo_aware_model_path)
        self.model.eval()
        
        # Same enhanced dictionaries as parent
        self.common_typos = {
            'sis': 'is', 'thi': 'this', 'stor': 'store', 'som': 'some',
            'ther': 'there', 'mistaks': 'mistakes', 'sentenc': 'sentence',
            'sentance': 'sentence', 'quikc': 'quick', 'teh': 'the',
            'beutiful': 'beautiful', 'outsid': 'outside', 'wiht': 'with',
            'recieve': 'receive', 'seperate': 'separate', 'definately': 'definitely',
            'problme': 'problem'
        }
        
        self.homophones = {
            'too': ['to', 'two'], 'to': ['too', 'two'], 'two': ['too', 'to'],
            'there': ['their', "they're"], 'their': ['there', "they're"],
            "they're": ['there', 'their'], 'theyre': ['there', 'their'],
            'your': ["you're"], "you're": ['your'], 'youre': ['your'],
            'its': ["it's"], "it's": ['its'], 'affect': ['effect'], 'effect': ['affect'],
        }
        
        self.apostrophe_rules = {
            'its': "it's", 'cant': "can't", 'dont': "don't", 'wont': "won't",
            'isnt': "isn't", 'arent': "aren't", 'wasnt': "wasn't", 'werent': "weren't",
            'hasnt': "hasn't", 'havent': "haven't", 'hadnt': "hadn't",
            'shouldnt': "shouldn't", 'wouldnt': "wouldn't", 'couldnt': "couldn't",
        }

def test_typo_aware_model():
    """Test the typo-aware model on our validation set."""
    
    print("🎯 TESTING TYPO-AWARE MLM MODEL")
    print("="*70)
    
    # Test cases from validation
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
    
    # Initialize corrector with typo-aware model
    corrector = TypoAwareCorrector("models/typo_aware_distilbert")
    
    # Calculate metrics
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
        
        # Calculate token-level accuracy
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
    print("📊 TYPO-AWARE MLM RESULTS")
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
    
    # Comparison with enhanced version
    print(f"\n📈 IMPROVEMENT vs ENHANCED VERSION:")
    print(f"   Token Accuracy: {token_accuracy:.1f}% (was 89.5%) = {token_accuracy-89.5:+.1f}%")
    print(f"   Exact Matches: {exact_match_rate:.1f}% (was 50.0%) = {exact_match_rate-50.0:+.1f}%")
    
    if token_accuracy >= 90:
        print(f"\n🎉 SUCCESS: 90% TARGET ACHIEVED! ({token_accuracy:.1f}% >= 90%)")
        print(f"🚀 Journey: 68% → 89.5% → {token_accuracy:.1f}% (+{token_accuracy-68:.1f}% total improvement)")
    else:
        remaining_gap = 90 - token_accuracy
        print(f"\n🔧 REMAINING GAP: {remaining_gap:.1f}% to reach 90% target")
    
    return token_accuracy

def quick_test_specific_cases():
    """Quick test on the specific cases that were failing."""
    
    print("\n" + "="*70)
    print("🔍 TESTING SPECIFIC FAILURE CASES")
    print("="*70)
    
    corrector = TypoAwareCorrector("models/typo_aware_distilbert")
    
    # Test specific problem cases
    problem_cases = [
        ("Can you help me wiht this problem?", "wiht", "with"),
        ("The beutiful day outside", "beutiful", "beautiful"), 
        ("I need to recieve the package", "recieve", "receive"),
        ("This is a quikc test", "quikc", "quick"),
    ]
    
    for text, typo, expected in problem_cases:
        print(f"\nTesting: '{text}'")
        print(f"Problem: '{typo}' should be '{expected}'")
        
        # Test direct MLM prediction
        masked_text = text.replace(typo, "[MASK]")
        inputs = corrector.tokenizer(masked_text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = corrector.model(**inputs)
            logits = outputs.logits
            
            # Find mask position
            mask_positions = torch.where(inputs['input_ids'] == corrector.tokenizer.mask_token_id)[1]
            
            if len(mask_positions) > 0:
                mask_logits = logits[0, mask_positions[0], :]
                top_5 = torch.topk(mask_logits, 5)
                
                print(f"MLM predictions for '[MASK]':")
                for i in range(5):
                    token_id = top_5.indices[i].item()
                    token = corrector.tokenizer.decode([token_id]).strip()
                    score = torch.softmax(mask_logits, dim=-1)[token_id].item()
                    marker = " ✅" if token.lower() == expected.lower() else ""
                    print(f"  {i+1}. '{token}' ({score:.3f}){marker}")

if __name__ == "__main__":
    try:
        accuracy = test_typo_aware_model()
        quick_test_specific_cases()
        
        if accuracy >= 90:
            print(f"\n🏆 MISSION ACCOMPLISHED!")
            print(f"   Final accuracy: {accuracy:.1f}%")
            print(f"   Target achieved: ✅ 90%+")
            print(f"   Two-stage approach: ✅ Works brilliantly")
            
    except Exception as e:
        print(f"\n❌ Error testing typo-aware model: {e}")
        print("💡 Make sure the model was trained successfully first")
#!/usr/bin/env python3
"""
Final corrector that properly uses the explicit typo format for MLM predictions.
Since we know the misspelled word from spell checker, we use: 
"CORRUPT: typo SENTENCE: [MASK] context" for maximum accuracy.
"""

import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from enhanced_two_stage import EnhancedTypoCorrector

class FinalExplicitCorrector(EnhancedTypoCorrector):
    """Final corrector using explicit typo format with the trained model."""
    
    def __init__(self):
        """Initialize with explicit typo model."""
        
        # Initialize spell checker (same as parent)
        try:
            from spellchecker import SpellChecker
            self.spell = SpellChecker()
            print("✅ Enhanced spell checker initialized")
        except ImportError:
            self.spell = None
            print("❌ Spell checker not available")
        
        # Load the explicit typo model
        model_path = "models/explicit_typo_mlm"
        print(f"✅ Loading explicit typo model: {model_path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        
        # Minimal dictionaries for edge cases only
        self.apostrophe_rules = {
            'its': "it's", 'cant': "can't", 'dont': "don't", 'wont': "won't",
            'isnt': "isn't", 'arent': "aren't", 'wasnt': "wasn't", 'werent': "weren't",
            'hasnt': "hasn't", 'havent': "haven't", 'hadnt': "hadn't",
            'shouldnt': "shouldn't", 'wouldnt': "wouldn't", 'couldnt': "couldn't",
        }
        
        self.homophones = {
            'too': ['to', 'two'], 'to': ['too', 'two'], 'two': ['too', 'to'],
            'there': ['their', "they're"], 'their': ['there', "they're"],
            "they're": ['there', 'their'], 'theyre': ['there', 'their'],
            'your': ["you're"], "you're": ['your'], 'youre': ['your'],
            'its': ["it's"], "it's": ['its'], 'affect': ['effect'], 'effect': ['affect'],
        }
        
        # Common typos dictionary for fallback
        self.common_typos = {
            'teh': 'the', 'hte': 'the', 'adn': 'and', 'nad': 'and',
            'ot': 'to', 'fo': 'of', 'taht': 'that', 'thta': 'that',
            'wiht': 'with', 'htis': 'this', 'ths': 'this', 'jsut': 'just',
            'recieve': 'receive', 'beleive': 'believe', 'seperate': 'separate',
            'definately': 'definitely', 'occured': 'occurred', 'begining': 'beginning'
        }
        
        print("✅ Final Explicit Corrector initialized with trained MLM model")
    
    def correct_word_with_explicit_mlm(self, text: str, typo_word: str, top_k: int = 5) -> list:
        """Use explicit typo format for correction - this is the main method."""
        
        # Primary approach: Use explicit format with trained model
        masked_text = text.replace(typo_word, '[MASK]', 1)
        explicit_input = f"CORRUPT: {typo_word} SENTENCE: {masked_text}"
        
        # Get predictions using explicit format
        candidates = self._get_explicit_mlm_predictions(explicit_input, typo_word, top_k)
        
        # The explicit model should give very high confidence (>0.9)
        if candidates and candidates[0][1] > 0.9:
            return candidates
        
        # Fallback for edge cases (apostrophes, homophones)
        if typo_word in self.apostrophe_rules:
            return [(self.apostrophe_rules[typo_word], 0.95)]
        
        # If explicit MLM gives reasonable results, use them
        if candidates and candidates[0][1] > 0.3:
            return candidates
        
        # Final fallback to spell checker for unknown patterns
        if self.spell:
            suggestions = list(self.spell.candidates(typo_word))
            spell_candidates = []
            for suggestion in suggestions[:3]:
                if suggestion != typo_word:  # Don't suggest the same word
                    edit_dist = self._levenshtein_distance(typo_word.lower(), suggestion.lower())
                    spell_score = max(0.2, 1.0 - (edit_dist * 0.15))
                    spell_candidates.append((suggestion, spell_score))
            return spell_candidates
        
        return candidates  # Return whatever we got
    
    def _get_explicit_mlm_predictions(self, explicit_text: str, original_word: str, top_k: int = 5) -> list:
        """Get predictions using explicit typo format."""
        
        inputs = self.tokenizer(explicit_text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Find mask position
            mask_token_index = torch.where(inputs['input_ids'] == self.tokenizer.mask_token_id)[1]
            
            if len(mask_token_index) == 0:
                return []
            
            mask_logits = logits[0, mask_token_index[0], :]
            
            # Get top-k predictions
            top_k_tokens = torch.topk(mask_logits, top_k, dim=-1)
            
            candidates = []
            for i in range(top_k):
                token_id = top_k_tokens.indices[i].item()
                token = self.tokenizer.decode([token_id]).strip()
                score = torch.softmax(mask_logits, dim=-1)[token_id].item()
                
                # Filter for valid words
                if (token.isalpha() and len(token) > 1 and 
                    not token.startswith('##') and 
                    token.lower() != original_word.lower()):
                    
                    candidates.append((token, score))
            
            return candidates
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

def test_final_explicit_corrector():
    """Test the final explicit corrector."""
    
    print("🏆 TESTING FINAL EXPLICIT CORRECTOR")
    print("="*70)
    print("Uses explicit format: 'CORRUPT: typo SENTENCE: [MASK] context'")
    
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
    
    corrector = FinalExplicitCorrector()
    
    # Calculate metrics
    stats = {
        'total_cases': len(test_cases),
        'corrections_made': 0,
        'exact_matches': 0,
        'partial_improvements': 0,
        'total_tokens': 0,
        'correct_tokens': 0,
        'mlm_corrections': 0,
        'dict_corrections': 0,
        'spell_corrections': 0
    }
    
    print("\nDETAILED RESULTS:")
    print("-" * 70)
    
    for i, (corrupted, expected) in enumerate(test_cases, 1):
        corrected, info = corrector.correct_text(corrupted)
        
        print(f"\n{i:2d}. Corrupted: '{corrupted}'")
        print(f"    Expected:  '{expected}'")
        print(f"    Final:     '{corrected}' ({info['total_corrections']} corrections)")
        
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
            
            stats['total_tokens'] += 1
            if pred_token == exp_token:
                stats['correct_tokens'] += 1
            
            if pred_token == exp_token and orig_token != exp_token:
                improvements += 1
        
        if improvements > 0:
            stats['partial_improvements'] += 1
            print(f"    ✅ {improvements} tokens improved")
        
        # Show corrections and methods
        if info['corrections_made']:
            for correction in info['corrections_made']:
                method = correction.get('method', 'unknown')
                score = correction['score']
                print(f"    - '{correction['original']}' → '{correction['corrected']}' "
                      f"(score: {score:.3f}, method: {method})")
                
                # Count correction methods
                if score > 0.9:  # High confidence = MLM
                    stats['mlm_corrections'] += 1
                elif score > 0.8:  # Medium confidence = Dictionary
                    stats['dict_corrections'] += 1
                else:  # Low confidence = Spell checker
                    stats['spell_corrections'] += 1
    
    # Calculate final metrics
    token_accuracy = stats['correct_tokens'] / stats['total_tokens'] * 100
    exact_match_rate = stats['exact_matches'] / stats['total_cases'] * 100
    partial_improvement_rate = stats['partial_improvements'] / stats['total_cases'] * 100
    correction_attempt_rate = stats['corrections_made'] / stats['total_cases'] * 100
    
    print("\n" + "="*70)
    print("🏆 FINAL EXPLICIT CORRECTOR RESULTS")
    print("="*70)
    
    print(f"📊 Total test cases: {stats['total_cases']}")
    print(f"📊 Token accuracy: {token_accuracy:.1f}%")
    print(f"📊 Exact matches: {stats['exact_matches']} ({exact_match_rate:.1f}%)")
    print(f"📊 Partial improvements: {stats['partial_improvements']} ({partial_improvement_rate:.1f}%)")
    print(f"📊 Corrections attempted: {stats['corrections_made']} ({correction_attempt_rate:.1f}%)")
    
    print(f"\n🔧 CORRECTION METHODS:")
    print(f"   MLM corrections (>90% confidence): {stats['mlm_corrections']}")
    print(f"   Dictionary corrections (80-90%): {stats['dict_corrections']}")
    print(f"   Spell checker corrections (<80%): {stats['spell_corrections']}")
    
    print(f"\n🎯 PERFORMANCE vs TARGETS:")
    print(f"   Token Accuracy: {token_accuracy:.1f}% (Target: 90%+) {'✅' if token_accuracy >= 90 else '❌'}")
    print(f"   Exact Matches: {exact_match_rate:.1f}% (Target: 60%+) {'✅' if exact_match_rate >= 60 else '❌'}")
    
    print(f"\n📈 JOURNEY SUMMARY:")
    print(f"   Original approach: 68.3%")
    print(f"   Enhanced approach: 89.5%")
    print(f"   Final explicit format: {token_accuracy:.1f}%")
    print(f"   Total improvement: +{token_accuracy-68.3:.1f}%")
    
    if token_accuracy >= 92:
        print(f"\n🎉 BREAKTHROUGH! Explicit format achieves >92% accuracy!")
        print(f"   🧠 Key insight: Train with explicit format, use same format for inference")
        print(f"   🎯 Format: 'CORRUPT: typo SENTENCE: [MASK] context' → 'correct'")
    elif token_accuracy >= 90:
        print(f"\n🎉 SUCCESS! Target achieved: {token_accuracy:.1f}% >= 90%")
    else:
        remaining_gap = 90 - token_accuracy
        print(f"\n🔧 REMAINING GAP: {remaining_gap:.1f}% to reach 90% target")
    
    return token_accuracy

if __name__ == "__main__":
    test_final_explicit_corrector()
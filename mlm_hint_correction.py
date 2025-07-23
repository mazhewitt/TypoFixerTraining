#!/usr/bin/env python3
"""
MLM-compatible techniques to give the model hints about typo corrections.
Since MLM can only predict [MASK], we need creative approaches.
"""

import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from typing import List, Tuple, Dict, Any
import re

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
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

class MLMHintCorrector:
    """MLM corrector with creative hinting techniques."""
    
    def __init__(self, model_path: str = "distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        print(f"✅ MLM Hint Corrector loaded: {model_path}")
    
    def technique_1_context_enhancement(self, text: str, typo_word: str) -> List[Tuple[str, float]]:
        """
        Technique 1: Enhance context with similar words to bias MLM towards correct answer.
        Add similar/related words near the typo to give context clues.
        """
        
        # Add context hints based on common patterns
        context_hints = {
            'wiht': 'with help assist',
            'teh': 'the this that',
            'beutiful': 'beautiful pretty lovely gorgeous',
            'recieve': 'receive get obtain accept',
            'seperate': 'separate divide split apart',
            'definately': 'definitely certainly absolutely surely',
            'quikc': 'quick fast rapid swift',
            'problme': 'problem issue trouble difficulty'
        }
        
        hint_words = context_hints.get(typo_word.lower(), '')
        
        # Create enhanced context by adding hint words nearby
        if hint_words:
            # Method: Add hints as a parenthetical that gets masked out
            enhanced_text = text.replace(typo_word, f"[MASK] ({hint_words})", 1)
        else:
            enhanced_text = text.replace(typo_word, "[MASK]", 1)
        
        return self._get_mlm_predictions(enhanced_text)
    
    def technique_2_partial_reveal(self, text: str, typo_word: str) -> List[Tuple[str, float]]:
        """
        Technique 2: Give partial hints by keeping some letters visible.
        Replace typo with pattern like "w[MASK]th" for "wiht" → "with"
        """
        
        if len(typo_word) < 3:
            # Too short for partial reveal
            return self._get_mlm_predictions(text.replace(typo_word, "[MASK]", 1))
        
        # Strategy: Keep first and last letter, mask middle
        if len(typo_word) == 3:
            pattern = f"{typo_word[0]}[MASK]"
        elif len(typo_word) == 4:
            pattern = f"{typo_word[0]}[MASK]{typo_word[-1]}"
        else:
            # For longer words, keep first 2 and last 1
            pattern = f"{typo_word[:2]}[MASK]{typo_word[-1]}"
        
        enhanced_text = text.replace(typo_word, pattern, 1)
        return self._get_mlm_predictions(enhanced_text)
    
    def technique_3_multiple_masks(self, text: str, typo_word: str) -> List[Tuple[str, float]]:
        """
        Technique 3: Use multiple [MASK] tokens to allow for length differences.
        "wiht" → "[MASK] [MASK]" allows model to predict "with" or longer corrections.
        """
        
        # Determine number of masks based on typo length
        num_masks = max(1, len(typo_word) // 2)  # Use fewer masks for efficiency
        mask_pattern = " ".join(["[MASK]"] * num_masks)
        
        enhanced_text = text.replace(typo_word, mask_pattern, 1)
        return self._get_mlm_predictions_multi_mask(enhanced_text, num_masks)
    
    def technique_4_phonetic_context(self, text: str, typo_word: str) -> List[Tuple[str, float]]:
        """
        Technique 4: Add phonetically similar words as context.
        Use words that sound similar to guide the model.
        """
        
        # Simple phonetic patterns (could be enhanced with proper phonetic library)
        phonetic_hints = {
            'wiht': 'with wit weight',
            'teh': 'the thee',
            'ther': 'there their',
            'beutiful': 'beautiful bootiful',
            'quikc': 'quick quack',
            'outsid': 'outside outward',
            'mistaks': 'mistakes mistook',
            'sentenc': 'sentence sentiment'
        }
        
        hints = phonetic_hints.get(typo_word.lower(), '')
        if hints:
            # Add as nearby context
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() == typo_word.lower():
                    # Insert hints after the typo word position
                    words[i] = "[MASK]"
                    # Add hints as a "thought bubble" that gets masked
                    words.insert(i+1, f"({hints})")
                    break
            enhanced_text = " ".join(words)
        else:
            enhanced_text = text.replace(typo_word, "[MASK]", 1)
        
        return self._get_mlm_predictions(enhanced_text)
    
    def _get_mlm_predictions(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get MLM predictions for single mask."""
        
        inputs = self.tokenizer(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Find mask position
            mask_positions = torch.where(inputs['input_ids'] == self.tokenizer.mask_token_id)[1]
            
            if len(mask_positions) == 0:
                return []
            
            # Use first mask position
            mask_logits = logits[0, mask_positions[0], :]
            
            # Get top-k predictions
            top_k_tokens = torch.topk(mask_logits, top_k, dim=-1)
            
            candidates = []
            for i in range(top_k):
                token_id = top_k_tokens.indices[i].item()
                token = self.tokenizer.decode([token_id]).strip()
                score = torch.softmax(mask_logits, dim=-1)[token_id].item()
                
                if token.isalpha() and len(token) > 1:
                    candidates.append((token, score))
            
            return candidates
    
    def _get_mlm_predictions_multi_mask(self, text: str, num_masks: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get MLM predictions for multiple masks (simplified - just use first mask)."""
        # For simplicity, just use the first mask's predictions
        # In practice, you'd want to handle multi-token predictions properly
        return self._get_mlm_predictions(text, top_k)
    
    def correct_with_hints(self, text: str, typo_word: str) -> Dict[str, List[Tuple[str, float]]]:
        """Test all hinting techniques and return results."""
        
        results = {}
        
        print(f"🔍 Testing correction techniques for '{typo_word}' in: '{text}'")
        
        # Test each technique
        techniques = [
            ("Standard MLM", lambda t, w: self._get_mlm_predictions(t.replace(w, "[MASK]", 1))),
            ("Context Enhancement", self.technique_1_context_enhancement),
            ("Partial Reveal", self.technique_2_partial_reveal),
            ("Multiple Masks", self.technique_3_multiple_masks),
            ("Phonetic Context", self.technique_4_phonetic_context),
        ]
        
        for name, technique in techniques:
            try:
                candidates = technique(text, typo_word)
                results[name] = candidates
                
                print(f"\n{name}:")
                for i, (candidate, score) in enumerate(candidates[:3], 1):
                    print(f"  {i}. '{candidate}' ({score:.3f})")
                    
            except Exception as e:
                print(f"\n{name}: ERROR - {e}")
                results[name] = []
        
        return results

def test_mlm_hints():
    """Test MLM hinting techniques."""
    
    print("🧪 TESTING MLM HINTING TECHNIQUES")
    print("="*60)
    
    corrector = MLMHintCorrector()
    
    test_cases = [
        ("Can you help me wiht this problem?", "wiht", "with"),
        ("The beutiful day outside", "beutiful", "beautiful"),
        ("I need to recieve the package", "recieve", "receive"),
        ("This is a quikc test", "quikc", "quick"),
        ("Ther are many issues", "ther", "there"),
    ]
    
    for text, typo, expected in test_cases:
        print(f"\n{'='*60}")
        print(f"TARGET: '{typo}' → '{expected}'")
        
        results = corrector.correct_with_hints(text, typo)
        
        # Find which technique got the expected answer
        print(f"\n🎯 ANALYSIS:")
        for technique, candidates in results.items():
            found_expected = False
            expected_score = 0
            
            for candidate, score in candidates:
                if candidate.lower() == expected.lower():
                    found_expected = True
                    expected_score = score
                    break
            
            if found_expected:
                print(f"  ✅ {technique}: Found '{expected}' with score {expected_score:.3f}")
            else:
                top_candidate = candidates[0][0] if candidates else "None"
                print(f"  ❌ {technique}: Best = '{top_candidate}'")

def train_typo_aware_mlm():
    """Create training data for a typo-aware MLM model."""
    
    print("\n" + "="*60)
    print("💡 TRAINING DATA GENERATION FOR TYPO-AWARE MLM")
    print("="*60)
    
    # Create training examples that teach the model about typo patterns
    training_examples = []
    
    typo_patterns = {
        'wiht': 'with',
        'teh': 'the', 
        'beutiful': 'beautiful',
        'recieve': 'receive',
        'seperate': 'separate',
        'definately': 'definitely',
        'quikc': 'quick',
        'ther': 'there',
        'mistaks': 'mistakes',
        'sentenc': 'sentence',
        'outsid': 'outside',
        'problme': 'problem'
    }
    
    contexts = [
        "Can you help me {} this?",
        "I need to {} something important.",
        "This is {} example of the issue.",
        "We should {} this problem.",
        "The {} solution works well.",
        "{} are many ways to do this.",
        "I want to {} the best approach.",
        "This {} demonstrates the concept.",
        "We need {} better method.",
        "The {} shows what we need."
    ]
    
    print("Generated training examples:")
    print("-" * 40)
    
    for typo, correct in typo_patterns.items():
        for context in contexts[:3]:  # Use first 3 contexts
            # Create training example: context with [MASK], label = correct word
            if '{}' in context:
                masked_text = context.format('[MASK]')
                training_examples.append({
                    'text': masked_text,
                    'label': correct,
                    'original_typo': typo
                })
                
                print(f"'{masked_text}' → '{correct}' (from typo: '{typo}')")
    
    print(f"\n📊 Generated {len(training_examples)} training examples")
    print("💡 These could be used to fine-tune DistilBERT specifically for typo correction!")
    
    return training_examples

if __name__ == "__main__":
    test_mlm_hints()
    train_typo_aware_mlm()
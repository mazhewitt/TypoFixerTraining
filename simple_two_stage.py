#!/usr/bin/env python3
"""
Simple but effective two-stage approach:
1. Use spell checker (pyspellchecker) to identify typos
2. Use masked language model to correct them
"""

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    print("⚠️ spellchecker not available. Install with: pip install pyspellchecker")

import torch
import re
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from typing import List, Tuple, Dict, Any

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

class SimpleTypoCorrector:
    """Simple but effective two-stage typo corrector."""
    
    def __init__(self, model_path: str = "distilbert-base-uncased"):
        """Initialize with spell checker + MLM."""
        
        # Initialize spell checker
        if SPELLCHECKER_AVAILABLE:
            self.spell = SpellChecker()
            print("✅ Spell checker initialized")
        else:
            self.spell = None
            print("❌ Spell checker not available")
        
        # Initialize MLM
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        print(f"✅ MLM model loaded: {model_path}")
    
    def detect_typos(self, text: str) -> List[str]:
        """Detect misspelled words using spell checker."""
        if not self.spell:
            return []  # Fallback: no detection
        
        # Extract words (letters only)
        words = re.findall(r"[a-zA-Z']+", text.lower())
        
        # Find misspelled words
        misspelled = self.spell.unknown(words)
        return list(misspelled)
    
    def correct_word_with_mlm(self, text: str, target_word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Use MLM to correct a specific word in context."""
        
        # Find the word in text and replace with [MASK]
        # Simple approach: replace first occurrence
        words = text.split()
        masked_words = []
        word_found = False
        
        for word in words:
            clean_word = re.sub(r'[^\w\']', '', word.lower())
            if clean_word == target_word.lower() and not word_found:
                masked_words.append('[MASK]')
                word_found = True
            else:
                masked_words.append(word)
        
        if not word_found:
            return []
        
        masked_text = ' '.join(masked_words)
        
        # Use MLM to predict
        inputs = self.tokenizer(masked_text, return_tensors='pt')
        
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
                token = self.tokenizer.decode([token_id])
                score = torch.softmax(mask_logits, dim=-1)[token_id].item()
                
                # Filter out subwords and non-alphabetic
                if token.isalpha() and len(token) > 1:
                    candidates.append((token, score))
        
        return candidates
    
    def correct_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Correct typos in text using spell checker + MLM."""
        
        # Stage 1: Detect typos
        typos = self.detect_typos(text)
        
        if not typos:
            return text, {
                'original': text,
                'corrected': text,
                'typos_detected': 0,
                'corrections_made': [],
                'total_corrections': 0
            }
        
        print(f"🔍 Detected typos: {typos}")
        
        # Stage 2: Correct each typo
        corrected_text = text
        corrections_made = []
        
        for typo in typos:
            candidates = self.correct_word_with_mlm(corrected_text, typo)
            
            if not candidates:
                continue
            
            # Find best candidate considering edit distance
            best_score = -1
            best_candidate = typo
            
            for candidate, mlm_score in candidates:
                # Skip if too different
                edit_dist = levenshtein_distance(typo, candidate.lower())
                if edit_dist > 3:  # Too different
                    continue
                
                # Combined score (favor MLM score but penalize large edits)
                combined_score = mlm_score - (0.05 * edit_dist)  # Lower edit penalty
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            # Apply correction
            if best_candidate != typo and best_score > 0.05:  # Lower confidence threshold
                # Replace in text (case-preserving)
                old_pattern = re.compile(re.escape(typo), re.IGNORECASE)
                corrected_text = old_pattern.sub(best_candidate, corrected_text, count=1)
                
                corrections_made.append({
                    'original': typo,
                    'corrected': best_candidate,
                    'score': best_score,
                    'candidates': candidates[:3]
                })
        
        # Prepare stats
        stats = {
            'original': text,
            'corrected': corrected_text,
            'typos_detected': len(typos),
            'corrections_made': corrections_made,
            'total_corrections': len(corrections_made)
        }
        
        return corrected_text, stats

def test_simple_corrector():
    """Test the simple corrector."""
    
    print("🚀 Testing Simple Two-Stage Corrector")
    print("="*50)
    
    corrector = SimpleTypoCorrector()
    
    test_sentences = [
        "The quikc brown fox jumps over teh lazy dog",
        "I went too the stor to buy som milk", 
        "Ther are many mistaks in this sentance",
        "Its a beutiful day outsid today",
        "This is a perfeclty clean sentence"
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Testing: '{sentence}'")
        
        corrected, stats = corrector.correct_text(sentence)
        
        print(f"   Result: '{corrected}'")
        print(f"   Stats: {stats['typos_detected']} detected, {stats['total_corrections']} corrected")
        
        if stats['corrections_made']:
            for correction in stats['corrections_made']:
                print(f"   - '{correction['original']}' → '{correction['corrected']}' (score: {correction['score']:.3f})")

if __name__ == "__main__":
    test_simple_corrector()
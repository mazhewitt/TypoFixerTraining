#!/usr/bin/env python3
"""
Fixed corrector that addresses the two critical bugs:
1. Case-sensitive [MASK] replacement 
2. MLM prediction selection logic
"""
import torch
import re
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from enhanced_two_stage import EnhancedTypoCorrector

class FixedExplicitCorrector(EnhancedTypoCorrector):
    """Fixed corrector with bug fixes for accurate typo correction."""
    
    def __init__(self):
        """Initialize with explicit typo model."""
        
        # Initialize spell checker
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
        
        print("✅ Fixed Explicit Corrector initialized")

    def case_insensitive_replace(self, text: str, typo_word: str, replacement: str) -> str:
        """Replace typo_word with replacement, preserving original case."""
        
        # Find the actual occurrence in the text (case-insensitive)
        pattern = re.compile(re.escape(typo_word), re.IGNORECASE)
        match = pattern.search(text)
        
        if not match:
            return text
        
        # Get the actual word from the text (with original case)
        actual_word = match.group()
        
        # Preserve case: if original was capitalized, capitalize replacement
        if actual_word[0].isupper() and len(replacement) > 0:
            replacement = replacement[0].upper() + replacement[1:]
        elif actual_word.isupper():
            replacement = replacement.upper()
        elif actual_word.islower():
            replacement = replacement.lower()
        
        # Replace with proper case
        return pattern.sub(replacement, text, count=1)

    def create_explicit_format(self, text: str, typo_word: str) -> str:
        """Create explicit format with proper case-insensitive [MASK] replacement."""
        
        # Use case-insensitive replacement to insert [MASK]
        pattern = re.compile(re.escape(typo_word), re.IGNORECASE)
        masked_text = pattern.sub('[MASK]', text, count=1)
        
        # Create explicit format
        explicit_input = f"CORRUPT: {typo_word.lower()} SENTENCE: {masked_text}"
        
        return explicit_input

    def get_explicit_mlm_prediction(self, text: str, typo_word: str, top_k: int = 5) -> list:
        """Get MLM prediction using explicit format with fixed bugs."""
        
        # Create explicit format with proper masking
        explicit_input = self.create_explicit_format(text, typo_word)
        
        print(f"   MLM input: {explicit_input}")
        
        # Tokenize
        inputs = self.tokenizer(explicit_input, return_tensors="pt", truncation=True, max_length=128)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Find [MASK] token position
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
        
        if len(mask_positions) == 0:
            print(f"   ❌ No [MASK] found in: {explicit_input}")
            return []
        
        mask_pos = mask_positions[0]
        mask_logits = logits[0, mask_pos]
        
        # Get top predictions with confidence scores
        top_predictions = torch.topk(mask_logits, top_k)
        probabilities = torch.softmax(mask_logits, dim=0)
        
        candidates = []
        for score, token_id in zip(top_predictions.values, top_predictions.indices):
            word = self.tokenizer.decode(token_id).strip()
            confidence = probabilities[token_id].item()
            
            # Filter out subwords and special tokens
            if word and not word.startswith('##') and word.isalpha():
                candidates.append((word, confidence))
        
        print(f"   MLM predictions: {candidates[:3]}")
        return candidates

    def correct_word_with_explicit_mlm(self, text: str, typo_word: str, top_k: int = 5) -> list:
        """Correct word using explicit MLM format with fixed logic."""
        
        print(f"   🔧 Correcting '{typo_word}' in: {text}")
        
        # Use explicit format MLM (this should be very confident)
        candidates = self.get_explicit_mlm_prediction(text, typo_word, top_k)
        
        # If MLM gives confident results (>50%), use them
        if candidates and candidates[0][1] > 0.5:
            print(f"   ✅ MLM confident: {candidates[0]}")
            return candidates
        
        # Fallback to dictionaries if MLM confidence is low
        if typo_word.lower() in self.common_typos:
            correction = self.common_typos[typo_word.lower()]
            print(f"   📖 Dictionary: {correction}")
            return [(correction, 0.95)]
        
        if typo_word.lower() in self.apostrophe_rules:
            correction = self.apostrophe_rules[typo_word.lower()]
            print(f"   📖 Apostrophe: {correction}")
            return [(correction, 0.90)]
        
        # Final fallback to spell checker
        if self.spell:
            suggestions = list(self.spell.candidates(typo_word))
            if suggestions:
                print(f"   🔤 Spell check: {suggestions[0]}")
                return [(suggestions[0], 0.8)]
        
        print(f"   ❌ No correction found")
        return []

    def correct_text(self, text: str):
        """Enhanced correction with fixed bugs."""
        
        print(f"\n🔧 CORRECTING: {text}")
        
        # Stage 1: Enhanced detection
        typos = self.enhanced_detect_typos(text)
        
        if not typos:
            print("✅ No typos detected")
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
            print(f"\n--- Processing typo: {typo} ---")
            
            # Get correction candidates
            candidates = self.correct_word_with_explicit_mlm(corrected_text, typo, top_k=3)
            
            if candidates and candidates[0][1] > 0.5:  # High confidence threshold
                correction = candidates[0][0]
                confidence = candidates[0][1]
                
                # Apply correction with proper case handling
                old_text = corrected_text
                corrected_text = self.case_insensitive_replace(corrected_text, typo, correction)
                
                if corrected_text != old_text:
                    corrections_made.append({
                        'original': typo,
                        'corrected': correction,
                        'score': confidence,
                        'method': 'explicit_mlm'
                    })
                    print(f"   ✅ Applied: '{typo}' → '{correction}' ({confidence:.1%})")
                else:
                    print(f"   ⚠️ No change made for '{typo}'")
            else:
                print(f"   ❌ Low confidence, skipping '{typo}'")
        
        print(f"\n🎯 FINAL RESULT: {corrected_text}")
        
        return corrected_text, {
            'original': text,
            'corrected': corrected_text,
            'typos_detected': len(typos),
            'corrections_made': corrections_made,
            'total_corrections': len(corrections_made)
        }

if __name__ == "__main__":
    # Quick test
    corrector = FixedExplicitCorrector()
    
    test_cases = [
        "Thi is a test",
        "The quikc brown fox",
        "Can you halp me solv this problm?",
        "She recieved a beutiful gift"
    ]
    
    for test in test_cases:
        result, metadata = corrector.correct_text(test)
        print(f"Original: {test}")
        print(f"Corrected: {result}")
        print(f"Corrections: {len(metadata['corrections_made'])}")
        print("-" * 50)
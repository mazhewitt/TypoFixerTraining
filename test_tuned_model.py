#!/usr/bin/env python3
"""
Test the fine-tuned model with adjusted TypoCorrector parameters.
"""

import logging
from src.typo_correction import TypoCorrector
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_with_different_thresholds():
    """Test the fine-tuned model with various threshold configurations."""
    
    model_dir = "models/optimized_typo_fixer"
    
    # Load model
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForMaskedLM.from_pretrained(model_dir)
    
    test_sentences = [
        "Thi sis a test sentenc with typos",
        "The quikc brown fox jumps over teh lazy dog", 
        "I went too the stor to buy som milk",
        "Ther are many mistaks in this sentance",
        "Its a beutiful day outsid today"
    ]
    
    # Test different threshold configurations
    configs = [
        {"name": "Current (Conservative)", "threshold": -4.0, "penalty": 1.0, "passes": 2},
        {"name": "More Aggressive", "threshold": -3.0, "penalty": 0.8, "passes": 3},
        {"name": "Very Aggressive", "threshold": -2.5, "penalty": 0.5, "passes": 3},
        {"name": "Moderate", "threshold": -3.5, "penalty": 1.2, "passes": 2},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Threshold: {config['threshold']}, Penalty: {config['penalty']}, Passes: {config['passes']}")
        print('='*60)
        
        corrector = TypoCorrector(
            model=model,
            tokenizer=tokenizer,
            low_prob_threshold=config['threshold'],
            edit_penalty_lambda=config['penalty'],
            top_k=8,
            max_passes=config['passes'],
            device="cuda"
        )
        
        total_corrections = 0
        for sentence in test_sentences:
            corrected, stats = corrector.correct_typos(sentence)
            
            print(f"Original:  '{sentence}'")
            print(f"Corrected: '{corrected}'")
            print(f"Changes:   {stats['total_corrections']} in {stats['passes_used']} passes")
            
            if stats['corrections_made']:
                for correction in stats['corrections_made']:
                    print(f"  - '{correction['original']}' → '{correction['corrected']}' (score: {correction['score']:.2f})")
            
            total_corrections += stats['total_corrections']
            print()
        
        print(f"Total corrections made: {total_corrections}")

if __name__ == "__main__":
    test_with_different_thresholds()
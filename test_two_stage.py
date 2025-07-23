#!/usr/bin/env python3
"""
Test the two-stage typo correction approach.
"""

import logging
from src.two_stage_correction import TwoStageTypoCorrector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_two_stage_correction():
    """Test the two-stage approach with our problematic model."""
    
    print("🚀 Testing Two-Stage Typo Correction")
    print("="*60)
    
    # For now, we'll use the base DistilBERT for correction
    # since our fine-tuned model learned the wrong patterns
    corrector = TwoStageTypoCorrector(
        correction_model_path="distilbert-base-uncased",  # Use base model
        detection_model_path=None,  # Use heuristic detection
        detection_threshold=-3.5,  # More sensitive detection
        edit_penalty_lambda=1.0,
        max_corrections=3
    )
    
    test_sentences = [
        "Thi sis a test sentenc with typos",
        "The quikc brown fox jumps over teh lazy dog", 
        "I went too the stor to buy som milk",
        "Ther are many mistaks in this sentance",
        "Its a beutiful day outsid today"
    ]
    
    print("\nTwo-Stage Correction Results:")
    print("-"*60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Testing: '{sentence}'")
        
        try:
            corrected, stats = corrector.correct_text(sentence)
            
            print(f"   Result:    '{corrected}'")
            print(f"   Detected:  {stats['typos_detected']} typos")
            print(f"   Corrected: {stats['total_corrections']} typos")
            
            if stats['corrections_made']:
                print("   Details:")
                for correction in stats['corrections_made']:
                    print(f"     Pos {correction['position']}: '{correction['original']}' → '{correction['corrected']}' "
                          f"(conf: {correction['confidence']:.2f}, score: {correction['score']:.2f})")
            else:
                print("   Details: No corrections made")
                
        except Exception as e:
            print(f"   ERROR: {e}")

def test_with_fine_tuned_model():
    """Test with the fine-tuned model to see if two-stage helps."""
    
    print("\n" + "="*60)
    print("Testing with Fine-Tuned Model (for comparison)")
    print("="*60)
    
    try:
        # Test with our fine-tuned model that learned wrong patterns
        corrector = TwoStageTypoCorrector(
            correction_model_path="models/optimized_typo_fixer",
            detection_model_path=None,
            detection_threshold=-2.0,  # Very sensitive since model is confused
            edit_penalty_lambda=0.5,   # Low penalty to encourage corrections
            max_corrections=5
        )
        
        test_sentence = "The quikc brown fox jumps over teh lazy dog"
        print(f"\nTesting: '{test_sentence}'")
        
        corrected, stats = corrector.correct_text(test_sentence)
        
        print(f"Result:    '{corrected}'")
        print(f"Detected:  {stats['typos_detected']} typos")
        print(f"Corrected: {stats['total_corrections']} typos")
        
        if stats['corrections_made']:
            print("Details:")
            for correction in stats['corrections_made']:
                print(f"  Pos {correction['position']}: '{correction['original']}' → '{correction['corrected']}' "
                      f"(conf: {correction['confidence']:.2f}, score: {correction['score']:.2f})")
        
    except Exception as e:
        print(f"ERROR with fine-tuned model: {e}")

def compare_approaches():
    """Compare single-stage vs two-stage approaches."""
    
    print("\n" + "="*60)
    print("COMPARISON: Single-Stage vs Two-Stage")
    print("="*60)
    
    from src.typo_correction import load_pretrained_corrector
    
    # Single-stage (original TypoCorrector)
    try:
        single_stage = load_pretrained_corrector("distilbert-base-uncased")
        test_sentence = "The quikc brown fox jumps over teh lazy dog"
        
        print(f"\nTest sentence: '{test_sentence}'")
        print("-" * 40)
        
        # Single-stage result
        single_result, single_stats = single_stage.correct_typos(test_sentence)
        print(f"Single-stage: '{single_result}' ({single_stats['total_corrections']} corrections)")
        
        # Two-stage result
        two_stage = TwoStageTypoCorrector(
            correction_model_path="distilbert-base-uncased",
            detection_threshold=-3.5
        )
        two_result, two_stats = two_stage.correct_text(test_sentence)
        print(f"Two-stage:    '{two_result}' ({two_stats['total_corrections']} corrections)")
        
        print(f"\nDetection phase: {two_stats['typos_detected']} typos detected")
        
    except Exception as e:
        print(f"Comparison failed: {e}")

if __name__ == "__main__":
    test_two_stage_correction()
    test_with_fine_tuned_model()
    compare_approaches()
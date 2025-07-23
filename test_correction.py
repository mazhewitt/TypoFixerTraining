#!/usr/bin/env python3
"""
Test script for the BERT-based typo correction algorithm using non-fine-tuned DistilBERT.
"""

import logging
from src.typo_correction import load_pretrained_corrector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_typo_correction():
    """Test the typo correction algorithm with various examples."""
    
    # Load non-fine-tuned DistilBERT corrector
    logger.info("Loading non-fine-tuned DistilBERT corrector...")
    corrector = load_pretrained_corrector(
        model_name="distilbert-base-uncased",
        low_prob_threshold=-6.0,  # More lenient threshold for base model
        edit_penalty_lambda=1.5,  # Lower edit penalty
        top_k=5,
        max_passes=2
    )
    
    # Test sentences with various typo types
    test_sentences = [
        # Word joining examples (our new corruption type)
        "I thin kthis is a good exampl eof the problem",
        "The quic kbrown fox jump sover the lazy dog",
        "Can yo uhelp me wit hthis tas kplease",
        
        # Keyboard neighbor typos
        "The quikc brown fox jumps over teh lazy dog",
        "I webt to the store yesterday",
        "This sentebce has some typos in it",
        
        # Character drops
        "This is a tst sentence with droped letters",
        "I ned help with this problm",
        "The waether is really nce today",
        
        # Character doubles
        "This senttence haas dooubled charracters",
        "I neeed heelp wiith thiis",
        
        # Transpositions
        "The quick bronw fox jmups over the lazy dog",
        "This senetnce has some transposiiotns",
        
        # Space splits
        "This sent ence has sp ace issues",
        "I ne ed he lp with th is",
        
        # Clean sentences (should remain unchanged)
        "This is a perfectly clean sentence",
        "The weather is really nice today",
        "Can you help me with this task please"
    ]
    
    logger.info(f"Testing {len(test_sentences)} sentences...")
    print("\n" + "="*80)
    print("TYPO CORRECTION TEST RESULTS")
    print("="*80)
    
    total_corrections = 0
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i:2d}. Testing: '{sentence}'")
        
        try:
            corrected, stats = corrector.correct_typos(sentence)
            
            print(f"    Result:  '{corrected}'")
            print(f"    Passes:  {stats['passes_used']}")
            print(f"    Changes: {stats['total_corrections']}")
            
            if stats['corrections_made']:
                print("    Details:")
                for correction in stats['corrections_made']:
                    print(f"      Pass {correction['pass']}: "
                          f"'{correction['original']}' -> '{correction['corrected']}' "
                          f"(score: {correction['score']:.2f})")
                
                total_corrections += stats['total_corrections']
            else:
                print("    Details: No corrections made")
        
        except Exception as e:
            print(f"    ERROR: {e}")
            logger.error(f"Failed to correct sentence {i}: {e}")
    
    print("\n" + "="*80)
    print(f"SUMMARY: {total_corrections} total corrections across {len(test_sentences)} sentences")
    print("="*80)

def test_simple_interface():
    """Test the simple correction interface."""
    print("\n" + "-"*50)
    print("TESTING SIMPLE INTERFACE")
    print("-"*50)
    
    from src.typo_correction import correct_sentence
    
    test_cases = [
        "I thin kthis wil lwork wel l",
        "The quikc brownf ox jump sover",
        "This is a perfeclty clean sentence"
    ]
    
    for sentence in test_cases:
        try:
            corrected = correct_sentence(sentence)
            print(f"'{sentence}' -> '{corrected}'")
        except Exception as e:
            print(f"ERROR correcting '{sentence}': {e}")

if __name__ == "__main__":
    print("🚀 Testing BERT-based typo correction with non-fine-tuned DistilBERT")
    print(f"📋 This tests the iterative masked language modeling approach")
    
    try:
        test_typo_correction()
        test_simple_interface()
        
        print("\n✅ Testing completed!")
        print("\n💡 Notes:")
        print("   - Non-fine-tuned models may not perform as well as fine-tuned ones")
        print("   - The algorithm identifies low-probability tokens and tries to improve them")
        print("   - Edit distance penalty prevents too many changes from original text")
        print("   - Multiple passes allow for iterative improvement")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Make sure transformers and torch are installed")
        print("   - Check internet connection for model download")
        print("   - Verify Python path includes src/ directory")
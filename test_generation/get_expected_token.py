#!/usr/bin/env python3
"""Get the expected first token from typo_fixer_complete.py for our test sentence."""

import sys
import os
sys.path.append('/Users/mazdahewitt/projects/train-typo-fixer')

from typo_fixer_complete import CoreMLTypoFixer

def main():
    """Get first token for test sentence."""
    model_dir = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane-flex"
    tokenizer_path = "mazhewitt/qwen-typo-fixer"
    test_sentence = "This setence has multple typos in it"
    
    print(f"Getting expected first token for: '{test_sentence}'")
    
    try:
        # Initialize typo fixer
        fixer = CoreMLTypoFixer(model_dir, tokenizer_path)
        
        # Run typo correction and capture first token
        corrected = fixer.fix_typos(test_sentence, max_new_tokens=1, use_basic=True)
        
        print(f"Corrected: '{corrected}'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
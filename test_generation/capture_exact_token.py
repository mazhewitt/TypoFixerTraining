#!/usr/bin/env python3
"""
Capture the exact first token from typo_fixer_complete.py for our test sentence.
Uses the same prompt settings as the original implementation.
"""

import sys
import os
import json
sys.path.append('/Users/mazdahewitt/projects/train-typo-fixer')

from typo_fixer_complete import CoreMLTypoFixer

def main():
    """Capture exact first token with same settings as original."""
    model_dir = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane-flex"
    tokenizer_path = "mazhewitt/qwen-typo-fixer"
    test_sentence = "This setence has multple typos in it"
    
    print(f"Capturing exact first token for: '{test_sentence}'")
    print("Using same settings as typo_fixer_complete.py")
    print()
    
    try:
        # Initialize typo fixer
        fixer = CoreMLTypoFixer(model_dir, tokenizer_path)
        
        # Create the exact same prompt as the original (use_basic=True by default)
        basic_prompt = fixer.create_basic_prompt(test_sentence)
        print(f"Basic prompt: '{basic_prompt}'")
        
        # Tokenize with same settings (max_length=64 for basic prompts)
        tokenized = fixer.tokenize_input(basic_prompt, max_length=64)
        print(f"Tokenized shape: {tokenized.shape}")
        
        # Show tokens for verification
        decoded = fixer.tokenizer.decode(tokenized[0], skip_special_tokens=False)
        print(f"Decoded prompt: '{decoded}'")
        print()
        
        # Run with exactly same parameters as main() in typo_fixer_complete.py
        print("Running typo correction with exact same parameters...")
        corrected = fixer.fix_typos(test_sentence, max_new_tokens=15, use_basic=True)
        
        print(f"Corrected: '{corrected}'")
        
        # Save the settings for our test generation
        settings = {
            "test_sentence": test_sentence,
            "use_basic": True,
            "max_length": 64,
            "max_new_tokens": 15,
            "basic_prompt": basic_prompt,
            "tokenized_shape": list(tokenized.shape),
            "decoded_prompt": decoded
        }
        
        with open("test_generation/exact_settings.json", 'w') as f:
            json.dump(settings, f, indent=2)
        
        print(f"\n✅ Saved exact settings to exact_settings.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
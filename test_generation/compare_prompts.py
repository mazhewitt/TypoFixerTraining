#!/usr/bin/env python3
"""Compare behavior between basic and few-shot prompts."""

import sys
sys.path.append('/Users/mazdahewitt/projects/train-typo-fixer')

from typo_fixer_complete import CoreMLTypoFixer

def main():
    model_dir = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane-flex"
    tokenizer_path = "mazhewitt/qwen-typo-fixer"
    test_sentence = "This setence has multple typos in it"
    
    print("🔍 COMPARING BASIC vs FEW-SHOT PROMPTS")
    print("=" * 60)
    
    fixer = CoreMLTypoFixer(model_dir, tokenizer_path)
    
    # Test basic prompt
    print("\n📝 BASIC PROMPT:")
    basic_prompt = fixer.create_basic_prompt(test_sentence)
    print(f"Prompt: '{basic_prompt}'")
    print("First few tokens:")
    try:
        corrected_basic = fixer.fix_typos(test_sentence, max_new_tokens=5, use_basic=True)
        print(f"Result: '{corrected_basic}'")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    
    # Test few-shot prompt
    print("\n📝 FEW-SHOT PROMPT:")
    few_shot_prompt = fixer.create_few_shot_prompt(test_sentence)
    print(f"Prompt (first 100 chars): '{few_shot_prompt[:100]}...'")
    print("First few tokens:")
    try:
        corrected_few_shot = fixer.fix_typos(test_sentence, max_new_tokens=5, use_basic=False)
        print(f"Result: '{corrected_few_shot}'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
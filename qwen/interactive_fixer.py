#!/usr/bin/env python3
"""
Interactive typo fixer - enter sentences and get corrections.
"""

from typo_fixer_complete import CoreMLTypoFixer

def main():
    print("🔧 Interactive CoreML Typo Fixer")
    print("=" * 50)
    
    # Initialize the fixer
    model_dir = "/Users/mazhewitt/projects/TypoFixerTraining/models/qwen-typo-fixer-ane"
    tokenizer_path = "/Users/mazhewitt/projects/TypoFixerTraining/models/qwen-typo-fixer"
    
    fixer = CoreMLTypoFixer(model_dir, tokenizer_path)
    
    print("\nEnter sentences with typos (press Ctrl+C to exit):")
    print("-" * 50)
    
    try:
        while True:
            # Get user input
            text = input("\n> ").strip()
            
            if not text:
                continue
                
            if text.lower() in ['quit', 'exit', 'bye']:
                break
            
            # Fix typos
            corrected = fixer.fix_typos(text, max_new_tokens=25, use_basic=True)
            
            print(f"📝 Original:  {text}")
            print(f"✨ Corrected: {corrected}")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
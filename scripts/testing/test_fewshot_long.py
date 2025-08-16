#!/usr/bin/env python3
import time
from typo_fixer_complete import CoreMLTypoFixer

def main():
    model_dir = "models/qwen-typo-fixer-ane-flex"
    tokenizer_path = "mazhewitt/qwen-typo-fixer"

    fixer = CoreMLTypoFixer(model_dir, tokenizer_path)

    long_sentence = (
        "Thiss is a longg sentnce with severl typoes, includng mispellngs and grammer "
        "errros; it shoud stil be corected propery by the fixr even if it goes a bit longer than usual."
    )

    start = time.time()
    corrected = fixer.fix_typos(long_sentence, max_new_tokens=60, temperature=0.0, use_basic=False)
    dur = time.time() - start

    print("\n=== Few-shot long sentence example ===")
    print("Original:", long_sentence)
    print("Corrected:", corrected)
    print(f"Time: {dur:.2f}s")

if __name__ == "__main__":
    main()

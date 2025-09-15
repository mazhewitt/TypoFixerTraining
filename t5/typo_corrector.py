#!/usr/bin/env python3
"""
T5/ByT5 Typo Corrector - Correct typos using a trained seq2seq model.
Usage:
  python3 t5/typo_corrector.py -m models/byt5-small-typo-fixer "Text with typos"
  python3 t5/typo_corrector.py -m models/t5-small-typo-fixer-extended -i
"""

import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TypoCorrector:
    """T5/ByT5-based typo corrector."""

    def __init__(self, model_path: str = "models/t5-small-typo-fixer-extended"):
        self.model_path = model_path
        self.device = (
            "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained model and tokenizer."""
        mp = Path(self.model_path)
        if not mp.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        print(f"📥 Loading model from {self.model_path}")
        try:
            # Auto classes support both T5 and ByT5
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            # Prefer running on MPS/CUDA if available
            self.model.to(self.device)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"✅ Model loaded: {total_params:,} parameters on {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def correct(self, text: str, max_length: int = 128, num_beams: int = 4, verbose: bool = False) -> str:
        """Correct typos in the given text."""
        if not text or not text.strip():
            return text

        prompt = "fix spelling errors only, don't change the meaning of the text:"
        input_text = f"{prompt} {text.strip()}"

        if verbose:
            print(f"Input: '{text}'")
            print(f"Formatted: '{input_text}'")

        try:
            enc = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            pad_id = self.tokenizer.pad_token_id
            if pad_id is None and hasattr(self.tokenizer, "eos_token_id"):
                pad_id = self.tokenizer.eos_token_id

            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=pad_id,
                )

            corrected = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            if verbose:
                print(f"Generated: '{corrected}'")
            return corrected
        except Exception as e:
            print(f"❌ Error during correction: {e}")
            return text

    def correct_batch(self, texts, **kwargs):
        return [self.correct(t, **kwargs) for t in texts]


def main():
    parser = argparse.ArgumentParser(
        description="Correct typos using a trained T5/ByT5 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 t5/typo_corrector.py -m models/byt5-small-typo-fixer "I beleive this is teh correct answr."
  python3 t5/typo_corrector.py -m models/t5-small-typo-fixer-extended "The qucik brown fox jumps over teh lazy dog" --verbose
  echo "She recieved her degre yesterday" | python3 t5/typo_corrector.py --stdin -m models/byt5-small-typo-fixer
        """,
    )

    parser.add_argument("text", nargs="?", help="Text to correct (or use --stdin)")
    parser.add_argument("--model-path", "-m", default="models/t5-small-typo-fixer-extended", help="Path to trained model")
    parser.add_argument("--max-length", "-l", type=int, default=128, help="Maximum output length")
    parser.add_argument("--num-beams", "-b", type=int, default=4, help="Number of beams for generation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
    parser.add_argument("--stdin", action="store_true", help="Read input from stdin")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    try:
        corrector = TypoCorrector(args.model_path)
    except Exception as e:
        print(f"❌ Failed to initialize corrector: {e}")
        sys.exit(1)

    if args.interactive:
        print("🔧 Interactive Typo Corrector (type 'quit' to exit)")
        print("=" * 50)
        while True:
            try:
                text = input("\nEnter text to correct: ").strip()
                if text.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                if not text:
                    continue
                corr = corrector.correct(text, max_length=args.max_length, num_beams=args.num_beams, verbose=args.verbose)
                print(f"✅ Corrected: {corr}")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    elif args.stdin:
        try:
            text = sys.stdin.read().strip()
            if text:
                print(corrector.correct(text, max_length=args.max_length, num_beams=args.num_beams, verbose=args.verbose))
        except Exception as e:
            print(f"❌ Error reading from stdin: {e}")
            sys.exit(1)
    elif args.text:
        print(corrector.correct(args.text, max_length=args.max_length, num_beams=args.num_beams, verbose=args.verbose))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
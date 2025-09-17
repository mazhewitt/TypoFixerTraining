#!/usr/bin/env python3
"""
Conservative typo correction inference with few-shot prompting
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ConservativeTypoCorrector:
    def __init__(self, model_path="models/qwen-enhanced-typo-fixer/checkpoint-5500"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def create_conservative_prompt(self, text):
        """Create a few-shot prompt that encourages conservative corrections."""
        return f"""<|im_start|>user
Fix only spelling errors. Keep times, dates, and all other details exactly the same.

Examples:
- "I cant beleive its raining agian today." → "I can't believe it's raining again today."
- "The conferance will be held at 3pm tomorow." → "The conference will be held at 3pm tomorrow."
- "Please send me the documnet by friday morning." → "Please send me the document by friday morning."
- "We need to discus the projcet detials today" → "We need to discuss the project details today"

Correct the typos: {text}<|im_end|>
<|im_start|>assistant\n"""

    def correct_typos(self, text):
        """Correct typos in the given text conservatively."""
        prompt = self.create_conservative_prompt(text)

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=25,  # Conservative - just enough for corrections
                do_sample=True,
                temperature=0.05,   # Very low creativity
                top_p=0.9,         # Conservative nucleus sampling
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )

        # Extract correction
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = outputs[0][len(prompt_tokens):]
        correction = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        # Clean up
        if correction.startswith('\n'):
            correction = correction[1:]
        if '\nassistant\n' in correction:
            correction = correction.split('\nassistant\n')[0].strip()
        if correction.startswith('assistant\n'):
            correction = correction[10:].strip()

        return correction

def main():
    """Test the conservative typo corrector."""
    corrector = ConservativeTypoCorrector()

    test_cases = [
        "The meetign will start at 9 oclock tommorrow.",
        "I beleive this is teh correct answr.",
        "She recieved the packege yesteday afternoon.",
        "We need to discus the projcet detials today",
        "The resturant serves excellnt food evry day",
        "Please confrim your attendence by 2pm friday",
        "The confernce call is schedled for 10:30am",
        "I'll be ariving at the airprot around 6pm"
    ]

    print("🔧 Conservative Typo Correction with Few-Shot Examples")
    print("="*60)

    for test_case in test_cases:
        correction = corrector.correct_typos(test_case)
        print(f"Original:  '{test_case}'")
        print(f"Corrected: '{correction}'")
        print("-" * 60)

if __name__ == "__main__":
    main()
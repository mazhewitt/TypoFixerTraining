#!/usr/bin/env python3
"""
Test few-shot prompting for conservative typo correction
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_few_shot_prompts():
    # Load the best model
    model_path = "models/qwen-enhanced-typo-fixer/checkpoint-5500"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    test_cases = [
        "The meetign will start at 9 oclock tommorrow.",
        "I beleive this is teh correct answr.",
        "She recieved the packege yesteday afternoon."
    ]

    # Define few-shot examples showing conservative corrections
    few_shot_examples = [
        {
            "input": "I cant beleive its raining agian today.",
            "output": "I can't believe it's raining again today."
        },
        {
            "input": "The conferance will be held at 3pm tomorow.",
            "output": "The conference will be held at 3pm tomorrow."
        },
        {
            "input": "Please send me the documnet by friday morning.",
            "output": "Please send me the document by friday morning."
        }
    ]

    prompt_templates = [
        {
            "name": "Zero-shot (current)",
            "template": lambda text: f"<|im_start|>user\nCorrect the typos in this text: {text}<|im_end|>\n<|im_start|>assistant\n"
        },
        {
            "name": "Few-shot examples",
            "template": lambda text: f"""<|im_start|>user
Here are examples of correcting typos while keeping everything else the same:

Input: I cant beleive its raining agian today.
Output: I can't believe it's raining again today.

Input: The conferance will be held at 3pm tomorow.
Output: The conference will be held at 3pm tomorrow.

Input: Please send me the documnet by friday morning.
Output: Please send me the document by friday morning.

Now correct the typos in this text: {text}<|im_end|>
<|im_start|>assistant\n"""
        },
        {
            "name": "Few-shot with explicit instruction",
            "template": lambda text: f"""<|im_start|>user
Fix only spelling errors. Keep times, dates, and all other details exactly the same.

Examples:
- "I cant beleive its raining agian today." → "I can't believe it's raining again today."
- "The conferance will be held at 3pm tomorow." → "The conference will be held at 3pm tomorrow."
- "Please send me the documnet by friday morning." → "Please send me the document by friday morning."

Correct the typos: {text}<|im_end|>
<|im_start|>assistant\n"""
        },
        {
            "name": "Concise few-shot",
            "template": lambda text: f"""<|im_start|>user
Fix spelling only:

cant → can't
beleive → believe
agian → again
conferance → conference
tomorow → tomorrow
documnet → document

Correct: {text}<|im_end|>
<|im_start|>assistant\n"""
        },
        {
            "name": "Pattern-based few-shot",
            "template": lambda text: f"""<|im_start|>user
Correct spelling errors only. Examples:

Wrong: "The meetign is at 5pm tomorow"
Right: "The meeting is at 5pm tomorrow"
(Only fixed: meetign→meeting, tomorow→tomorrow. Kept time: 5pm)

Wrong: "I recieved the email yesteday morning"
Right: "I received the email yesterday morning"
(Only fixed: recieved→received, yesteday→yesterday. Kept time: morning)

Now correct: {text}<|im_end|>
<|im_start|>assistant\n"""
        }
    ]

    for template_config in prompt_templates:
        print(f"\n{'='*80}")
        print(f"🧪 Testing: {template_config['name']}")
        print(f"{'='*80}")

        for test_case in test_cases:
            prompt = template_config['template'](test_case)

            # Show the prompt (truncated for readability)
            print(f"Prompt type: {template_config['name']}")
            if len(prompt) > 200:
                print(f"Prompt preview: {prompt[:200]}...")
            else:
                print(f"Full prompt: {prompt}")

            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,  # Conservative length
                    do_sample=True,
                    temperature=0.1,    # Low creativity
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Extract correction
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = outputs[0][len(prompt_tokens):]
            correction = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

            # Clean up
            if correction.startswith('\n'):
                correction = correction[1:]
            if '\nassistant\n' in correction:
                correction = correction.split('\nassistant\n')[0].strip()
            if correction.startswith('assistant\n'):
                correction = correction[10:].strip()

            print(f"Original:  '{test_case}'")
            print(f"Corrected: '{correction}'")
            print(f"{'-'*60}")

if __name__ == "__main__":
    test_few_shot_prompts()
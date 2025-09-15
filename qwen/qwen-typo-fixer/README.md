---
language: en
tags:
- text-generation
- typo-correction
- qwen
- fine-tuned
license: apache-2.0
datasets:
- custom-typo-dataset
base_model: Qwen/Qwen3-0.6B
---

# Qwen 0.6B Typo Correction Model

Fine-tuned Qwen 0.6B model for automatic typo correction, achieving 40.0% sentence accuracy.

## Model Details

- **Base Model**: Qwen/Qwen3-0.6B
- **Fine-tuning Data**: 21,265 examples from multiple high-quality sources
- **Training Time**: 34.0 minutes on RTX 5090
- **Final Accuracy**: 40.0%
- **Target**: 90.0% sentence accuracy

## Data Sources

- ✅ Norvig's 20k misspellings from Google spellcheck logs
- ✅ Holbrook/Birkbeck academic typo correction datasets  
- ✅ Wikipedia revision history typo corrections
- ✅ Enhanced keyboard layout error simulation
- ✅ WikiText natural sentence extraction

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mazhewitt/qwen-typo-fixer-v2")
model = AutoModelForCausalLM.from_pretrained("mazhewitt/qwen-typo-fixer-v2")

# Correct typos
prompt = "Correct the typos: I beleive this is teh correct answr."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
correction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
print(correction)  # "I believe this is the correct answer."
```

## Performance

- **Sentence Accuracy**: 40.0%
- **Training Examples**: 21,265
- **Validation Examples**: 2,362
- **GPU**: RTX 5090 with BFloat16 + Flash Attention 2

## Deployment

Optimized for Apple Neural Engine deployment via anemll conversion pipeline.

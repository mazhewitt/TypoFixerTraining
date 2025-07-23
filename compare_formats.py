#!/usr/bin/env python3
"""
Compare explicit format vs standard format when calling the MLM.
Since we trained with explicit format, we should use it for best results.
"""

import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

def compare_formats():
    """Compare explicit vs standard format predictions."""
    
    print("🔍 COMPARING MLM FORMATS")
    print("="*60)
    
    # Load the explicit-trained model
    model_path = "models/explicit_typo_mlm"
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForMaskedLM.from_pretrained(model_path)
    model.eval()
    
    test_cases = [
        ("Can you help me wiht this problem?", "wiht", "with"),
        ("It's a beutiful day outside.", "beutiful", "beautiful"),
        ("I will recieve the package tomorrow.", "recieve", "receive"),
        ("That was a quikc response.", "quikc", "quick"),
        ("Ther are many options.", "ther", "there"),
    ]
    
    for text, typo, expected in test_cases:
        print(f"\nTest: '{text}' ('{typo}' → '{expected}')")
        print("-" * 50)
        
        # Format 1: Standard format (what we used to do)
        standard_input = text.replace(typo, "[MASK]", 1)
        print(f"Standard: '{standard_input}'")
        
        standard_predictions = get_predictions(model, tokenizer, standard_input, expected)
        print(f"  Predictions: {standard_predictions[:3]}")
        
        # Format 2: Explicit format (how we trained the model)
        explicit_input = f"CORRUPT: {typo} SENTENCE: {text.replace(typo, '[MASK]', 1)}"
        print(f"Explicit: '{explicit_input}'")
        
        explicit_predictions = get_predictions(model, tokenizer, explicit_input, expected)
        print(f"  Predictions: {explicit_predictions[:3]}")
        
        # Compare results
        standard_found = any(pred[0].lower() == expected.lower() for pred in standard_predictions[:3])
        explicit_found = any(pred[0].lower() == expected.lower() for pred in explicit_predictions[:3])
        
        print(f"  Result: Standard {'✅' if standard_found else '❌'} | Explicit {'✅' if explicit_found else '❌'}")

def get_predictions(model, tokenizer, text, expected, top_k=5):
    """Get MLM predictions for given text."""
    
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Find mask position
        mask_positions = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
        
        if len(mask_positions) == 0:
            return []
        
        mask_logits = logits[0, mask_positions[0], :]
        top_k_tokens = torch.topk(mask_logits, top_k)
        
        predictions = []
        for i in range(top_k):
            token_id = top_k_tokens.indices[i].item()
            token = tokenizer.decode([token_id]).strip()
            score = torch.softmax(mask_logits, dim=-1)[token_id].item()
            
            if token.isalpha() and len(token) > 1:
                marker = " ✅" if token.lower() == expected.lower() else ""
                predictions.append((f"{token}{marker}", f"{score:.3f}"))
        
        return predictions

if __name__ == "__main__":
    compare_formats()
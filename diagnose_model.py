#!/usr/bin/env python3
"""
Diagnostic tool to understand why the fine-tuned model isn't making corrections.
"""

import torch
import logging
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_model_behavior():
    """Diagnose why the model isn't making corrections."""
    
    model_dir = "models/optimized_typo_fixer"
    
    # Load model
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForMaskedLM.from_pretrained(model_dir)
    model.eval()
    
    # Test sentence with obvious typos
    test_sentence = "The quikc brown fox jumps over teh lazy dog"
    
    print(f"Diagnosing: '{test_sentence}'")
    print("="*60)
    
    # Tokenize
    inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        # Get token probabilities
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get log prob of actual tokens
        token_log_probs = torch.gather(log_probs, dim=-1, index=input_ids[0].unsqueeze(-1)).squeeze(-1)
        
        print("Token Analysis:")
        print("-" * 40)
        for i in range(1, input_ids.shape[1] - 1):  # Skip CLS and SEP
            if attention_mask[0, i] == 0:
                break
                
            token_id = input_ids[0, i].item()
            token = tokenizer.decode([token_id], skip_special_tokens=True) 
            log_prob = token_log_probs[i].item()
            
            # Get top-5 alternatives
            top_logits, top_indices = torch.topk(logits[i], 5)
            top_tokens = [tokenizer.decode([idx.item()], skip_special_tokens=True) for idx in top_indices]
            top_probs = torch.log_softmax(logits[i], dim=-1)[top_indices].tolist()
            
            print(f"Position {i:2d}: '{token}' (log_prob: {log_prob:.3f})")
            print(f"  Top alternatives:")
            for j, (alt_token, alt_prob) in enumerate(zip(top_tokens, top_probs)):
                marker = " ←" if j == 0 and alt_token != token else ""
                print(f"    {j+1}. '{alt_token}' ({alt_prob:.3f}){marker}")
            
            # Check if this looks like a typo
            if log_prob < -3.0:
                print(f"  ** SUSPICIOUS: Low probability token **")
            print()

def test_specific_corrections():
    """Test specific typo corrections manually."""
    
    model_dir = "models/optimized_typo_fixer"
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForMaskedLM.from_pretrained(model_dir)
    model.eval()
    
    # Test specific typo corrections
    test_cases = [
        ("The [MASK] brown fox", ["quick", "quikc"]),  # Should prefer "quick"
        ("jumps over [MASK] lazy dog", ["the", "teh"]),  # Should prefer "the"
        ("This [MASK] a test", ["is", "sis"]),  # Should prefer "is"
    ]
    
    print("\nSpecific Correction Tests:")
    print("="*60)
    
    for masked_text, candidates in test_cases:
        print(f"\nTesting: '{masked_text}'")
        
        # Tokenize
        inputs = tokenizer(masked_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            
            # Find mask position
            mask_pos = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
            
            # Get probabilities for candidates
            print("Candidate probabilities:")
            for candidate in candidates:
                candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
                if len(candidate_ids) == 1:
                    candidate_id = candidate_ids[0]
                    log_prob = torch.log_softmax(logits[mask_pos], dim=-1)[candidate_id].item()
                    print(f"  '{candidate}': {log_prob:.3f}")

if __name__ == "__main__":
    diagnose_model_behavior()
    test_specific_corrections()
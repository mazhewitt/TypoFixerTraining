#!/usr/bin/env python3
"""
Test the training data format to verify it's set up correctly.
"""

import json
from transformers import DistilBertTokenizer

def test_training_format():
    """Test how training data is being processed."""
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load a few examples
    examples = []
    with open("data/processed/realistic_train.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            examples.append(json.loads(line.strip()))
    
    print("Training Data Analysis:")
    print("=" * 60)
    
    for i, example in enumerate(examples):
        corrupted = example['corrupted']
        clean = example['clean']
        
        print(f"\nExample {i+1}:")
        print(f"Corrupted: '{corrupted}'")
        print(f"Clean:     '{clean}'")
        
        # Tokenize both
        corrupted_enc = tokenizer(corrupted, truncation=True, padding='max_length', 
                                 max_length=64, return_tensors='pt')
        clean_enc = tokenizer(clean, truncation=True, padding='max_length',
                            max_length=64, return_tensors='pt')
        
        input_ids = corrupted_enc['input_ids'].squeeze()
        labels = clean_enc['input_ids'].squeeze()
        mask = input_ids != labels
        
        print("\nTokenization:")
        print("Position | Input Token | Label Token | Train?")
        print("-" * 50)
        
        for pos in range(min(10, len(input_ids))):
            input_token = tokenizer.decode([input_ids[pos].item()], skip_special_tokens=True)
            label_token = tokenizer.decode([labels[pos].item()], skip_special_tokens=True)
            train_on = "YES" if mask[pos] else "NO"
            
            print(f"{pos:8d} | {input_token:11s} | {label_token:11s} | {train_on}")
        
        print()

def test_simple_case():
    """Test with a simple, obvious case."""
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create a simple test case
    corrupted = "The quikc brown fox"
    clean = "The quick brown fox"
    
    print("\nSimple Test Case:")
    print("=" * 40)
    print(f"Corrupted: '{corrupted}'")
    print(f"Clean:     '{clean}'")
    
    # Tokenize
    corrupted_enc = tokenizer(corrupted, return_tensors='pt')
    clean_enc = tokenizer(clean, return_tensors='pt')
    
    input_ids = corrupted_enc['input_ids'].squeeze()
    labels = clean_enc['input_ids'].squeeze()
    mask = input_ids != labels
    
    print("\nWhat the model should learn:")
    print("Input -> Expected Output")
    print("-" * 30)
    
    for pos in range(len(input_ids)):
        if mask[pos]:
            input_token = tokenizer.decode([input_ids[pos].item()], skip_special_tokens=True)
            label_token = tokenizer.decode([labels[pos].item()], skip_special_tokens=True)
            print(f"'{input_token}' -> '{label_token}'")

if __name__ == "__main__":
    test_training_format()
    test_simple_case()
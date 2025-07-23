#!/usr/bin/env python3
"""
Train a binary classification model to detect typos at the token level.
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertForTokenClassification, 
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from difflib import SequenceMatcher
import argparse

class TypoDetectionDataset(Dataset):
    """Dataset for token-level typo detection."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Loading detection data from {data_file}")
        
        with open(data_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.examples.append(data)
        
        print(f"Loaded {len(self.examples)} examples for detection training")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        corrupted = example['corrupted']
        clean = example['clean']
        
        # Tokenize corrupted text
        inputs = self.tokenizer(
            corrupted,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        # Create labels: 1 for typo tokens, 0 for correct tokens
        labels = torch.zeros_like(input_ids)
        
        # Use simple word-level alignment to identify typo positions
        corrupted_words = corrupted.lower().split()
        clean_words = clean.lower().split()
        
        # Find different words using sequence matching
        matcher = SequenceMatcher(None, corrupted_words, clean_words)
        
        # Mark different words as typos (simplified approach)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():  
            if tag in ['replace', 'delete']:
                # Mark tokens in corrupted words as typos
                for word_idx in range(i1, i2):
                    if word_idx < len(corrupted_words):
                        # Find token positions for this word (approximate)
                        word = corrupted_words[word_idx]
                        word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
                        
                        # Simple heuristic: mark first few tokens corresponding to this word
                        # This is approximate but good enough for training
                        start_pos = min(word_idx + 1, len(input_ids) - 1)  # +1 for CLS
                        end_pos = min(start_pos + len(word_tokens), len(input_ids) - 1)
                        
                        for pos in range(start_pos, end_pos):
                            if pos < len(labels) and attention_mask[pos] == 1:
                                labels[pos] = 1  # Mark as typo
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def main():
    parser = argparse.ArgumentParser(description="Train typo detection model")
    parser.add_argument('--data_file', type=str, default='data/processed/realistic_train.jsonl')
    parser.add_argument('--output_dir', type=str, default='models/typo_detector')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    print("🔍 Training Typo Detection Model...")
    print(f"📁 Data: {args.data_file}")
    print(f"📁 Output: {args.output_dir}")
    
    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2  # 0=correct, 1=typo
    )
    
    # Create dataset
    dataset = TypoDetectionDataset(args.data_file, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("🚀 Starting detection model training...")
    trainer.train()
    
    # Save
    print(f"💾 Saving detection model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("✅ Detection model training completed!")
    print(f"🧪 Test with: python3 test_two_stage.py")

if __name__ == "__main__":
    main()
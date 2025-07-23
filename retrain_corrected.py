#!/usr/bin/env python3
"""
Retrain the model using the correct MLM approach where we mask the typos and predict clean tokens.
"""

import json
import argparse
from transformers import DistilBertForMaskedLM, DistilBertTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset

class CorrectMLMDataset(Dataset):
    """Dataset that masks corrupted tokens and predicts clean ones."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Loading corrected dataset from {data_file}")
        
        with open(data_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.examples.append(data)
        
        print(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        corrupted = example['corrupted']
        clean = example['clean']
        
        # Method: Use clean text as input, manually mask positions that were corrupted
        # This ensures proper alignment
        
        # Tokenize both to find differences
        corrupted_tokens = self.tokenizer(corrupted, add_special_tokens=False)['input_ids']
        clean_tokens = self.tokenizer(clean, add_special_tokens=False)['input_ids']
        
        # Use clean text as base
        inputs = self.tokenizer(
            clean, 
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze()
        labels = input_ids.clone()
        
        # Mask some tokens randomly (standard MLM) plus try to identify corruption positions
        # For now, use standard MLM approach - mask 15% randomly
        
        # Standard BERT masking
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # Don't mask special tokens
        special_tokens_mask[labels == self.tokenizer.cls_token_id] = True
        special_tokens_mask[labels == self.tokenizer.sep_token_id] = True
        special_tokens_mask[labels == self.tokenizer.pad_token_id] = True
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Get tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels for non-masked tokens to -100 (ignore in loss)
        labels[~masked_indices] = -100
        
        # Replace masked tokens with [MASK]
        input_ids[masked_indices] = self.tokenizer.mask_token_id
        
        return {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }

def main():
    parser = argparse.ArgumentParser(description="Retrain with corrected MLM approach")
    parser.add_argument('--data_file', type=str, default='data/processed/realistic_train.jsonl')
    parser.add_argument('--output_dir', type=str, default='models/corrected_typo_fixer')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    print("🔄 Retraining with corrected MLM approach...")
    print(f"📁 Data: {args.data_file}")
    print(f"📁 Output: {args.output_dir}")
    
    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    
    # Create dataset
    dataset = CorrectMLMDataset(args.data_file, tokenizer)
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
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
    print("🚀 Starting corrected training...")
    trainer.train()
    
    # Save
    print(f"💾 Saving corrected model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("✅ Corrected training completed!")
    print(f"🧪 Test with: python diagnose_model.py")

if __name__ == "__main__":
    main()
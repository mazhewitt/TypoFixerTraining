#!/usr/bin/env python3
"""
Simple ByT5 training script without complex dependencies.
Uses basic PyTorch training loop instead of HuggingFace Trainer.
"""

import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import argparse

class TypoDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256, prefix="fix typos:"):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = prefix
        
        print(f"📚 Loading dataset from {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except:
                    continue
        
        print(f"✅ Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare input and target
        source = f"{self.prefix} {item['corrupted']}"
        target = item['clean']
        
        # Tokenize
        source_encoding = self.tokenizer(
            source, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def train_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', default='data/enhanced_training_balanced.jsonl')
    parser.add_argument('--output-dir', default='models/byt5-simple-typo-fixer')
    parser.add_argument('--model-name', default='google/byt5-small')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--prefix', default='fix typos:')
    parser.add_argument('--hub-model-id', default='mazhewitt/byt5-simple-typo-fixer')
    
    args = parser.parse_args()
    
    print("🚀 SIMPLE BYT5 TRAINING")
    print(f"📁 Model: {args.model_name}")
    print(f"📊 Data: {args.train_file}")
    print(f"🎯 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🧠 Learning rate: {args.learning_rate}")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    if torch.cuda.is_available():
        print(f"🔥 GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # Load tokenizer and model
    print("🔧 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    # Move to device(s)
    if torch.cuda.device_count() > 1:
        print(f"🔥 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model.to(device)
    
    # Load dataset
    dataset = TypoDataset(args.train_file, tokenizer, args.max_length, args.prefix)
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Training loop
    model.train()
    total_steps = len(train_loader) * args.epochs
    step = 0
    
    print("🚀 Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            if torch.cuda.device_count() > 1:
                loss = loss.mean()  # For DataParallel
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log every 100 steps
            if step % 100 == 0:
                avg_loss = epoch_loss / (step % len(train_loader) + 1)
                print(f"  Step {step}/{total_steps}, Loss: {avg_loss:.4f}")
        
        # Validation
        if epoch % 1 == 0:  # Validate every epoch
            model.eval()
            val_loss = 0
            val_steps = 0
            
            print("🔍 Validating...")
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()
                    
                    val_loss += loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            print(f"📊 Epoch {epoch + 1} - Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")
            
            model.train()
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time / 60:.1f} minutes")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract model from DataParallel if needed
    model_to_save = model.module if hasattr(model, 'module') else model
    
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"💾 Model saved to: {args.output_dir}")
    
    # Test the model
    print("\n🧪 Testing model...")
    model_to_save.eval()
    
    test_cases = [
        "I beleive this is teh correct answr.",
        "The qick brown fox jumps ovr the lazy dog.",
        "Please chck your email for futher instructions."
    ]
    
    for test_input in test_cases:
        input_text = f"{args.prefix} {test_input}"
        inputs = tokenizer(input_text, return_tensors='pt', max_length=args.max_length, truncation=True)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model_to_save.generate(
                **inputs,
                max_length=args.max_length,
                num_beams=4,
                early_stopping=True
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  '{test_input}' → '{result}'")
    
    print(f"\n🎉 Training complete!")
    print(f"📁 Model: {args.output_dir}")
    
    # Try to upload to HuggingFace (optional)
    try:
        print(f"\n🚀 Uploading to HuggingFace: {args.hub_model_id}")
        model_to_save.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print("✅ Upload successful!")
    except Exception as e:
        print(f"⚠️  Upload failed: {e}")
        print("💡 You can upload manually later with:")
        print(f"   huggingface-cli upload {args.hub_model_id} {args.output_dir}")

if __name__ == "__main__":
    train_model()
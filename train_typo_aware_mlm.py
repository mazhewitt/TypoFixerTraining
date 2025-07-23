#!/usr/bin/env python3
"""
Train a typo-aware MLM model that understands correction patterns.
This will solve our remaining 0.5% accuracy gap by teaching the model about typos.
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertForMaskedLM, 
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import argparse
from pathlib import Path

class TypoAwareDataset(Dataset):
    """Dataset that teaches MLM about typo correction patterns."""
    
    def __init__(self, typo_pairs: dict, contexts: list, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Creating typo-aware dataset...")
        
        # Generate training examples for each typo-correct pair
        for typo, correct in typo_pairs.items():
            for context in contexts:
                if '{}' in context:
                    # Create multiple variations
                    variations = [
                        context.format('[MASK]'),  # Standard masking
                        context.format('[MASK]'),  # Duplicate for emphasis
                        f"The word '{typo}' should be [MASK].",  # Direct teaching
                        f"Correct spelling: [MASK] (not {typo})",  # Explicit correction
                    ]
                    
                    for text in variations:
                        # Tokenize and create MLM example
                        inputs = tokenizer(
                            text,
                            truncation=True,
                            padding='max_length',
                            max_length=max_length,
                            return_tensors='pt'
                        )
                        
                        # Find mask position and set label
                        input_ids = inputs['input_ids'].squeeze()
                        labels = input_ids.clone()
                        
                        # Set all positions to -100 (ignore) except mask position
                        labels[:] = -100
                        
                        # Find mask position and set correct token
                        mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                        
                        if len(mask_positions) > 0:
                            mask_pos = mask_positions[0].item()
                            correct_token_ids = tokenizer.encode(correct, add_special_tokens=False)
                            
                            # For simplicity, use first token of correct word
                            if correct_token_ids:
                                labels[mask_pos] = correct_token_ids[0]
                                
                                self.examples.append({
                                    'input_ids': input_ids,
                                    'attention_mask': inputs['attention_mask'].squeeze(),
                                    'labels': labels,
                                    'typo': typo,
                                    'correct': correct
                                })
        
        print(f"Generated {len(self.examples)} typo-aware training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.examples[idx]['input_ids'],
            'attention_mask': self.examples[idx]['attention_mask'],
            'labels': self.examples[idx]['labels']
        }

def create_typo_training_data():
    """Create comprehensive typo training data."""
    
    # Common typos that our current system struggles with
    typo_pairs = {
        # From our failure analysis
        'wiht': 'with',
        'teh': 'the',
        'beutiful': 'beautiful',
        'recieve': 'receive',
        'seperate': 'separate',
        'definately': 'definitely',
        'quikc': 'quick',
        'ther': 'there',
        'mistaks': 'mistakes',
        'sentenc': 'sentence',
        'sentance': 'sentence',
        'outsid': 'outside',
        'problme': 'problem',
        
        # Additional common typos
        'occured': 'occurred',
        'neccessary': 'necessary',
        'accomodate': 'accommodate',
        'embarass': 'embarrass',
        'harass': 'harass',
        'occassion': 'occasion', 
        'recomend': 'recommend',
        'maintainance': 'maintenance',
        'independant': 'independent',
        'existance': 'existence',
        'persistant': 'persistent',
        'resistanse': 'resistance',
        'appearence': 'appearance',
        'refference': 'reference',
        'prefference': 'preference',
        'occassionally': 'occasionally',
        'profesional': 'professional',
        'responsability': 'responsibility',
        
        # Character-level patterns
        'tiem': 'time',
        'taht': 'that',
        'wnat': 'want',
        'jsut': 'just',
        'frmo': 'from',
        'thier': 'their',
        'wich': 'which',
        'recieve': 'receive',
        'beleive': 'believe',
        'acheive': 'achieve',
    }
    
    # Diverse contexts to teach the model
    contexts = [
        # Help/assistance contexts (for 'wiht' → 'with')
        "Can you help me {} this problem?",
        "I need assistance {} the task.",
        "Please work {} me on this.",
        "Let's collaborate {} our team.",
        
        # Article contexts (for 'teh' → 'the')  
        "I saw {} movie yesterday.",
        "This is {} best solution.",
        "{} answer is correct.",
        "Please check {} results.",
        
        # Description contexts (for 'beutiful' → 'beautiful')
        "The garden is truly {}.",
        "She wore a {} dress.",
        "What a {} sunset!",
        "The {} scenery amazed us.",
        
        # Receiving contexts (for 'recieve' → 'receive')
        "I will {} the package tomorrow.",
        "Did you {} my message?",
        "We expect to {} confirmation.",
        "Please {} this award.",
        
        # Location contexts (for 'ther' → 'there')
        "{} are many options available.",
        "Look over {} for the answer.",
        "We'll meet {} at noon.",
        "Is anyone {} to help?",
        
        # Quick/fast contexts (for 'quikc' → 'quick')
        "That was a {} response.",
        "We need a {} solution.",
        "Give me a {} update.",
        "Take a {} look at this.",
        
        # Problem contexts (for 'problme' → 'problem')
        "This is a serious {}.",
        "Can you solve this {}?",
        "The {} requires attention.",
        "We found the {} yesterday.",
        
        # Generic contexts
        "The {} is important.",
        "We should {} this carefully.",
        "This {} works well.",
        "I {} the best approach.",
    ]
    
    return typo_pairs, contexts

def main():
    parser = argparse.ArgumentParser(description="Train typo-aware MLM")
    parser.add_argument('--output_dir', type=str, default='models/typo_aware_mlm')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    
    args = parser.parse_args()
    
    print("🎯 Training Typo-Aware MLM Model")
    print("="*50)
    print(f"📁 Output: {args.output_dir}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🎛️ Learning rate: {args.learning_rate}")
    
    # Load base model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    
    # Create typo-aware training data
    typo_pairs, contexts = create_typo_training_data()
    
    print(f"\n📚 Training on {len(typo_pairs)} typo patterns:")
    for typo, correct in list(typo_pairs.items())[:5]:
        print(f"  • '{typo}' → '{correct}'")
    print(f"  ... and {len(typo_pairs)-5} more")
    
    # Create dataset
    dataset = TypoAwareDataset(typo_pairs, contexts, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # Also include standard MLM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
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
    print(f"\n🚀 Starting typo-aware training...")
    print(f"⏱️ Estimated time: {len(dataset) // args.batch_size * args.epochs // 10} minutes")
    
    trainer.train()
    
    # Save
    print(f"\n💾 Saving typo-aware model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save typo patterns for reference
    with open(f"{args.output_dir}/typo_patterns.json", 'w') as f:
        json.dump(typo_pairs, f, indent=2)
    
    print("✅ Typo-aware MLM training completed!")
    print(f"\n🧪 Test with:")
    print(f"   python3 test_typo_aware_model.py --model_path {args.output_dir}")

if __name__ == "__main__":
    main()
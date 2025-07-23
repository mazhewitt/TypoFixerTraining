#!/usr/bin/env python3
"""
Train with explicit typo correction format:
Input:  "CORRUPT: wakl SENTENCE: [MASK] to the shops"
Target: ["walk"]

This teaches the model to understand the corruption pattern directly.
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

class ExplicitTypoDataset(Dataset):
    """Dataset with explicit typo correction format."""
    
    def __init__(self, typo_pairs: dict, contexts: list, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Creating explicit typo correction dataset...")
        
        # Generate training examples for each typo-correct pair
        for typo, correct in typo_pairs.items():
            for context in contexts:
                if '{}' in context:
                    # Create explicit format: "CORRUPT: typo SENTENCE: [MASK] context"
                    sentence = context.format('[MASK]')
                    explicit_text = f"CORRUPT: {typo} SENTENCE: {sentence}"
                    
                    # Tokenize and create MLM example
                    inputs = tokenizer(
                        explicit_text,
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
                        
                        # Use first token of correct word, ensuring it's not [MASK]
                        if correct_token_ids and correct_token_ids[0] != tokenizer.mask_token_id:
                            labels[mask_pos] = correct_token_ids[0]
                            
                            self.examples.append({
                                'input_ids': input_ids,
                                'attention_mask': inputs['attention_mask'].squeeze(),
                                'labels': labels,
                                'typo': typo,
                                'correct': correct,
                                'explicit_text': explicit_text
                            })
        
        print(f"Generated {len(self.examples)} explicit typo training examples")
        
        # Show some examples
        print(f"\nSample training examples:")
        for i in range(min(3, len(self.examples))):
            example = self.examples[i]
            print(f"  Input: '{example['explicit_text']}'")
            print(f"  Target: '{example['correct']}' (from typo: '{example['typo']}')")
            print()
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.examples[idx]['input_ids'],
            'attention_mask': self.examples[idx]['attention_mask'],
            'labels': self.examples[idx]['labels']
        }

def create_explicit_typo_data():
    """Create comprehensive typo training data with explicit format."""
    
    # Common typos that our system targets
    typo_pairs = {
        # Primary failures we want to fix
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
        'thi': 'this',
        'sis': 'is',
        'som': 'some',
        'stor': 'store',
        
        # Additional common patterns
        'wakl': 'walk',
        'talke': 'talk',
        'wriet': 'write',
        'raed': 'read',
        'helo': 'hello',
        'wrold': 'world',
        'peopel': 'people',
        'becuase': 'because',
        'woudl': 'would',
        'coudl': 'could',
        'shoudl': 'should',
        'freind': 'friend',
        'recieved': 'received',
        'beleive': 'believe',
        'acheive': 'achieve',
        'occured': 'occurred',
        'neccessary': 'necessary',
        'accomodate': 'accommodate',
        'embarass': 'embarrass',
        'occassion': 'occasion', 
        'recomend': 'recommend',
        'maintainance': 'maintenance',
        'independant': 'independent',
        'existance': 'existence',
        'persistant': 'persistent',
        'appearence': 'appearance',
        'refference': 'reference',
        'prefference': 'preference',
        'profesional': 'professional',
        'responsability': 'responsibility',
    }
    
    # Diverse contexts for training
    contexts = [
        # Action contexts
        "I want to {} to the store.",
        "Can you {} me with this?",
        "Let's {} this problem together.",
        "I need to {} something important.",
        "We should {} the solution.",
        
        # Description contexts  
        "This is {} example.",
        "The {} day was wonderful.",
        "It's a {} solution.",
        "That was {} experience.",
        "The {} answer is correct.",
        
        # Location/existence contexts
        "{} are many options.",
        "I found {} yesterday.",
        "Look {} for help.",
        "We went {} together.",
        "The {} shows the result.", 
        
        # Communication contexts
        "I {} the message.",
        "Please {} this information.",
        "We {} the confirmation.",
        "Did you {} my email?",
        "I want to {} with you.",
        
        # General contexts
        "The {} is important.",
        "We found {} solution.",
        "This {} works well.",
        "I {} the best approach.",
        "Please check {} results.",
        "The {} demonstrates this.",
        "We need {} better method.",
        "This {} shows what we need.",
        "I want {} better outcome.",
        "The {} provides clarity.",
    ]
    
    return typo_pairs, contexts

def main():
    parser = argparse.ArgumentParser(description="Train explicit typo correction MLM")
    parser.add_argument('--output_dir', type=str, default='models/explicit_typo_mlm')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    
    args = parser.parse_args()
    
    print("🎯 Training Explicit Typo Correction MLM")
    print("="*50)
    print(f"📁 Output: {args.output_dir}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🎛️ Learning rate: {args.learning_rate}")
    print(f"📝 Format: 'CORRUPT: typo SENTENCE: [MASK] context' → 'correct'")
    
    # Load base model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    
    # Create explicit typo training data
    typo_pairs, contexts = create_explicit_typo_data()
    
    print(f"\n📚 Training on {len(typo_pairs)} typo patterns:")
    for typo, correct in list(typo_pairs.items())[:5]:
        print(f"  • 'CORRUPT: {typo} SENTENCE: [MASK] ...' → '{correct}'")
    print(f"  ... and {len(typo_pairs)-5} more")
    
    # Create dataset
    dataset = ExplicitTypoDataset(typo_pairs, contexts, tokenizer)
    
    # Custom data collator - we already have labels set up
    def custom_data_collator(features):
        """Simple data collator that just batches our pre-labeled examples."""
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.stack([f[key] for f in features])
        return batch
    
    data_collator = custom_data_collator
    
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
    print(f"\n🚀 Starting explicit typo training...")
    print(f"⏱️ Estimated time: {len(dataset) // args.batch_size * args.epochs // 10} minutes")
    
    trainer.train()
    
    # Save
    print(f"\n💾 Saving explicit typo model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metadata
    metadata = {
        'format': 'CORRUPT: {typo} SENTENCE: [MASK] {context}',
        'target': '{correct_word}',
        'typo_pairs': typo_pairs,
        'num_examples': len(dataset),
        'training_args': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        }
    }
    
    with open(f"{args.output_dir}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Explicit typo MLM training completed!")
    print(f"\n🧪 Test with:")
    print(f"   python3 test_explicit_typo_model.py --model_path {args.output_dir}")
    
    # Quick test
    print(f"\n🔍 Quick test:")
    test_inputs = [
        "CORRUPT: wiht SENTENCE: Can you help me [MASK] this problem?",
        "CORRUPT: beutiful SENTENCE: It's a [MASK] day outside.",
        "CORRUPT: recieve SENTENCE: I will [MASK] the package tomorrow."
    ]
    
    model.eval()
    for test_input in test_inputs:
        inputs = tokenizer(test_input, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Find mask position
            mask_positions = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
            
            if len(mask_positions) > 0:
                mask_logits = logits[0, mask_positions[0], :]
                predicted_token_id = torch.argmax(mask_logits).item()
                predicted_token = tokenizer.decode([predicted_token_id]).strip()
                
                print(f"  Input: '{test_input}'")
                print(f"  Prediction: '{predicted_token}'")
                print()

if __name__ == "__main__":
    main()
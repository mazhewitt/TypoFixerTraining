#!/usr/bin/env python3
"""
Validation script for DistilBERT typo correction model.
Evaluates token-level accuracy on corrupted→clean reconstruction.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TypoValidator:
    def __init__(self, model_dir: str):
        """Initialize validator with trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {model_dir}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForMaskedLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def predict_tokens(self, corrupted_text: str, max_length: int = 64) -> str:
        """Predict corrected text using the trained model."""
        # Tokenize input
        inputs = self.tokenizer(
            corrupted_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Decode predictions
        predicted_text = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
        return predicted_text.strip()
    
    def calculate_token_accuracy(self, predicted: str, target: str) -> Tuple[float, int, int]:
        """Calculate token-level accuracy between predicted and target text."""
        pred_tokens = predicted.split()
        target_tokens = target.split()
        
        # Align tokens for comparison (simple approach)
        max_len = max(len(pred_tokens), len(target_tokens))
        
        correct = 0
        total = max_len
        
        for i in range(max_len):
            pred_token = pred_tokens[i] if i < len(pred_tokens) else ""
            target_token = target_tokens[i] if i < len(target_tokens) else ""
            
            if pred_token == target_token:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, correct, total
    
    def evaluate_dataset(self, test_file: str, max_samples: int = None) -> Dict[str, float]:
        """Evaluate model on test dataset."""
        logger.info(f"Evaluating on {test_file}")
        
        total_accuracy = 0.0
        total_correct = 0
        total_tokens = 0
        sample_count = 0
        
        examples = []
        
        with open(test_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(tqdm(f, desc="Evaluating")):
                if max_samples and line_idx >= max_samples:
                    break
                
                data = json.loads(line.strip())
                corrupted = data['corrupted']
                clean = data['clean']
                
                # Get model prediction
                predicted = self.predict_tokens(corrupted)
                
                # Calculate accuracy
                accuracy, correct, total = self.calculate_token_accuracy(predicted, clean)
                
                total_accuracy += accuracy
                total_correct += correct
                total_tokens += total
                sample_count += 1
                
                # Store examples for analysis
                if len(examples) < 10:
                    examples.append({
                        'corrupted': corrupted,
                        'clean': clean,
                        'predicted': predicted,
                        'accuracy': accuracy
                    })
        
        # Calculate overall metrics
        avg_accuracy = total_accuracy / sample_count if sample_count > 0 else 0.0
        token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        results = {
            'average_sentence_accuracy': avg_accuracy,
            'token_level_accuracy': token_accuracy,
            'total_samples': sample_count,
            'total_tokens': total_tokens,
            'correct_tokens': total_correct
        }
        
        # Print results
        logger.info("=== Validation Results ===")
        logger.info(f"Samples evaluated: {sample_count:,}")
        logger.info(f"Average sentence accuracy: {avg_accuracy:.3f}")
        logger.info(f"Token-level accuracy: {token_accuracy:.3f}")
        logger.info(f"Correct tokens: {total_correct:,} / {total_tokens:,}")
        
        # Show example predictions
        logger.info("\n=== Example Predictions ===")
        for i, example in enumerate(examples[:5]):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Corrupted:  '{example['corrupted']}'")
            logger.info(f"Clean:      '{example['clean']}'")
            logger.info(f"Predicted:  '{example['predicted']}'")
            logger.info(f"Accuracy:   {example['accuracy']:.3f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Validate DistilBERT typo correction model")
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--test_file', type=str, required=True,
                       help='JSONL file with test examples')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--output_file', type=str, default=None,
                       help='JSON file to save results')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = TypoValidator(args.model_dir)
    
    # Run evaluation
    results = validator.evaluate_dataset(args.test_file, args.max_samples)
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")
    
    # Check if target accuracy achieved
    target_accuracy = 0.92
    if results['token_level_accuracy'] >= target_accuracy:
        logger.info(f"✅ Target accuracy {target_accuracy:.1%} achieved!")
    else:
        logger.warning(f"❌ Target accuracy {target_accuracy:.1%} not achieved. "
                      f"Current: {results['token_level_accuracy']:.1%}")

if __name__ == "__main__":
    main()
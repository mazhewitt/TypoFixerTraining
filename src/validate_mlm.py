#!/usr/bin/env python3
"""
MLM-specific validation for DistilBERT typo correction model.
Uses proper masked language modeling inference approach.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import re

import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher
from typo_correction import TypoCorrector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLMTypoValidator:
    def __init__(self, model_dir: str):
        """Initialize validator with trained MLM model using the new TypoCorrector."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {model_dir}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForMaskedLM.from_pretrained(model_dir)
        
        # Initialize the new TypoCorrector with optimized parameters for validation
        self.corrector = TypoCorrector(
            model=self.model,
            tokenizer=self.tokenizer,
            low_prob_threshold=-4.0,  # More sensitive for fine-tuned models
            edit_penalty_lambda=1.0,  # Lower penalty for more corrections
            top_k=8,
            max_passes=2,  # Fewer passes for faster validation
            device=str(self.device)
        )
        
        logger.info(f"Model loaded on {self.device} with TypoCorrector")
    
    
    def correct_text_mlm(self, corrupted_text: str, max_length: int = 128) -> str:
        """Correct text using the new TypoCorrector algorithm."""
        corrected_text, stats = self.corrector.correct_typos(corrupted_text)
        
        # Log correction stats for debugging
        if stats['total_corrections'] > 0:
            logger.debug(f"Corrections made: {stats['total_corrections']} in {stats['passes_used']} passes")
            for correction in stats['corrections_made']:
                logger.debug(f"  '{correction['original']}' -> '{correction['corrected']}' (score: {correction['score']:.2f})")
        
        return corrected_text
    
    def evaluate_corrections(self, test_cases: List[Tuple[str, str]], max_length: int = 128) -> Dict:
        """Evaluate model on test cases."""
        logger.info(f"Evaluating {len(test_cases)} test cases...")
        
        results = {
            'total_cases': len(test_cases),
            'corrections_attempted': 0,
            'exact_matches': 0,
            'partial_matches': 0,
            'token_accuracy': 0.0,
            'examples': []
        }
        
        total_tokens = 0
        correct_tokens = 0
        
        for i, (corrupted, expected) in enumerate(tqdm(test_cases, desc="Evaluating")):
            predicted = self.correct_text_mlm(corrupted, max_length)
            
            # Check if correction was attempted
            if predicted != corrupted:
                results['corrections_attempted'] += 1
            
            # Check exact match
            if predicted.lower().strip() == expected.lower().strip():
                results['exact_matches'] += 1
            
            # Check partial match (token-level accuracy)
            pred_tokens = predicted.lower().split()
            exp_tokens = expected.lower().split()
            
            # Align tokens for comparison
            max_len = max(len(pred_tokens), len(exp_tokens))
            for j in range(max_len):
                total_tokens += 1
                pred_token = pred_tokens[j] if j < len(pred_tokens) else ""
                exp_token = exp_tokens[j] if j < len(exp_tokens) else ""
                
                if pred_token == exp_token:
                    correct_tokens += 1
            
            # Check partial improvement
            orig_tokens = corrupted.lower().split()
            improvements = 0
            for j in range(min(len(pred_tokens), len(exp_tokens), len(orig_tokens))):
                if pred_tokens[j] == exp_tokens[j] and orig_tokens[j] != exp_tokens[j]:
                    improvements += 1
            
            if improvements > 0:
                results['partial_matches'] += 1
            
            # Store examples
            if len(results['examples']) < 10:
                results['examples'].append({
                    'corrupted': corrupted,
                    'expected': expected,
                    'predicted': predicted,
                    'improvements': improvements
                })
        
        # Calculate final metrics
        results['token_accuracy'] = correct_tokens / total_tokens if total_tokens > 0 else 0
        results['exact_match_rate'] = results['exact_matches'] / len(test_cases)
        results['correction_attempt_rate'] = results['corrections_attempted'] / len(test_cases)
        results['partial_match_rate'] = results['partial_matches'] / len(test_cases)
        
        return results
    
    def mlm_evaluate(self, corrupted: str, expected: str, max_length: int) -> str:
        """Evaluate by masking all positions where corrupted != expected."""
        tokens = self.tokenizer(corrupted, truncation=True, padding='max_length',
                                max_length=max_length, return_tensors='pt')
        labels = self.tokenizer(expected, truncation=True, padding='max_length',
                                max_length=max_length, return_tensors='pt')['input_ids']

        mask = tokens['input_ids'] != labels
        masked_input = tokens['input_ids'].clone()
        masked_input[mask] = self.tokenizer.mask_token_id
        # Move tensors to the same device as the model
        masked_input = masked_input.to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        input_ids_device = tokens['input_ids'].to(self.device)

        with torch.no_grad():
            logits = self.model(masked_input, attention_mask=attention_mask).logits
            preds = logits.argmax(-1)

        # restore unchanged tokens
        preds[~mask] = input_ids_device[~mask]

        return self.tokenizer.decode(preds[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="MLM-based validation for DistilBERT typo correction")
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--test_file', type=str,
                       help='JSONL file with test examples')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--output_file', type=str,
                       help='JSON file to save results')
    
    args = parser.parse_args()
    
    logger.info("🎯 Starting MLM-based validation...")
    
    # Initialize validator
    validator = MLMTypoValidator(args.model_dir)
    
    # Prepare test cases
    if args.test_file:
        logger.info(f"Loading test cases from {args.test_file}")
        test_cases = []
        with open(args.test_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= args.max_samples:
                    break
                data = json.loads(line.strip())
                test_cases.append((data['corrupted'], data['clean']))
    else:
        logger.info("Using built-in test cases")
        test_cases = [
            ("Thi sis a test sentenc with typos", "This is a test sentence with typos"),
            ("The quikc brown fox jumps over teh lazy dog", "The quick brown fox jumps over the lazy dog"),
            ("I went too the stor to buy som milk", "I went to the store to buy some milk"),
            ("Ther are many mistaks in this sentance", "There are many mistakes in this sentence"),
            ("Its a beutiful day outsid today", "It's a beautiful day outside today"),
            ("Can you help me wiht this problme?", "Can you help me with this problem?"),
            ("They're going to there house over their", "They're going to their house over there"),
            ("Your presentation was excellent, you're very talented", "Your presentation was excellent, you're very talented"),
            ("The affect of this change will effect everyone", "The effect of this change will affect everyone"),
            ("I need advise about how to advice my team", "I need advice about how to advise my team")
        ]
    
    logger.info(f"Evaluating on {len(test_cases)} test cases...")
    
    # Run evaluation
    results = validator.evaluate_corrections(test_cases, args.max_length)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("🎯 MLM VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"📊 Total test cases: {results['total_cases']}")
    logger.info(f"📊 Exact matches: {results['exact_matches']} ({results['exact_match_rate']*100:.1f}%)")
    logger.info(f"📊 Partial improvements: {results['partial_matches']} ({results['partial_match_rate']*100:.1f}%)")
    logger.info(f"📊 Corrections attempted: {results['corrections_attempted']} ({results['correction_attempt_rate']*100:.1f}%)")
    logger.info(f"📊 Token accuracy: {results['token_accuracy']*100:.1f}%")
    
    # Show examples
    logger.info(f"\n📝 Example Results:")
    for i, example in enumerate(results['examples'][:5], 1):
        logger.info(f"\n{i}. Corrupted:  '{example['corrupted']}'")
        logger.info(f"   Expected:   '{example['expected']}'")
        logger.info(f"   Predicted:  '{example['predicted']}'")
        logger.info(f"   Improvements: {example['improvements']}")
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {args.output_file}")
    
    logger.info("="*60)
    
    # Return success/failure based on performance
    if results['token_accuracy'] >= 0.5 or results['partial_match_rate'] >= 0.7:
        logger.info("✅ Model shows reasonable typo correction capability!")
        return 0
    else:
        logger.info("⚠️ Model performance below expected threshold")
        return 1

if __name__ == "__main__":
    exit(main())
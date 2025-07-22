#!/usr/bin/env python3
"""
Production validation script for DistilBERT typo correction model.
Comprehensive evaluation including token accuracy, BLEU scores, and error analysis.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

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
    
    def predict_tokens(self, corrupted_text: str, max_length: int = 128) -> str:
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
    
    def calculate_token_accuracy(self, corrupted_texts: List[str], clean_texts: List[str], 
                                max_length: int = 128) -> Dict[str, float]:
        """Calculate comprehensive token-level accuracy metrics."""
        logger.info(f"Calculating token accuracy on {len(corrupted_texts)} examples...")
        
        total_tokens = 0
        correct_tokens = 0
        perfect_matches = 0
        corrections_made = 0
        
        # Track per-example metrics
        example_accuracies = []
        
        for corrupted, clean in tqdm(zip(corrupted_texts, clean_texts), 
                                   total=len(corrupted_texts), desc="Evaluating"):
            predicted = self.predict_tokens(corrupted, max_length)
            
            # Tokenize all texts for fair comparison
            clean_tokens = self.tokenizer.encode(clean, add_special_tokens=False)
            predicted_tokens = self.tokenizer.encode(predicted, add_special_tokens=False)
            corrupted_tokens = self.tokenizer.encode(corrupted, add_special_tokens=False)
            
            # Pad to same length for comparison
            max_len = max(len(clean_tokens), len(predicted_tokens))
            clean_tokens += [self.tokenizer.pad_token_id] * (max_len - len(clean_tokens))
            predicted_tokens += [self.tokenizer.pad_token_id] * (max_len - len(predicted_tokens))
            
            # Calculate token accuracy for this example
            example_correct = sum(1 for c, p in zip(clean_tokens, predicted_tokens) if c == p)
            example_total = len(clean_tokens)
            example_accuracy = example_correct / example_total if example_total > 0 else 0
            
            example_accuracies.append(example_accuracy)
            
            # Update totals
            correct_tokens += example_correct
            total_tokens += example_total
            
            # Check perfect match
            if predicted.lower().strip() == clean.lower().strip():
                perfect_matches += 1
            
            # Check if any correction was attempted
            if predicted != corrupted:
                corrections_made += 1
        
        metrics = {
            'token_accuracy': correct_tokens / total_tokens if total_tokens > 0 else 0,
            'perfect_match_rate': perfect_matches / len(corrupted_texts),
            'correction_attempt_rate': corrections_made / len(corrupted_texts),
            'mean_example_accuracy': np.mean(example_accuracies),
            'median_example_accuracy': np.median(example_accuracies),
            'std_example_accuracy': np.std(example_accuracies),
            'total_examples': len(corrupted_texts),
            'total_tokens': total_tokens,
            'correct_tokens': correct_tokens
        }
        
        return metrics
    
    def analyze_error_types(self, corrupted_texts: List[str], clean_texts: List[str],
                           max_length: int = 128, sample_size: int = 100) -> Dict:
        """Analyze types of errors the model makes."""
        logger.info(f"Analyzing error types on {sample_size} examples...")
        
        # Sample subset for detailed analysis
        indices = np.random.choice(len(corrupted_texts), min(sample_size, len(corrupted_texts)), replace=False)
        
        error_analysis = {
            'overcorrection': 0,  # Model changes correct words
            'undercorrection': 0,  # Model misses obvious typos
            'wrong_correction': 0,  # Model attempts correction but gets it wrong
            'successful_correction': 0,  # Model correctly fixes typos
            'examples': []
        }
        
        for i in tqdm(indices, desc="Error analysis"):
            corrupted = corrupted_texts[i]
            clean = clean_texts[i]
            predicted = self.predict_tokens(corrupted, max_length)
            
            # Tokenize for analysis
            clean_words = clean.lower().split()
            corrupted_words = corrupted.lower().split()
            predicted_words = predicted.lower().split()
            
            # Analyze each position
            max_words = max(len(clean_words), len(corrupted_words), len(predicted_words))
            
            for j in range(max_words):
                clean_word = clean_words[j] if j < len(clean_words) else ""
                corrupted_word = corrupted_words[j] if j < len(corrupted_words) else ""
                predicted_word = predicted_words[j] if j < len(predicted_words) else ""
                
                if clean_word == corrupted_word and predicted_word != clean_word:
                    error_analysis['overcorrection'] += 1
                elif clean_word != corrupted_word and predicted_word == corrupted_word:
                    error_analysis['undercorrection'] += 1
                elif clean_word != corrupted_word and predicted_word != clean_word and predicted_word != corrupted_word:
                    error_analysis['wrong_correction'] += 1
                elif clean_word != corrupted_word and predicted_word == clean_word:
                    error_analysis['successful_correction'] += 1
            
            # Save example for inspection
            if len(error_analysis['examples']) < 20:
                error_analysis['examples'].append({
                    'corrupted': corrupted,
                    'clean': clean,
                    'predicted': predicted
                })
        
        return error_analysis
    
    def benchmark_speed(self, test_texts: List[str], max_length: int = 128, 
                       num_runs: int = 50) -> Dict[str, float]:
        """Benchmark model inference speed."""
        logger.info(f"Benchmarking inference speed with {num_runs} runs...")
        
        # Select random test texts
        test_sample = np.random.choice(test_texts, min(num_runs, len(test_texts)), replace=True)
        
        # Warmup
        for _ in range(5):
            self.predict_tokens(test_sample[0], max_length)
        
        # Benchmark
        times = []
        for text in tqdm(test_sample, desc="Benchmarking"):
            start_time = time.time()
            self.predict_tokens(text, max_length)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time_ms': np.mean(times) * 1000,
            'median_time_ms': np.median(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'throughput_per_sec': 1.0 / np.mean(times),
            'total_runs': num_runs
        }

def load_test_data(test_file: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Load test data from JSONL file."""
    corrupted_texts = []
    clean_texts = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            data = json.loads(line.strip())
            corrupted_texts.append(data['corrupted'])
            clean_texts.append(data['clean'])
    
    return corrupted_texts, clean_texts

def main():
    parser = argparse.ArgumentParser(description="Production validation for DistilBERT typo correction model")
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--test_file', type=str, 
                       help='JSONL file with test examples (if not provided, uses built-in examples)')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples to evaluate (default: 1000)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='JSON file to save results')
    parser.add_argument('--target_accuracy', type=float, default=0.92,
                       help='Target token accuracy threshold (default: 0.92)')
    parser.add_argument('--benchmark_speed', action='store_true',
                       help='Include speed benchmarking')
    parser.add_argument('--error_analysis', action='store_true',
                       help='Perform detailed error analysis')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length (default: 128)')
    
    args = parser.parse_args()
    
    logger.info("🎯 Starting production model validation...")
    logger.info(f"📁 Model directory: {args.model_dir}")
    logger.info(f"🎯 Target accuracy: {args.target_accuracy:.1%}")
    
    # Initialize validator
    validator = TypoValidator(args.model_dir)
    
    # Load test data
    if args.test_file:
        logger.info(f"📖 Loading test data from {args.test_file}")
        corrupted_texts, clean_texts = load_test_data(args.test_file, args.max_samples)
    else:
        logger.info("📖 Using built-in test examples")
        # Built-in test examples for quick validation
        corrupted_texts = [
            "Thi sis a test sentenc with typos",
            "The quikc brown fox jumps over teh lazy dog",
            "I went too the stor to buy som milk",
            "Ther are many mistaks in this sentance",
            "Its a beutiful day outsid today",
            "Can you help me wiht this problme?",
            "They're going to there house over their",
            "Your presentation was excellent, you're very talented",
            "The affect of this change will effect everyone",
            "I need advise about how to advice my team"
        ]
        clean_texts = [
            "This is a test sentence with typos",
            "The quick brown fox jumps over the lazy dog", 
            "I went to the store to buy some milk",
            "There are many mistakes in this sentence",
            "It's a beautiful day outside today",
            "Can you help me with this problem?",
            "They're going to their house over there",
            "Your presentation was excellent, you're very talented",
            "The effect of this change will affect everyone",
            "I need advice about how to advise my team"
        ]
    
    logger.info(f"📊 Loaded {len(corrupted_texts):,} test examples")
    
    # Run comprehensive evaluation
    results = {}
    
    # 1. Token accuracy evaluation
    logger.info("1️⃣ Calculating token accuracy...")
    accuracy_metrics = validator.calculate_token_accuracy(
        corrupted_texts, clean_texts, args.max_length
    )
    results['accuracy_metrics'] = accuracy_metrics
    
    # 2. Speed benchmarking (optional)
    if args.benchmark_speed:
        logger.info("2️⃣ Benchmarking inference speed...")
        speed_metrics = validator.benchmark_speed(
            corrupted_texts, args.max_length
        )
        results['speed_metrics'] = speed_metrics
    
    # 3. Error analysis (optional)
    if args.error_analysis:
        logger.info("3️⃣ Performing error analysis...")
        error_metrics = validator.analyze_error_types(
            corrupted_texts, clean_texts, args.max_length
        )
        results['error_analysis'] = error_metrics
    
    # Print comprehensive results
    logger.info("\n" + "="*60)
    logger.info("🎯 PRODUCTION VALIDATION RESULTS")
    logger.info("="*60)
    
    # Accuracy results
    acc = accuracy_metrics
    logger.info(f"📊 Token Accuracy: {acc['token_accuracy']:.3f} ({acc['token_accuracy']*100:.1f}%)")
    logger.info(f"📊 Perfect Matches: {acc['perfect_match_rate']:.3f} ({acc['perfect_match_rate']*100:.1f}%)")
    logger.info(f"📊 Correction Attempts: {acc['correction_attempt_rate']:.3f} ({acc['correction_attempt_rate']*100:.1f}%)")
    logger.info(f"📊 Mean Example Accuracy: {acc['mean_example_accuracy']:.3f}")
    logger.info(f"📊 Examples Evaluated: {acc['total_examples']:,}")
    logger.info(f"📊 Total Tokens: {acc['total_tokens']:,}")
    
    # Speed results
    if args.benchmark_speed:
        speed = results['speed_metrics']
        logger.info(f"⚡ Average Inference Time: {speed['mean_time_ms']:.1f}ms")
        logger.info(f"⚡ Throughput: {speed['throughput_per_sec']:.1f} corrections/sec")
    
    # Error analysis results
    if args.error_analysis:
        errors = results['error_analysis']
        total_errors = sum(errors[k] for k in ['overcorrection', 'undercorrection', 'wrong_correction', 'successful_correction'])
        logger.info(f"\n🔍 Error Analysis (sample of {total_errors} word corrections):")
        logger.info(f"   ✅ Successful corrections: {errors['successful_correction']}")
        logger.info(f"   ❌ Wrong corrections: {errors['wrong_correction']}")
        logger.info(f"   ⚠️ Overcorrections: {errors['overcorrection']}")
        logger.info(f"   ⚠️ Undercorrections: {errors['undercorrection']}")
    
    # Show examples
    logger.info(f"\n📝 Example Corrections:")
    for i in range(min(5, len(corrupted_texts))):
        predicted = validator.predict_tokens(corrupted_texts[i], args.max_length)
        logger.info(f"  {i+1}. Input:     '{corrupted_texts[i]}'")
        logger.info(f"     Expected:  '{clean_texts[i]}'")
        logger.info(f"     Predicted: '{predicted}'")
        logger.info("")
    
    # Check if target accuracy achieved
    token_acc = accuracy_metrics['token_accuracy']
    logger.info("="*60)
    if token_acc >= args.target_accuracy:
        logger.info(f"✅ SUCCESS: Target accuracy {args.target_accuracy:.1%} achieved!")
        logger.info(f"🎉 Model achieved {token_acc:.1%} token accuracy")
        success = True
    else:
        logger.info(f"❌ Target accuracy {args.target_accuracy:.1%} not achieved")
        logger.info(f"📊 Current accuracy: {token_acc:.1%}")
        logger.info(f"📈 Need improvement: {args.target_accuracy - token_acc:.3f}")
        success = False
    
    logger.info("="*60)
    
    # Save results
    results['validation_summary'] = {
        'target_accuracy': args.target_accuracy,
        'achieved_accuracy': token_acc,
        'success': success,
        'model_dir': args.model_dir,
        'test_samples': len(corrupted_texts),
        'timestamp': time.time()
    }
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {args.output_file}")
    
    # Return appropriate exit code
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
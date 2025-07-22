#!/usr/bin/env python3
"""
Baseline testing for Qwen3-0.6B on typo correction task.
Tests the model's out-of-the-box performance before fine-tuning.
"""

import argparse
import json
import logging
import time
from typing import List, Tuple, Dict
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenBaselineTester:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """Initialize Qwen model for baseline testing."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto" if self.device.type == 'cuda' else None,
            trust_remote_code=True
        )
        
        if self.device.type == 'cpu':
            self.model.to(self.device)
            
        self.model.eval()
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded on {self.device}")
    
    def correct_text_with_prompt(self, corrupted_text: str, max_new_tokens: int = 50) -> str:
        """Use Qwen with a correction prompt to fix typos."""
        
        # Create a clear instruction prompt for typo correction
        prompt = f"""Fix the typos in this text. Only return the corrected text, nothing else.

Text with typos: {corrupted_text}
Corrected text:"""
        
        # Tokenize with attention to avoid warnings
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        ).to(self.device)
        
        # Generate correction
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature for consistency
                do_sample=False,  # Deterministic for baseline testing
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            end_time = time.time()
        
        # Decode the generated text
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the corrected text (after "Corrected text:")
        if "Corrected text:" in full_output:
            correction = full_output.split("Corrected text:")[-1].strip()
        else:
            # Fallback: take everything after the original prompt
            correction = full_output[len(prompt):].strip()
        
        # Clean up the correction
        correction = re.sub(r'^\s*["\']|["\']?\s*$', '', correction)  # Remove quotes
        correction = correction.split('\n')[0].strip()  # Take first line only
        
        inference_time = end_time - start_time
        return correction, inference_time
    
    def evaluate_baseline(self, test_cases: List[Tuple[str, str]]) -> Dict:
        """Evaluate Qwen baseline performance on test cases."""
        logger.info(f"Evaluating Qwen baseline on {len(test_cases)} test cases...")
        
        results = {
            'model_name': 'Qwen/Qwen3-0.6B',
            'total_cases': len(test_cases),
            'exact_matches': 0,
            'partial_improvements': 0,
            'token_accuracy': 0.0,
            'avg_inference_time': 0.0,
            'examples': []
        }
        
        total_tokens = 0
        correct_tokens = 0
        total_time = 0
        
        for i, (corrupted, expected) in enumerate(tqdm(test_cases, desc="Testing baseline")):
            try:
                predicted, inference_time = self.correct_text_with_prompt(corrupted)
                total_time += inference_time
                
                # Check exact match (case insensitive)
                if predicted.lower().strip() == expected.lower().strip():
                    results['exact_matches'] += 1
                
                # Calculate token-level accuracy
                pred_tokens = predicted.lower().split()
                exp_tokens = expected.lower().split()
                
                # Align tokens for comparison (simplified)
                max_len = max(len(pred_tokens), len(exp_tokens))
                for j in range(max_len):
                    total_tokens += 1
                    pred_token = pred_tokens[j] if j < len(pred_tokens) else ""
                    exp_token = exp_tokens[j] if j < len(exp_tokens) else ""
                    
                    if pred_token == exp_token:
                        correct_tokens += 1
                
                # Check for partial improvements
                orig_tokens = corrupted.lower().split()
                improvements = 0
                for j in range(min(len(pred_tokens), len(exp_tokens), len(orig_tokens))):
                    if pred_tokens[j] == exp_tokens[j] and orig_tokens[j] != exp_tokens[j]:
                        improvements += 1
                
                if improvements > 0:
                    results['partial_improvements'] += 1
                
                # Store example
                results['examples'].append({
                    'corrupted': corrupted,
                    'expected': expected,
                    'predicted': predicted,
                    'improvements': improvements,
                    'inference_time': inference_time
                })
                
            except Exception as e:
                logger.error(f"Error processing case {i}: {e}")
                results['examples'].append({
                    'corrupted': corrupted,
                    'expected': expected,
                    'predicted': f"ERROR: {str(e)}",
                    'improvements': 0,
                    'inference_time': 0
                })
        
        # Calculate final metrics
        results['token_accuracy'] = correct_tokens / total_tokens if total_tokens > 0 else 0
        results['exact_match_rate'] = results['exact_matches'] / len(test_cases)
        results['partial_improvement_rate'] = results['partial_improvements'] / len(test_cases)
        results['avg_inference_time'] = total_time / len(test_cases) if len(test_cases) > 0 else 0
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Baseline testing for Qwen3-0.6B typo correction")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-0.6B",
                       help='Qwen model name to test')
    parser.add_argument('--test_file', type=str,
                       help='JSONL file with test examples (optional)')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum number of samples to test')
    parser.add_argument('--output_file', type=str,
                       help='JSON file to save results')
    
    args = parser.parse_args()
    
    logger.info("🎯 Starting Qwen3-0.6B baseline testing...")
    
    # Initialize tester
    tester = QwenBaselineTester(args.model_name)
    
    # Prepare test cases (same as the BERT validation)
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
        logger.info("Using built-in test cases (same as BERT validation)")
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
    
    # Limit test cases if requested
    test_cases = test_cases[:args.max_samples]
    logger.info(f"Testing on {len(test_cases)} cases...")
    
    # Run baseline evaluation
    results = tester.evaluate_baseline(test_cases)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("🎯 QWEN3-0.6B BASELINE RESULTS")
    logger.info("="*60)
    logger.info(f"📊 Model: {results['model_name']}")
    logger.info(f"📊 Total test cases: {results['total_cases']}")
    logger.info(f"📊 Exact matches: {results['exact_matches']} ({results['exact_match_rate']*100:.1f}%)")
    logger.info(f"📊 Partial improvements: {results['partial_improvements']} ({results['partial_improvement_rate']*100:.1f}%)")
    logger.info(f"📊 Token accuracy: {results['token_accuracy']*100:.1f}%")
    logger.info(f"📊 Average inference time: {results['avg_inference_time']*1000:.1f}ms")
    
    # Show examples
    logger.info(f"\n📝 Example Results:")
    for i, example in enumerate(results['examples'][:5], 1):
        logger.info(f"\n{i}. Corrupted:  '{example['corrupted']}'")
        logger.info(f"   Expected:   '{example['expected']}'")
        logger.info(f"   Predicted:  '{example['predicted']}'")
        logger.info(f"   Improvements: {example['improvements']}")
        logger.info(f"   Time: {example['inference_time']*1000:.1f}ms")
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {args.output_file}")
    
    logger.info("="*60)
    
    # Summary assessment
    if results['exact_match_rate'] >= 0.3 or results['token_accuracy'] >= 0.7:
        logger.info("✅ Qwen shows promising baseline capability for typo correction!")
    else:
        logger.info("ℹ️ Qwen baseline results - room for improvement with fine-tuning")

if __name__ == "__main__":
    main()
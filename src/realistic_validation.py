#!/usr/bin/env python3
"""
Realistic sentence validation for Qwen typo correction.
Focuses on single-sentence corrections with natural text patterns.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticSentenceValidator:
    def __init__(self, model_path: str):
        """Initialize with focus on single sentence correction."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading Qwen model for realistic sentence validation...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto" if self.device.type == 'cuda' else None,
            trust_remote_code=True
        )
        
        if self.device.type == 'cpu':
            self.model.to(self.device)
            
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model ready for realistic sentence validation")
    
    def correct_sentence(self, corrupted: str) -> Tuple[str, float]:
        """Generate correction for a single corrupted sentence."""
        
        # Simple, direct prompt for sentence correction
        prompt = f"Correct the typos: {corrupted}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # Keep it short for single sentences
        ).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,  # Short correction for single sentence
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        end_time = time.time()
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract correction (after the prompt)
        if "Correct the typos:" in generated_text:
            correction = generated_text.split("Correct the typos:")[-1].strip()
            # Remove the original corrupted text if it's repeated
            if correction.startswith(corrupted):
                correction = correction[len(corrupted):].strip()
        else:
            correction = generated_text[len(prompt):].strip()
        
        # Clean up
        correction = re.sub(r'^\s*["\']|["\']?\s*$', '', correction)
        correction = correction.split('\n')[0].strip()
        
        inference_time = end_time - start_time
        return correction, inference_time
    
    def calculate_sentence_metrics(self, predicted: str, expected: str) -> Dict[str, float]:
        """Calculate metrics focused on sentence-level correction quality."""
        
        pred_words = predicted.lower().split()
        exp_words = expected.lower().split()
        
        # Word-level accuracy
        correct_words = sum(1 for p, e in zip(pred_words, exp_words) if p == e)
        total_words = max(len(pred_words), len(exp_words))
        word_accuracy = correct_words / total_words if total_words > 0 else 0
        
        # Character-level edit distance
        def edit_distance(s1, s2):
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            distances = range(len(s1) + 1)
            for i2, c2 in enumerate(s2):
                new_distances = [i2 + 1]
                for i1, c1 in enumerate(s1):
                    if c1 == c2:
                        new_distances.append(distances[i1])
                    else:
                        new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
                distances = new_distances
            return distances[-1]
        
        edit_dist = edit_distance(predicted.lower(), expected.lower())
        max_len = max(len(predicted), len(expected))
        edit_similarity = 1 - (edit_dist / max_len) if max_len > 0 else 1
        
        # Exact match
        exact_match = predicted.lower().strip() == expected.lower().strip()
        
        return {
            'exact_match': exact_match,
            'word_accuracy': word_accuracy,
            'edit_similarity': edit_similarity,
            'word_count_diff': abs(len(pred_words) - len(exp_words))
        }
    
    def validate_realistic_sentences(self, test_data: List[Dict]) -> Dict:
        """Validate on realistic sentence data."""
        
        logger.info(f"🎯 Validating on {len(test_data)} realistic sentences...")
        
        results = {
            'total_sentences': len(test_data),
            'exact_matches': 0,
            'avg_word_accuracy': 0.0,
            'avg_edit_similarity': 0.0,
            'avg_inference_time': 0.0,
            'complexity_breakdown': {
                'simple': {'count': 0, 'exact_matches': 0, 'word_accuracy': 0.0},
                'medium': {'count': 0, 'exact_matches': 0, 'word_accuracy': 0.0},
                'complex': {'count': 0, 'exact_matches': 0, 'word_accuracy': 0.0}
            },
            'examples': []
        }
        
        total_word_accuracy = 0
        total_edit_similarity = 0
        total_time = 0
        
        for item in tqdm(test_data, desc="Validating sentences"):
            corrupted = item['corrupted']
            expected = item['clean']
            complexity = item.get('complexity', 'unknown')
            
            try:
                predicted, inference_time = self.correct_sentence(corrupted)
                total_time += inference_time
                
                # Calculate metrics
                metrics = self.calculate_sentence_metrics(predicted, expected)
                
                if metrics['exact_match']:
                    results['exact_matches'] += 1
                    if complexity in results['complexity_breakdown']:
                        results['complexity_breakdown'][complexity]['exact_matches'] += 1
                
                total_word_accuracy += metrics['word_accuracy']
                total_edit_similarity += metrics['edit_similarity']
                
                # Update complexity stats
                if complexity in results['complexity_breakdown']:
                    results['complexity_breakdown'][complexity]['count'] += 1
                    results['complexity_breakdown'][complexity]['word_accuracy'] += metrics['word_accuracy']
                
                # Store example
                if len(results['examples']) < 10:
                    results['examples'].append({
                        'corrupted': corrupted,
                        'expected': expected,
                        'predicted': predicted,
                        'complexity': complexity,
                        'metrics': metrics,
                        'inference_time': inference_time
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing sentence: {e}")
        
        # Calculate averages
        n = len(test_data)
        results['exact_match_rate'] = results['exact_matches'] / n
        results['avg_word_accuracy'] = total_word_accuracy / n
        results['avg_edit_similarity'] = total_edit_similarity / n
        results['avg_inference_time'] = total_time / n
        
        # Calculate complexity breakdowns
        for complexity_data in results['complexity_breakdown'].values():
            if complexity_data['count'] > 0:
                complexity_data['exact_match_rate'] = complexity_data['exact_matches'] / complexity_data['count']
                complexity_data['word_accuracy'] = complexity_data['word_accuracy'] / complexity_data['count']
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Validate Qwen on realistic single sentences")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to Qwen model')
    parser.add_argument('--test_file', type=str, required=True,
                       help='JSONL file with realistic test data')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum samples to test')
    parser.add_argument('--output_file', type=str,
                       help='JSON file to save results')
    
    args = parser.parse_args()
    
    # Load test data
    logger.info(f"📂 Loading test data from {args.test_file}")
    test_data = []
    
    with open(args.test_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            data = json.loads(line.strip())
            test_data.append(data)
    
    logger.info(f"📊 Loaded {len(test_data)} test sentences")
    
    # Run validation
    validator = RealisticSentenceValidator(args.model_path)
    results = validator.validate_realistic_sentences(test_data)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("🎯 REALISTIC SENTENCE VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"📊 Total sentences: {results['total_sentences']}")
    logger.info(f"📊 Exact matches: {results['exact_matches']} ({results['exact_match_rate']*100:.1f}%)")
    logger.info(f"📊 Average word accuracy: {results['avg_word_accuracy']*100:.1f}%")
    logger.info(f"📊 Average edit similarity: {results['avg_edit_similarity']*100:.1f}%")
    logger.info(f"📊 Average inference time: {results['avg_inference_time']*1000:.1f}ms")
    
    # Complexity breakdown
    logger.info(f"\n📋 Performance by sentence complexity:")
    for complexity, stats in results['complexity_breakdown'].items():
        if stats['count'] > 0:
            logger.info(f"   {complexity.capitalize()}: {stats['exact_matches']}/{stats['count']} exact matches ({stats['exact_match_rate']*100:.1f}%), {stats['word_accuracy']*100:.1f}% word accuracy")
    
    # Show examples
    logger.info(f"\n📝 Example corrections:")
    for i, example in enumerate(results['examples'][:3], 1):
        logger.info(f"\n{i}. {example['complexity'].capitalize()} sentence:")
        logger.info(f"   Corrupted:  \"{example['corrupted']}\"")
        logger.info(f"   Expected:   \"{example['expected']}\"")
        logger.info(f"   Predicted:  \"{example['predicted']}\"")
        logger.info(f"   Exact match: {example['metrics']['exact_match']}")
        logger.info(f"   Word accuracy: {example['metrics']['word_accuracy']*100:.1f}%")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {args.output_file}")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()
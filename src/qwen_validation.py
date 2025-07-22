#!/usr/bin/env python3
"""
Validation pipeline for Qwen-based typo correction with generative metrics.
Evaluates text-to-text generation quality using BLEU, ROUGE, and semantic similarity.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenValidator:
    def __init__(self, model_path: str):
        """Initialize Qwen validator."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading Qwen model from {model_path}")
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
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded on {self.device}")
    
    def correct_text(self, corrupted_text: str, max_new_tokens: int = 100) -> Tuple[str, float]:
        """Generate correction for corrupted text."""
        
        # Create correction prompt
        prompt = f"""Fix all the typos and errors in this text. Return only the corrected text:

Text: {corrupted_text}
Corrected text:"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Leave room for generation
        ).to(self.device)
        
        # Generate correction
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,  # Deterministic
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        end_time = time.time()
        
        # Extract generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the correction (after "Corrected text:")
        if "Corrected text:" in generated_text:
            correction = generated_text.split("Corrected text:")[-1].strip()
        else:
            # Fallback: take text after the prompt
            correction = generated_text[len(prompt):].strip()
        
        # Clean up
        correction = re.sub(r'^\s*["\']|["\']?\s*$', '', correction)
        correction = correction.split('\n')[0].strip()
        
        inference_time = end_time - start_time
        return correction, inference_time
    
    def calculate_bleu_score(self, predicted: str, reference: str) -> float:
        """Calculate BLEU score (simplified n-gram based)."""
        try:
            # Simple word-level BLEU approximation
            pred_tokens = predicted.lower().split()
            ref_tokens = reference.lower().split()
            
            if not pred_tokens or not ref_tokens:
                return 0.0
            
            # Calculate precision for n-grams (simplified to unigrams and bigrams)
            pred_unigrams = set(pred_tokens)
            ref_unigrams = set(ref_tokens)
            
            # Unigram precision
            unigram_matches = len(pred_unigrams & ref_unigrams)
            unigram_precision = unigram_matches / len(pred_unigrams) if pred_unigrams else 0
            
            # Bigram precision
            pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
            ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
            
            bigram_matches = len(pred_bigrams & ref_bigrams)
            bigram_precision = bigram_matches / len(pred_bigrams) if pred_bigrams else 0
            
            # Brevity penalty
            bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if ref_tokens else 0
            
            # Simplified BLEU (geometric mean of unigram and bigram precision with brevity penalty)
            if unigram_precision > 0 and bigram_precision > 0:
                bleu = bp * (unigram_precision * bigram_precision) ** 0.5
            else:
                bleu = bp * unigram_precision
                
            return bleu
            
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return 0.0
    
    def calculate_rouge_l(self, predicted: str, reference: str) -> float:
        """Calculate ROUGE-L score (longest common subsequence)."""
        try:
            pred_tokens = predicted.lower().split()
            ref_tokens = reference.lower().split()
            
            if not pred_tokens or not ref_tokens:
                return 0.0
            
            # Calculate LCS length
            def lcs_length(seq1, seq2):
                m, n = len(seq1), len(seq2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if seq1[i-1] == seq2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                return dp[m][n]
            
            lcs_len = lcs_length(pred_tokens, ref_tokens)
            
            # ROUGE-L = F-measure of LCS
            if lcs_len == 0:
                return 0.0
                
            precision = lcs_len / len(pred_tokens)
            recall = lcs_len / len(ref_tokens)
            
            if precision + recall == 0:
                return 0.0
                
            rouge_l = 2 * precision * recall / (precision + recall)
            return rouge_l
            
        except Exception as e:
            logger.warning(f"ROUGE-L calculation failed: {e}")
            return 0.0
    
    def calculate_edit_distance_similarity(self, predicted: str, reference: str) -> float:
        """Calculate similarity based on edit distance."""
        try:
            # Use Levenshtein distance at character level
            def levenshtein_distance(s1, s2):
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
            
            pred_clean = predicted.lower().strip()
            ref_clean = reference.lower().strip()
            
            if not pred_clean or not ref_clean:
                return 0.0
            
            max_len = max(len(pred_clean), len(ref_clean))
            if max_len == 0:
                return 1.0
                
            edit_dist = levenshtein_distance(pred_clean, ref_clean)
            similarity = 1.0 - (edit_dist / max_len)
            
            return max(0.0, similarity)
            
        except Exception as e:
            logger.warning(f"Edit distance calculation failed: {e}")
            return 0.0
    
    def evaluate_corrections(self, test_cases: List[Tuple[str, str]]) -> Dict:
        """Evaluate model with generative metrics."""
        logger.info(f"Evaluating Qwen on {len(test_cases)} test cases...")
        
        results = {
            'model_path': str(self.model.config._name_or_path) if hasattr(self.model.config, '_name_or_path') else 'qwen',
            'total_cases': len(test_cases),
            'exact_matches': 0,
            'bleu_scores': [],
            'rouge_l_scores': [],
            'edit_similarities': [],
            'avg_inference_time': 0.0,
            'examples': []
        }
        
        total_time = 0
        
        for i, (corrupted, expected) in enumerate(tqdm(test_cases, desc="Evaluating corrections")):
            try:
                predicted, inference_time = self.correct_text(corrupted)
                total_time += inference_time
                
                # Calculate metrics
                exact_match = predicted.lower().strip() == expected.lower().strip()
                if exact_match:
                    results['exact_matches'] += 1
                
                bleu_score = self.calculate_bleu_score(predicted, expected)
                rouge_score = self.calculate_rouge_l(predicted, expected)
                edit_sim = self.calculate_edit_distance_similarity(predicted, expected)
                
                results['bleu_scores'].append(bleu_score)
                results['rouge_l_scores'].append(rouge_score)
                results['edit_similarities'].append(edit_sim)
                
                # Store example
                results['examples'].append({
                    'corrupted': corrupted,
                    'expected': expected,
                    'predicted': predicted,
                    'exact_match': exact_match,
                    'bleu_score': bleu_score,
                    'rouge_l_score': rouge_score,
                    'edit_similarity': edit_sim,
                    'inference_time': inference_time
                })
                
            except Exception as e:
                logger.error(f"Error processing case {i}: {e}")
                results['examples'].append({
                    'corrupted': corrupted,
                    'expected': expected,
                    'predicted': f"ERROR: {str(e)}",
                    'exact_match': False,
                    'bleu_score': 0.0,
                    'rouge_l_score': 0.0,
                    'edit_similarity': 0.0,
                    'inference_time': 0.0
                })
        
        # Calculate averages
        results['exact_match_rate'] = results['exact_matches'] / len(test_cases)
        results['avg_bleu_score'] = np.mean(results['bleu_scores']) if results['bleu_scores'] else 0.0
        results['avg_rouge_l_score'] = np.mean(results['rouge_l_scores']) if results['rouge_l_scores'] else 0.0
        results['avg_edit_similarity'] = np.mean(results['edit_similarities']) if results['edit_similarities'] else 0.0
        results['avg_inference_time'] = total_time / len(test_cases)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Validate Qwen typo correction with generative metrics")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to Qwen model')
    parser.add_argument('--test_file', type=str,
                       help='JSONL file with test examples (optional)')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum number of samples to test')
    parser.add_argument('--output_file', type=str,
                       help='JSON file to save results')
    
    args = parser.parse_args()
    
    logger.info("🎯 Starting Qwen validation with generative metrics...")
    
    # Initialize validator
    validator = QwenValidator(args.model_path)
    
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
    
    test_cases = test_cases[:args.max_samples]
    logger.info(f"Testing on {len(test_cases)} cases...")
    
    # Run evaluation
    results = validator.evaluate_corrections(test_cases)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("🎯 QWEN VALIDATION RESULTS (GENERATIVE METRICS)")
    logger.info("="*60)
    logger.info(f"📊 Model: {results['model_path']}")
    logger.info(f"📊 Total test cases: {results['total_cases']}")
    logger.info(f"📊 Exact matches: {results['exact_matches']} ({results['exact_match_rate']*100:.1f}%)")
    logger.info(f"📊 Average BLEU score: {results['avg_bleu_score']:.3f}")
    logger.info(f"📊 Average ROUGE-L score: {results['avg_rouge_l_score']:.3f}")
    logger.info(f"📊 Average edit similarity: {results['avg_edit_similarity']:.3f}")
    logger.info(f"📊 Average inference time: {results['avg_inference_time']*1000:.1f}ms")
    
    # Show examples
    logger.info(f"\n📝 Example Results:")
    for i, example in enumerate(results['examples'][:3], 1):
        logger.info(f"\n{i}. Corrupted:  '{example['corrupted']}'")
        logger.info(f"   Expected:   '{example['expected']}'")
        logger.info(f"   Predicted:  '{example['predicted']}'")
        logger.info(f"   Exact match: {example['exact_match']}")
        logger.info(f"   BLEU: {example['bleu_score']:.3f}")
        logger.info(f"   ROUGE-L: {example['rouge_l_score']:.3f}")
        logger.info(f"   Edit sim: {example['edit_similarity']:.3f}")
        logger.info(f"   Time: {example['inference_time']*1000:.1f}ms")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {args.output_file}")
    
    logger.info("="*60)
    
    # Assessment
    overall_score = (results['avg_bleu_score'] + results['avg_rouge_l_score'] + results['avg_edit_similarity']) / 3
    if overall_score >= 0.7 or results['exact_match_rate'] >= 0.5:
        logger.info("✅ Qwen shows strong performance on typo correction!")
    else:
        logger.info("ℹ️ Qwen baseline results - good foundation for fine-tuning")

if __name__ == "__main__":
    main()
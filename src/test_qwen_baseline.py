#!/usr/bin/env python3
"""
Test script to evaluate Qwen's baseline typo correction performance.
"""

import argparse
import json
import random
from typing import List, Dict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_test_examples(data_file: str, num_samples: int = 100) -> List[Dict]:
    """Load test examples from the training data."""
    examples = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        all_examples = [json.loads(line.strip()) for line in f]
    
    # Sample random examples
    sampled = random.sample(all_examples, min(num_samples, len(all_examples)))
    
    for data in sampled:
        examples.append({
            'corrupted': data['corrupted'],
            'clean': data['clean'],
            'complexity': data.get('complexity', 'unknown'),
            'source': data.get('source', 'unknown')
        })
    
    return examples

def test_typo_correction(model, tokenizer, examples: List[Dict], device='cpu') -> Dict:
    """Test the model's typo correction performance."""
    model.eval()
    results = {
        'correct': 0,
        'total': 0,
        'by_complexity': {'simple': {'correct': 0, 'total': 0}, 
                          'medium': {'correct': 0, 'total': 0}, 
                          'complex': {'correct': 0, 'total': 0}},
        'by_source': {'holbrook': {'correct': 0, 'total': 0}, 
                      'wikitext': {'correct': 0, 'total': 0}},
        'examples': []
    }
    
    print(f"🧪 Testing {len(examples)} examples...")
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(examples)):
            corrupted = example['corrupted']
            expected = example['clean']
            complexity = example['complexity']
            source = example['source']
            
            # Create prompt
            prompt = f"Correct the typos: {corrupted}"
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=256
            ).to(device)
            
            # Generate correction
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.1,
                    num_return_sequences=1
                )
                
                # Decode generated text (skip prompt)
                generated_text = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[-1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Normalize for comparison
                pred_normalized = ' '.join(generated_text.lower().split())
                expected_normalized = ' '.join(expected.lower().split())
                
                # Check if correct
                is_correct = pred_normalized == expected_normalized
                
                # Update results
                results['total'] += 1
                results['by_complexity'][complexity]['total'] += 1
                results['by_source'][source]['total'] += 1
                
                if is_correct:
                    results['correct'] += 1
                    results['by_complexity'][complexity]['correct'] += 1
                    results['by_source'][source]['correct'] += 1
                
                # Store example for analysis
                example_result = {
                    'corrupted': corrupted,
                    'expected': expected,
                    'generated': generated_text,
                    'correct': is_correct,
                    'complexity': complexity,
                    'source': source
                }
                results['examples'].append(example_result)
                
                # Show progress for first few examples
                if i < 5:
                    status = "✅" if is_correct else "❌"
                    print(f"\n{status} Example {i+1}:")
                    print(f"   Corrupted: {corrupted}")
                    print(f"   Expected:  {expected}")
                    print(f"   Generated: {generated_text}")
                    print(f"   Complexity: {complexity}, Source: {source}")
                
            except Exception as e:
                print(f"⚠️ Error processing example {i+1}: {e}")
                results['total'] += 1
                results['by_complexity'][complexity]['total'] += 1
                results['by_source'][source]['total'] += 1
    
    return results

def print_results(results: Dict):
    """Print detailed results."""
    total = results['total']
    correct = results['correct']
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n📊 Overall Results:")
    print(f"   Total examples: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy:.1%}")
    
    print(f"\n📊 By Complexity:")
    for complexity, stats in results['by_complexity'].items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"   {complexity.capitalize()}: {stats['correct']}/{stats['total']} ({acc:.1%})")
    
    print(f"\n📊 By Source:")
    for source, stats in results['by_source'].items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"   {source.capitalize()}: {stats['correct']}/{stats['total']} ({acc:.1%})")
    
    # Show some failure examples
    print(f"\n❌ Sample Failures:")
    failures = [ex for ex in results['examples'] if not ex['correct']]
    for i, failure in enumerate(failures[:3]):
        print(f"\n{i+1}. Complexity: {failure['complexity']}")
        print(f"   Corrupted: {failure['corrupted']}")
        print(f"   Expected:  {failure['expected']}")
        print(f"   Generated: {failure['generated']}")

def main():
    parser = argparse.ArgumentParser(description="Test Qwen baseline typo correction")
    parser.add_argument('--model_path', type=str, default='models/qwen-0.6b',
                       help='Path to Qwen model')
    parser.add_argument('--test_file', type=str, default='data/enhanced_training_full.jsonl',
                       help='Test data file')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to test')
    parser.add_argument('--output', type=str,
                       help='Output file for detailed results')
    
    args = parser.parse_args()
    
    print("🤖 Loading Qwen model for baseline testing...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        pad_token='<|endoftext|>',
        eos_token='<|endoftext|>'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = next(model.parameters()).device
    print(f"✅ Model loaded on device: {device}")
    
    # Load test examples
    print(f"📖 Loading test examples from {args.test_file}")
    examples = load_test_examples(args.test_file, args.num_samples)
    
    # Run tests
    results = test_typo_correction(model, tokenizer, examples, device)
    
    # Print results
    print_results(results)
    
    # Save detailed results
    if args.output:
        print(f"💾 Saving detailed results to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary for training guidance
    accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
    print(f"\n🎯 Baseline Performance Summary:")
    print(f"   Current accuracy: {accuracy:.1%}")
    print(f"   Target accuracy: 90%")
    
    if accuracy < 0.9:
        improvement_needed = 0.9 - accuracy
        print(f"   Improvement needed: +{improvement_needed:.1%}")
        print(f"   📋 Recommendation: Fine-tuning is needed to reach 90% target")
    else:
        print(f"   🎉 Target already achieved! Fine-tuning may still improve consistency.")

if __name__ == "__main__":
    main()
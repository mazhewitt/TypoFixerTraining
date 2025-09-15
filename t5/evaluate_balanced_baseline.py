#!/usr/bin/env python3
"""
Updated baseline evaluation for T5-efficient-tiny with balanced punctuation examples.
Tests both with and without punctuation to match our training data.
"""

import json
import random
import time
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def evaluate_balanced_baseline():
    """Evaluate T5-efficient-tiny on balanced punctuation examples."""
    
    print("🧪 T5-efficient-tiny Balanced Baseline Evaluation")
    print("=" * 60)
    
    # Load model and tokenizer
    print("📥 Loading T5-efficient-tiny model...")
    tokenizer = T5Tokenizer.from_pretrained('google/t5-efficient-tiny', legacy=False)
    model = T5ForConditionalGeneration.from_pretrained('google/t5-efficient-tiny')
    
    # Move to MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🔧 Device: {device}")
    print()
    
    # Load sample from balanced training data
    print("📊 Loading sample from balanced training data...")
    training_samples = []
    
    try:
        with open('data/enhanced_training_balanced.jsonl', 'r') as f:
            lines = f.readlines()
            
            # Get examples with and without punctuation
            with_punct = []
            without_punct = []
            
            for line in lines:
                data = json.loads(line.strip())
                if data['clean'].endswith('.'):
                    with_punct.append(data)
                else:
                    without_punct.append(data)
            
            # Sample 8 from each category
            sample_with = random.sample(with_punct, min(8, len(with_punct)))
            sample_without = random.sample(without_punct, min(8, len(without_punct)))
            
            training_samples = sample_with + sample_without
            
    except FileNotFoundError:
        print("⚠️ Balanced training data not found, creating it first...")
        import subprocess
        subprocess.run(['python3', 'create_balanced_dataset.py'])
        
        # Try again
        with open('data/enhanced_training_balanced.jsonl', 'r') as f:
            lines = f.readlines()
            sample_lines = random.sample(lines, min(16, len(lines)))
            for line in sample_lines:
                training_samples.append(json.loads(line.strip()))
    
    print(f"📝 Testing on {len(training_samples)} balanced examples")
    
    # Categorize samples
    with_punct_samples = [s for s in training_samples if s['clean'].endswith('.')]
    without_punct_samples = [s for s in training_samples if not s['clean'].endswith('.')]
    
    print(f"  - With punctuation: {len(with_punct_samples)}")
    print(f"  - Without punctuation: {len(without_punct_samples)}")
    print()
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    # Test prompt format (best from previous analysis)
    prompt_format = "correct typos: {}"
    
    for i, test_case in enumerate(training_samples):
        has_punct = test_case['clean'].endswith('.')
        difficulty = test_case.get('complexity', 'unknown')
        
        print(f"📝 Test {i+1}/{len(training_samples)} ({'with' if has_punct else 'without'} punct, {difficulty})")
        print(f"   Input: '{test_case['corrupted']}'")
        print(f"   Expected: '{test_case['clean']}'")
        
        input_text = prompt_format.format(test_case['corrupted'])
        
        try:
            # Tokenize
            inputs = tokenizer(input_text, return_tensors='pt').to(device)
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=2,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            
            # Decode
            predicted = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            print(f"   Generated: '{predicted}'")
            print(f"   Time: {inference_time:.2f}s")
            
            # Evaluate accuracy
            expected_normalized = ' '.join(test_case['clean'].lower().replace('.', '').replace(',', '').split())
            predicted_normalized = ' '.join(predicted.lower().replace('.', '').replace(',', '').split())
            
            is_correct = expected_normalized == predicted_normalized
            
            # Check for partial matches (word overlap)
            expected_words = set(test_case['clean'].lower().split())
            predicted_words = set(predicted.lower().split())
            word_overlap = len(expected_words.intersection(predicted_words))
            is_partial = word_overlap > 0
            
            if is_correct:
                correct_predictions += 1
                print("   ✅ CORRECT")
            elif is_partial:
                print(f"   🟡 PARTIAL ({word_overlap} words match)")
            else:
                print("   ❌ INCORRECT")
            
            total_predictions += 1
            
            # Store result
            results.append({
                'test_case': i + 1,
                'input': test_case['corrupted'],
                'expected': test_case['clean'],
                'predicted': predicted,
                'has_punctuation': has_punct,
                'difficulty': difficulty,
                'source': test_case.get('source', 'dataset'),
                'inference_time': inference_time,
                'word_overlap': word_overlap,
                'is_correct': is_correct,
                'is_partial': is_partial
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            total_predictions += 1
            results.append({
                'test_case': i + 1,
                'input': test_case['corrupted'],
                'expected': test_case['clean'],
                'predicted': f'ERROR: {e}',
                'has_punctuation': has_punct,
                'difficulty': difficulty,
                'is_correct': False,
                'is_partial': False
            })
        
        print()
    
    # Calculate metrics
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    partial_matches = sum(1 for r in results if r['is_partial'] and not r['is_correct'])
    avg_inference_time = sum(r['inference_time'] for r in results if 'inference_time' in r) / len([r for r in results if 'inference_time' in r])
    
    print("📊 BALANCED BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total test cases: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Partial matches: {partial_matches}")
    print(f"Complete failures: {total_predictions - correct_predictions - partial_matches}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average inference time: {avg_inference_time:.2f}s")
    print()
    
    # Breakdown by punctuation
    with_punct_results = [r for r in results if r['has_punctuation']]
    without_punct_results = [r for r in results if not r['has_punctuation']]
    
    if with_punct_results:
        with_punct_correct = sum(1 for r in with_punct_results if r['is_correct'])
        with_punct_accuracy = (with_punct_correct / len(with_punct_results) * 100)
        print(f"📊 With punctuation: {with_punct_correct}/{len(with_punct_results)} ({with_punct_accuracy:.1f}%)")
    
    if without_punct_results:
        without_punct_correct = sum(1 for r in without_punct_results if r['is_correct'])
        without_punct_accuracy = (without_punct_correct / len(without_punct_results) * 100)
        print(f"📊 Without punctuation: {without_punct_correct}/{len(without_punct_results)} ({without_punct_accuracy:.1f}%)")
    
    print()
    
    return {
        'model_name': 'google/t5-efficient-tiny',
        'evaluation_type': 'balanced_punctuation_baseline',
        'evaluation_date': datetime.now().isoformat(),
        'device': device,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'test_cases': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy_percent': accuracy,
        'partial_matches': partial_matches,
        'average_inference_time': avg_inference_time,
        'with_punctuation_samples': len(with_punct_results) if with_punct_results else 0,
        'without_punctuation_samples': len(without_punct_results) if without_punct_results else 0,
        'detailed_results': results
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    baseline_results = evaluate_balanced_baseline()
    
    # Save results
    with open('balanced_baseline_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    print(f"💾 Results saved to balanced_baseline_results.json")
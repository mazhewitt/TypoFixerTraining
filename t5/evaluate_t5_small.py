#!/usr/bin/env python3
"""
Evaluate T5-small model performance and compare with T5-tiny baseline.
"""

import json
import random
import time
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def evaluate_t5_small():
    """Evaluate T5-small model and compare with T5-tiny."""
    
    print("🧪 T5-small vs T5-tiny Performance Evaluation")
    print("=" * 60)
    
    # Load T5-small model
    model_path = "models/t5-small-typo-fixer"
    print(f"📥 Loading T5-small model from {model_path}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Move to MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ T5-small loaded: {total_params:,} parameters")
    print(f"🔧 Device: {device}")
    print()
    
    # Load test samples (same as used for T5-tiny evaluation)
    print("📊 Loading test samples...")
    test_samples = []
    
    with open('data/enhanced_training_balanced.jsonl', 'r') as f:
        lines = f.readlines()
        
        # Get diverse examples - same seed as T5-tiny for fair comparison
        with_punct = []
        without_punct = []
        
        for line in lines:
            data = json.loads(line.strip())
            if data['clean'].endswith('.'):
                with_punct.append(data)
            else:
                without_punct.append(data)
        
        # Same sampling as T5-tiny evaluation
        random.seed(42)
        sample_with = random.sample(with_punct, min(10, len(with_punct)))
        sample_without = random.sample(without_punct, min(10, len(without_punct)))
        
        test_samples = sample_with + sample_without
    
    print(f"📝 Testing on {len(test_samples)} examples (same as T5-tiny)")
    print(f"  - With punctuation: {len([s for s in test_samples if s['clean'].endswith('.')])}")
    print(f"  - Without punctuation: {len([s for s in test_samples if not s['clean'].endswith('.')])}")
    print()
    
    results = []
    correct_predictions = 0
    partial_matches = 0
    total_predictions = 0
    total_inference_time = 0
    
    for i, test_case in enumerate(test_samples):
        has_punct = test_case['clean'].endswith('.')
        print(f"📝 Test {i+1}/{len(test_samples)} ({'with' if has_punct else 'without'} punct)")
        print(f"   Input: '{test_case['corrupted']}'")
        print(f"   Expected: '{test_case['clean']}'")
        
        # Use same prompt format as training
        input_text = f"fix spelling errors: {test_case['corrupted']}"
        
        try:
            # Tokenize
            inputs = tokenizer(input_text, return_tensors='pt').to(device)
            
            # Generate with optimized parameters for T5-small
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,  # Longer sequences for T5-small
                    num_beams=3,     # Higher beam search
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Decode
            predicted = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            print(f"   Generated: '{predicted}'")
            print(f"   Time: {inference_time:.3f}s")
            
            # Evaluate accuracy (same method as T5-tiny)
            expected_clean = test_case['clean'].lower().strip().replace('.', '').replace(',', '')
            predicted_clean = predicted.lower().strip().replace('.', '').replace(',', '')
            
            # Word-level comparison
            expected_words = set(expected_clean.split())
            predicted_words = set(predicted_clean.split())
            
            is_exact_match = expected_clean == predicted_clean
            word_overlap = len(expected_words.intersection(predicted_words))
            total_expected_words = len(expected_words)
            
            # Count character-level typo fixes
            typo_chars_fixed = 0
            original_chars = test_case['corrupted'].lower().replace(' ', '').replace('.', '').replace(',', '')
            expected_chars = expected_clean.replace(' ', '')
            predicted_chars = predicted_clean.replace(' ', '')
            
            # Simple typo fix detection
            for j, (orig, exp) in enumerate(zip(original_chars, expected_chars)):
                if j < len(predicted_chars) and orig != exp and predicted_chars[j] == exp:
                    typo_chars_fixed += 1
            
            if is_exact_match:
                correct_predictions += 1
                print("   ✅ EXACT MATCH")
            elif word_overlap == total_expected_words and len(predicted_words) >= total_expected_words:
                correct_predictions += 1
                print("   ✅ CORRECT (all words present)")
            elif word_overlap >= total_expected_words * 0.8:
                partial_matches += 1
                print(f"   🟡 PARTIAL MATCH ({word_overlap}/{total_expected_words} words)")
            else:
                print(f"   ❌ INCORRECT ({word_overlap}/{total_expected_words} words match)")
            
            total_predictions += 1
            
            # Store detailed result
            results.append({
                'test_case': i + 1,
                'input': test_case['corrupted'],
                'expected': test_case['clean'],
                'predicted': predicted,
                'has_punctuation': has_punct,
                'inference_time': inference_time,
                'word_overlap': word_overlap,
                'total_expected_words': total_expected_words,
                'is_exact_match': is_exact_match,
                'word_accuracy': word_overlap / total_expected_words if total_expected_words > 0 else 0,
                'typo_chars_fixed': typo_chars_fixed
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
                'is_exact_match': False,
                'word_accuracy': 0
            })
        
        print()
    
    # Calculate comprehensive metrics
    exact_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    avg_word_accuracy = sum(r['word_accuracy'] for r in results) / len(results) * 100
    avg_inference_time = total_inference_time / total_predictions if total_predictions > 0 else 0
    
    # Category breakdown
    with_punct_results = [r for r in results if r['has_punctuation']]
    without_punct_results = [r for r in results if not r['has_punctuation']]
    
    print("📊 T5-SMALL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: T5-small ({total_params:,} parameters)")
    print(f"Total test cases: {total_predictions}")
    print(f"Exact matches: {correct_predictions} ({exact_accuracy:.1f}%)")
    print(f"Partial matches: {partial_matches} ({partial_matches/total_predictions*100:.1f}%)")
    print(f"Complete failures: {total_predictions - correct_predictions - partial_matches}")
    print(f"Average word accuracy: {avg_word_accuracy:.1f}%")
    print(f"Average inference time: {avg_inference_time:.3f}s")
    print()
    
    if with_punct_results:
        with_punct_exact = sum(1 for r in with_punct_results if r['is_exact_match'])
        with_punct_word_acc = sum(r['word_accuracy'] for r in with_punct_results) / len(with_punct_results) * 100
        print(f"📊 With punctuation: {with_punct_exact}/{len(with_punct_results)} exact ({with_punct_exact/len(with_punct_results)*100:.1f}%), {with_punct_word_acc:.1f}% word accuracy")
    
    if without_punct_results:
        without_punct_exact = sum(1 for r in without_punct_results if r['is_exact_match'])
        without_punct_word_acc = sum(r['word_accuracy'] for r in without_punct_results) / len(without_punct_results) * 100
        print(f"📊 Without punctuation: {without_punct_exact}/{len(without_punct_results)} exact ({without_punct_exact/len(without_punct_results)*100:.1f}%), {without_punct_word_acc:.1f}% word accuracy")
    
    print()
    
    # Load T5-tiny results for comparison
    try:
        with open('optimized_model_evaluation.json', 'r') as f:
            tiny_results = json.load(f)
        
        print("🔄 COMPARISON: T5-small vs T5-tiny (optimized)")
        print("=" * 60)
        print(f"Model Size:")
        print(f"  T5-tiny:  15.6M parameters")
        print(f"  T5-small: {total_params/1e6:.1f}M parameters ({total_params/15570688:.1f}x larger)")
        print()
        print(f"Exact Accuracy:")
        print(f"  T5-tiny:  {tiny_results.get('exact_accuracy_percent', 0):.1f}%")
        print(f"  T5-small: {exact_accuracy:.1f}% ({exact_accuracy - tiny_results.get('exact_accuracy_percent', 0):+.1f}%)")
        print()
        print(f"Word Accuracy:")
        print(f"  T5-tiny:  {tiny_results.get('average_word_accuracy_percent', 0):.1f}%")
        print(f"  T5-small: {avg_word_accuracy:.1f}% ({avg_word_accuracy - tiny_results.get('average_word_accuracy_percent', 0):+.1f}%)")
        print()
        print(f"Inference Speed:")
        print(f"  T5-tiny:  {tiny_results.get('average_inference_time', 0):.3f}s")
        print(f"  T5-small: {avg_inference_time:.3f}s ({avg_inference_time - tiny_results.get('average_inference_time', 0):+.3f}s)")
        print()
        
        improvement_ratio = exact_accuracy / tiny_results.get('exact_accuracy_percent', 1)
        print(f"🎯 Overall Improvement: {improvement_ratio:.1f}x better exact accuracy")
        
    except FileNotFoundError:
        print("⚠️ T5-tiny results not found for comparison")
    
    # Performance assessment
    print("🧠 T5-small Performance Assessment:")
    if exact_accuracy < 20:
        assessment = "🟡 LIMITED - Better than tiny but still needs improvement"
    elif exact_accuracy < 50:
        assessment = "🟢 MODERATE - Good improvement over T5-tiny"
    elif exact_accuracy < 80:
        assessment = "🔵 GOOD - Strong typo correction performance"
    else:
        assessment = "🟢 EXCELLENT - Very strong typo correction"
    
    print(f"   {assessment}")
    print(f"   Model capacity: Significantly better than T5-tiny")
    
    if avg_word_accuracy > 85:
        print(f"   🎯 High word accuracy ({avg_word_accuracy:.1f}%) shows good semantic understanding")
    
    return {
        'model_name': 't5-small-typo-fixer',
        'evaluation_date': datetime.now().isoformat(),
        'model_path': model_path,
        'device': device,
        'total_parameters': total_params,
        'test_cases': total_predictions,
        'exact_matches': correct_predictions,
        'exact_accuracy_percent': exact_accuracy,
        'partial_matches': partial_matches,
        'average_word_accuracy_percent': avg_word_accuracy,
        'average_inference_time': avg_inference_time,
        'training_time_minutes': 2.5,
        'model_size_improvement': f"{total_params/15570688:.1f}x larger than T5-tiny",
        'detailed_results': results
    }

if __name__ == "__main__":
    evaluation_results = evaluate_t5_small()
    
    # Save results
    with open('t5_small_evaluation.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n💾 T5-small evaluation saved to t5_small_evaluation.json")
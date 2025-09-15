#!/usr/bin/env python3
"""
Baseline evaluation for T5-efficient-tiny before fine-tuning on typo correction.
Tests the model's ability to correct typos out-of-the-box.
"""

import json
import random
import time
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def evaluate_baseline_t5():
    """Evaluate T5-efficient-tiny baseline performance on typo correction."""
    
    print("🧪 T5-efficient-tiny Baseline Evaluation")
    print("=" * 50)
    
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
    
    # Test prompts - various difficulty levels
    test_cases = [
        # Simple single word errors
        {
            "corrupted": "I beleive this is correct.",
            "expected": "I believe this is correct.",
            "difficulty": "easy"
        },
        {
            "corrupted": "The qucik brown fox.",
            "expected": "The quick brown fox.",
            "difficulty": "easy"
        },
        # Multiple errors
        {
            "corrupted": "I beleive this is teh correct answr.",
            "expected": "I believe this is the correct answer.",
            "difficulty": "medium"
        },
        {
            "corrupted": "She recieved her degre last year.",
            "expected": "She received her degree last year.",
            "difficulty": "medium"
        },
        # Complex/multiple word errors
        {
            "corrupted": "The goverment anounced new polices yesterday.",
            "expected": "The government announced new policies yesterday.",
            "difficulty": "hard"
        },
        {
            "corrupted": "Unfortunatly, the experiance was dissapointing.",
            "expected": "Unfortunately, the experience was disappointing.",
            "difficulty": "hard"
        },
        # Context-dependent corrections
        {
            "corrupted": "Their going to there house over they're.",
            "expected": "They're going to their house over there.",
            "difficulty": "hard"
        },
        {
            "corrupted": "The studnets where asked to wright an esay.",
            "expected": "The students were asked to write an essay.",
            "difficulty": "hard"
        }
    ]
    
    # Sample some examples from training data
    print("📊 Loading sample training data...")
    training_samples = []
    try:
        with open('data/enhanced_training_full.jsonl', 'r') as f:
            lines = f.readlines()
            sample_lines = random.sample(lines, min(10, len(lines)))
            for line in sample_lines:
                data = json.loads(line.strip())
                training_samples.append({
                    "corrupted": data['corrupted'],
                    "expected": data['clean'],
                    "difficulty": data.get('complexity', 'unknown'),
                    "source": data.get('source', 'dataset')
                })
    except FileNotFoundError:
        print("⚠️ Training data not found, using manual test cases only")
    
    # Combine test cases
    all_test_cases = test_cases + training_samples[:8]  # Limit training samples
    
    print(f"🎯 Running baseline evaluation on {len(all_test_cases)} test cases...")
    print()
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    # Different prompt formats to try
    prompt_formats = [
        "correct typos: {}",
        "fix spelling: {}",
        "correct spelling errors: {}",
        "fix typos: {}"
    ]
    
    for i, test_case in enumerate(all_test_cases):
        print(f"📝 Test Case {i+1}/{len(all_test_cases)} ({test_case.get('difficulty', 'unknown')} difficulty)")
        print(f"   Input: '{test_case['corrupted']}'")
        print(f"   Expected: '{test_case['expected']}'")
        
        best_result = None
        best_score = -1
        
        # Try different prompt formats
        for prompt_format in prompt_formats:
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
                        num_beams=2,  # Light beam search
                        do_sample=False,
                        early_stopping=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                inference_time = time.time() - start_time
                
                # Decode
                predicted = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                # Simple scoring - check if any words match
                expected_words = set(test_case['expected'].lower().split())
                predicted_words = set(predicted.lower().split())
                word_overlap = len(expected_words.intersection(predicted_words))
                
                if word_overlap > best_score:
                    best_score = word_overlap
                    best_result = {
                        'prompt_format': prompt_format,
                        'predicted': predicted,
                        'inference_time': inference_time,
                        'word_overlap': word_overlap
                    }
                    
            except Exception as e:
                print(f"   ❌ Error with prompt '{prompt_format}': {e}")
                continue
        
        if best_result:
            predicted = best_result['predicted']
            print(f"   Generated: '{predicted}' (format: {best_result['prompt_format']})")
            print(f"   Time: {best_result['inference_time']:.2f}s")
            
            # Evaluate accuracy
            expected_normalized = ' '.join(test_case['expected'].lower().replace('.', '').replace(',', '').split())
            predicted_normalized = ' '.join(predicted.lower().replace('.', '').replace(',', '').split())
            
            is_correct = expected_normalized == predicted_normalized
            is_partial = best_result['word_overlap'] > 0
            
            if is_correct:
                correct_predictions += 1
                print("   ✅ CORRECT")
            elif is_partial:
                print(f"   🟡 PARTIAL ({best_result['word_overlap']} words match)")
            else:
                print("   ❌ INCORRECT")
            
            total_predictions += 1
            
            # Store result
            results.append({
                'test_case': i + 1,
                'input': test_case['corrupted'],
                'expected': test_case['expected'],
                'predicted': predicted,
                'difficulty': test_case.get('difficulty', 'unknown'),
                'source': test_case.get('source', 'manual'),
                'prompt_format': best_result['prompt_format'],
                'inference_time': best_result['inference_time'],
                'word_overlap': best_result['word_overlap'],
                'is_correct': is_correct,
                'is_partial': is_partial
            })
        else:
            print("   ❌ ALL FORMATS FAILED")
            total_predictions += 1
            results.append({
                'test_case': i + 1,
                'input': test_case['corrupted'],
                'expected': test_case['expected'],
                'predicted': 'ERROR - No output generated',
                'difficulty': test_case.get('difficulty', 'unknown'),
                'source': test_case.get('source', 'manual'),
                'is_correct': False,
                'is_partial': False
            })
        
        print()
    
    # Calculate final metrics
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    partial_matches = sum(1 for r in results if r['is_partial'] and not r['is_correct'])
    avg_inference_time = sum(r['inference_time'] for r in results if 'inference_time' in r) / len([r for r in results if 'inference_time' in r])
    
    print("📊 BASELINE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total test cases: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Partial matches: {partial_matches}")
    print(f"Complete failures: {total_predictions - correct_predictions - partial_matches}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average inference time: {avg_inference_time:.2f}s")
    print()
    
    # Breakdown by difficulty
    if any(r.get('difficulty') for r in results):
        print("📊 Accuracy by Difficulty:")
        difficulties = set(r['difficulty'] for r in results if r.get('difficulty'))
        for difficulty in sorted(difficulties):
            diff_results = [r for r in results if r.get('difficulty') == difficulty]
            diff_correct = sum(1 for r in diff_results if r['is_correct'])
            diff_accuracy = (diff_correct / len(diff_results) * 100) if diff_results else 0
            print(f"  {difficulty}: {diff_correct}/{len(diff_results)} ({diff_accuracy:.1f}%)")
        print()
    
    # Best performing prompt format
    if results:
        format_counts = {}
        for r in results:
            fmt = r.get('prompt_format', 'unknown')
            if fmt not in format_counts:
                format_counts[fmt] = {'total': 0, 'correct': 0}
            format_counts[fmt]['total'] += 1
            if r['is_correct']:
                format_counts[fmt]['correct'] += 1
        
        print("📊 Performance by Prompt Format:")
        for fmt, counts in sorted(format_counts.items()):
            accuracy_pct = (counts['correct'] / counts['total'] * 100) if counts['total'] > 0 else 0
            print(f"  '{fmt}': {counts['correct']}/{counts['total']} ({accuracy_pct:.1f}%)")
        print()
    
    return {
        'model_name': 'google/t5-efficient-tiny',
        'evaluation_date': datetime.now().isoformat(),
        'device': device,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'test_cases': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy_percent': accuracy,
        'partial_matches': partial_matches,
        'average_inference_time': avg_inference_time,
        'detailed_results': results,
        'prompt_formats_tested': prompt_formats
    }

if __name__ == "__main__":
    baseline_results = evaluate_baseline_t5()
    
    # Save results to JSON for later comparison
    with open('baseline_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    print(f"💾 Results saved to baseline_results.json")
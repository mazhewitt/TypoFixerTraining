#!/usr/bin/env python3
"""
Measure sentence-level accuracy for the Qwen typo correction model.
Tests how often the model produces exactly the correct sentence.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
from typing import List, Dict, Tuple
import time

def conservative_inference(model, tokenizer, prompt: str, temperature: float = 0.1) -> str:
    """Ultra-conservative inference for typo correction with temperature control."""
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Use very low temperature and sampling for minimal creativity
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,  # Slightly increased for complete sentences
            do_sample=True,     # Enable sampling to use temperature
            temperature=temperature,  # Very low temperature
            top_p=0.1,         # Very restrictive nucleus sampling
            top_k=3,           # Only consider top 3 tokens
            num_beams=1,       # No beam search for speed
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.02,  # Minimal repetition penalty
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Clean up output
    generated_text = ' '.join(generated_text.split())
    if '.' in generated_text:
        corrected = generated_text.split('.')[0].strip() + '.'
    else:
        corrected = generated_text.strip()
    
    corrected = corrected.replace('##', '').replace('#', '').strip()
    
    # Remove unwanted prefixes
    unwanted_prefixes = [
        'Here is', 'The corrected', 'Correction:', 'Fixed:', 'Answer:', 
        'The answer is', 'Result:', 'Output:', 'Corrected:'
    ]
    for prefix in unwanted_prefixes:
        if corrected.lower().startswith(prefix.lower()):
            corrected = corrected[len(prefix):].strip()
    
    return corrected

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove extra spaces and normalize punctuation
    normalized = ' '.join(text.strip().lower().split())
    # Remove trailing period for comparison
    if normalized.endswith('.'):
        normalized = normalized[:-1]
    return normalized

def create_test_dataset() -> List[Dict[str, str]]:
    """Create a comprehensive test dataset for sentence accuracy measurement."""
    test_cases = [
        # Basic spelling errors
        {
            "corrupted": "I beleive this is teh correct answr.",
            "clean": "I believe this is the correct answer."
        },
        {
            "corrupted": "She recieved her degre last year.",
            "clean": "She received her degree last year."
        },
        {
            "corrupted": "The resturant serves excelent food.",
            "clean": "The restaurant serves excellent food."
        },
        {
            "corrupted": "He is studyng for his final examintion.",
            "clean": "He is studying for his final examination."
        },
        {
            "corrupted": "We dicussed the importnt details yesterday.",
            "clean": "We discussed the important details yesterday."
        },
        
        # Complex typos
        {
            "corrupted": "The begining of the story was very excting.",
            "clean": "The beginning of the story was very exciting."
        },
        {
            "corrupted": "I definately need to imporve my skils.",
            "clean": "I definitely need to improve my skills."
        },
        {
            "corrupted": "The experiance was chalenging and rewardng.",
            "clean": "The experience was challenging and rewarding."
        },
        {
            "corrupted": "Please chekc your emial for more informaton.",
            "clean": "Please check your email for more information."
        },
        {
            "corrupted": "This documetn contians several importatn points.",
            "clean": "This document contains several important points."
        },
        
        # Keyboard errors
        {
            "corrupted": "The qucik brown fox jumoed over the fence.",
            "clean": "The quick brown fox jumped over the fence."
        },
        {
            "corrupted": "Teh meetign has been rescheduled for tommorow.",
            "clean": "The meeting has been rescheduled for tomorrow."
        },
        {
            "corrupted": "Thank you for your consideratin of my applicaton.",
            "clean": "Thank you for your consideration of my application."
        },
        
        # Multiple errors per sentence
        {
            "corrupted": "I am goign to teh store to buy som grocerys.",
            "clean": "I am going to the store to buy some groceries."
        },
        {
            "corrupted": "The waether forcast predicts rain tommorow mornign.",
            "clean": "The weather forecast predicts rain tomorrow morning."
        },
        
        # Common confusions
        {
            "corrupted": "I want to loose some weight this year.",
            "clean": "I want to lose some weight this year."
        },
        {
            "corrupted": "There going to there house tomorrow.",
            "clean": "They're going to their house tomorrow."
        },
        {
            "corrupted": "Its a beautiful day, isnt it?",
            "clean": "It's a beautiful day, isn't it?"
        },
        
        # Professional/academic contexts
        {
            "corrupted": "The research metodology was carefuly designed.",
            "clean": "The research methodology was carefully designed."
        },
        {
            "corrupted": "Students must submitt their assigments on time.",
            "clean": "Students must submit their assignments on time."
        },
        
        # Edge cases
        {
            "corrupted": "Teh.",
            "clean": "The."
        },
        {
            "corrupted": "This sentence has no errors.",
            "clean": "This sentence has no errors."
        },
        
        # Real-world examples
        {
            "corrupted": "Please find the attachd document for your reveiw.",
            "clean": "Please find the attached document for your review."
        },
        {
            "corrupted": "The technolgy has advaced significantely.",
            "clean": "The technology has advanced significantly."
        },
        {
            "corrupted": "Communication skils are esential for sucess.",
            "clean": "Communication skills are essential for success."
        },
        {
            "corrupted": "Enviromental protecton requires imediate atention.",
            "clean": "Environmental protection requires immediate attention."
        }
    ]
    
    return test_cases

def load_validation_dataset(file_path: str, sample_size: int = 100) -> List[Dict[str, str]]:
    """Load validation examples from the training dataset."""
    examples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_examples = [json.loads(line.strip()) for line in f]
        
        # Random sample for validation
        sampled = random.sample(all_examples, min(sample_size, len(all_examples)))
        
        for example in sampled:
            examples.append({
                "corrupted": example["corrupted"],
                "clean": example["clean"]
            })
    
    except FileNotFoundError:
        print(f"⚠️ Validation file {file_path} not found, using only manual test cases")
    
    return examples

def measure_accuracy(model, tokenizer, test_cases: List[Dict[str, str]], dataset_name: str) -> Dict:
    """Measure sentence accuracy on test cases."""
    print(f"\n📊 Testing {dataset_name} ({len(test_cases)} examples)")
    print("-" * 60)
    
    correct = 0
    total = 0
    results = []
    
    start_time = time.time()
    
    for i, case in enumerate(test_cases):
        prompt = f"Fix: {case['corrupted']}"
        expected = case['clean']
        
        # Get model prediction
        predicted = conservative_inference(model, tokenizer, prompt)
        
        # Normalize for comparison
        predicted_norm = normalize_text(predicted)
        expected_norm = normalize_text(expected)
        
        is_correct = predicted_norm == expected_norm
        if is_correct:
            correct += 1
        
        total += 1
        
        # Store result
        result = {
            "input": case['corrupted'],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "predicted_norm": predicted_norm,
            "expected_norm": expected_norm
        }
        results.append(result)
        
        # Show first few examples and errors
        if i < 5 or not is_correct and len([r for r in results if not r['correct']]) <= 10:
            status = "✅" if is_correct else "❌"
            print(f"{i+1:2d}. {status} Input:     {case['corrupted']}")
            print(f"     Expected:  {expected}")
            print(f"     Predicted: {predicted}")
            if not is_correct:
                print(f"     Norm Exp:  '{expected_norm}'")
                print(f"     Norm Pred: '{predicted_norm}'")
            print()
    
    elapsed_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    
    # Summary
    print(f"📈 {dataset_name} Results:")
    print(f"   Correct: {correct}/{total}")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Time: {elapsed_time:.1f}s ({elapsed_time/total:.3f}s per example)")
    
    return {
        "dataset": dataset_name,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "time": elapsed_time,
        "avg_time": elapsed_time / total,
        "results": results
    }

def main():
    print("🎯 Sentence Accuracy Measurement for Qwen Typo Correction")
    print("=" * 70)
    
    # Load model
    model_path = "mazhewitt/qwen-typo-fixer"
    print(f"🤖 Loading model: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test datasets
    all_results = {}
    
    # 1. Manual curated test cases
    manual_cases = create_test_dataset()
    manual_results = measure_accuracy(model, tokenizer, manual_cases, "Manual Test Cases")
    all_results["manual"] = manual_results
    
    # 2. Validation from training data
    validation_cases = load_validation_dataset("data/enhanced_training_full.jsonl", sample_size=50)
    if validation_cases:
        validation_results = measure_accuracy(model, tokenizer, validation_cases, "Training Data Sample")
        all_results["validation"] = validation_results
    
    # Overall summary
    print(f"\n🏆 OVERALL SENTENCE ACCURACY RESULTS")
    print("=" * 70)
    
    total_correct = 0
    total_examples = 0
    
    for name, results in all_results.items():
        accuracy = results["accuracy"]
        print(f"{results['dataset']:20s}: {results['correct']:3d}/{results['total']:3d} = {accuracy:6.1%}")
        total_correct += results["correct"]
        total_examples += results["total"]
    
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
    print("-" * 70)
    print(f"{'COMBINED ACCURACY':20s}: {total_correct:3d}/{total_examples:3d} = {overall_accuracy:6.1%}")
    
    # Target assessment
    target_accuracy = 0.90
    print(f"\n🎯 Target Assessment:")
    print(f"   Target: {target_accuracy:.1%}")
    print(f"   Achieved: {overall_accuracy:.1%}")
    
    if overall_accuracy >= target_accuracy:
        print(f"   Status: ✅ TARGET ACHIEVED!")
    elif overall_accuracy >= target_accuracy - 0.05:
        print(f"   Status: 🟡 Close to target ({(target_accuracy - overall_accuracy)*100:.1f}% gap)")
    else:
        print(f"   Status: ❌ Below target ({(target_accuracy - overall_accuracy)*100:.1f}% gap)")
    
    # Error analysis
    error_examples = []
    for results in all_results.values():
        error_examples.extend([r for r in results["results"] if not r["correct"]])
    
    if error_examples:
        print(f"\n🔍 Error Analysis ({len(error_examples)} errors):")
        print("Top error patterns:")
        for i, error in enumerate(error_examples[:5], 1):
            print(f"{i}. '{error['input']}' → '{error['predicted']}' (expected: '{error['expected']}')")

if __name__ == "__main__":
    main()
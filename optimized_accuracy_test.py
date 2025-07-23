#!/usr/bin/env python3
"""
Test the optimized approach (few-shot + best parameters) on full accuracy measurement.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
from typing import List, Dict
import time

def optimized_inference(model, tokenizer, corrupted_sentence: str) -> str:
    """Optimized inference using few-shot prompting with best parameters."""
    
    # Few-shot prompt with examples
    full_prompt = f"""Fix typos in these sentences:

Input: I beleive this is teh answer.
Output: I believe this is the answer.

Input: She recieved her degre yesterday.
Output: She received her degree yesterday.

Input: The resturant serves good food.
Output: The restaurant serves good food.

Input: {corrupted_sentence}
Output:"""
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with optimal parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,  # Greedy decoding works best
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.02,
        )
    
    # Decode
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Clean up output
    generated_text = ' '.join(generated_text.split())
    
    # Extract just the corrected sentence
    if '\n' in generated_text:
        generated_text = generated_text.split('\n')[0].strip()
    
    # Remove common suffixes that appear
    for suffix in ['Input:', 'Output:', 'Human:', 'Assistant:']:
        if suffix.lower() in generated_text.lower():
            generated_text = generated_text.split(suffix)[0].strip()
    
    if '.' in generated_text:
        corrected = generated_text.split('.')[0].strip() + '.'
    else:
        corrected = generated_text.strip()
    
    # Remove unwanted artifacts
    corrected = corrected.replace('##', '').replace('#', '').strip()
    
    return corrected

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    normalized = ' '.join(text.strip().lower().split())
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
        
        # Common confusions (challenging)
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

def measure_optimized_accuracy():
    print("🎯 Optimized Sentence Accuracy Test (Few-Shot + Best Parameters)")
    print("=" * 75)
    
    # Load model
    print("🤖 Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained("mazhewitt/qwen-typo-fixer", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("mazhewitt/qwen-typo-fixer", trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!\n")
    
    # Get test cases
    test_cases = create_test_dataset()
    
    print(f"📊 Testing on {len(test_cases)} examples")
    print("-" * 60)
    
    correct = 0
    total = 0
    results = []
    
    start_time = time.time()
    
    for i, case in enumerate(test_cases):
        corrupted = case['corrupted']
        expected = case['clean']
        
        # Get model prediction using optimized approach
        predicted = optimized_inference(model, tokenizer, corrupted)
        
        # Normalize for comparison
        predicted_norm = normalize_text(predicted)
        expected_norm = normalize_text(expected)
        
        is_correct = predicted_norm == expected_norm
        if is_correct:
            correct += 1
        
        total += 1
        
        # Store result
        result = {
            "input": corrupted,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "predicted_norm": predicted_norm,
            "expected_norm": expected_norm
        }
        results.append(result)
        
        # Show first few examples and errors
        if i < 5 or (not is_correct and len([r for r in results if not r['correct']]) <= 5):
            status = "✅" if is_correct else "❌"
            print(f"{i+1:2d}. {status} Input:     {corrupted}")
            print(f"     Expected:  {expected}")
            print(f"     Predicted: {predicted}")
            if not is_correct:
                print(f"     Norm Exp:  '{expected_norm}'")
                print(f"     Norm Pred: '{predicted_norm}'")
            print()
    
    elapsed_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    
    # Summary
    print(f"🏆 OPTIMIZED APPROACH RESULTS")
    print("=" * 75)
    print(f"   Correct: {correct}/{total}")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Time: {elapsed_time:.1f}s ({elapsed_time/total:.3f}s per example)")
    
    # Target assessment
    target_accuracy = 0.90
    print(f"\n🎯 Target Assessment:")
    print(f"   Target: {target_accuracy:.1%}")
    print(f"   Achieved: {accuracy:.1%}")
    
    if accuracy >= target_accuracy:
        print(f"   Status: ✅ TARGET ACHIEVED!")
    elif accuracy >= target_accuracy - 0.05:
        print(f"   Status: 🟡 Close to target ({(target_accuracy - accuracy)*100:.1f}% gap)")
    else:
        print(f"   Status: ❌ Below target ({(target_accuracy - accuracy)*100:.1f}% gap)")
    
    # Error analysis
    error_examples = [r for r in results if not r['correct']]
    if error_examples:
        print(f"\n🔍 Error Analysis ({len(error_examples)} errors):")
        print("Top error patterns:")
        for i, error in enumerate(error_examples[:3], 1):
            print(f"{i}. '{error['input']}' → '{error['predicted']}' (expected: '{error['expected']}')")
    
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "time": elapsed_time,
        "results": results
    }

if __name__ == "__main__":
    measure_optimized_accuracy()
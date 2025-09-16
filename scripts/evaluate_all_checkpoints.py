#!/usr/bin/env python3
"""
Evaluate all checkpoints to find the sweet spot for ByT5 training
"""

import argparse
import os
import json
import re
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def extract_step_number(checkpoint_path):
    """Extract step number from checkpoint path"""
    match = re.search(r'checkpoint-(\d+)', str(checkpoint_path))
    return int(match.group(1)) if match else 0

def quick_evaluate(model_path, tokenizer, prefix="fix typos:"):
    """Quick evaluation with key test cases"""
    
    test_cases = [
        ("I beleive this is teh correct answr.", "I believe this is the correct answer."),
        ("The qick brown fox jumps ovr the lazy dog.", "The quick brown fox jumps over the lazy dog."),
        ("Please chck your email for futher instructions.", "Please check your email for further instructions."),
        ("We need to discus this matter urgently.", "We need to discuss this matter urgently."),
        ("This is alredy completd.", "This is already completed."),
    ]
    
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        correct_words = 0
        total_words = 0
        perfect_sentences = 0
        
        results = []
        
        for original, expected in test_cases:
            input_text = f"{prefix} {original}"
            inputs = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True)
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128, num_beams=2, early_stopping=True)
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append((original, generated, expected))
            
            # Calculate accuracy
            orig_words = original.lower().split()
            gen_words = generated.lower().split()
            exp_words = expected.lower().split()
            
            # Count corrections
            for i, (orig_word, exp_word) in enumerate(zip(orig_words, exp_words)):
                if orig_word != exp_word:  # This word needs correction
                    total_words += 1
                    if i < len(gen_words) and gen_words[i] == exp_word:
                        correct_words += 1
            
            # Perfect sentence check
            if generated.lower().strip() == expected.lower().strip():
                perfect_sentences += 1
        
        word_accuracy = correct_words / max(1, total_words)
        sentence_accuracy = perfect_sentences / len(test_cases)
        
        return {
            "word_accuracy": word_accuracy,
            "sentence_accuracy": sentence_accuracy,
            "correct_words": correct_words,
            "total_words": total_words,
            "perfect_sentences": perfect_sentences,
            "total_sentences": len(test_cases),
            "sample_results": results[:3]  # Keep first 3 examples
        }
        
    except Exception as e:
        print(f"❌ Error evaluating {model_path}: {e}")
        return None

def evaluate_all_checkpoints(base_path, prefix="fix typos:"):
    """Evaluate all checkpoints in a directory"""
    
    base_path = Path(base_path)
    
    # Find all checkpoints
    checkpoints = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint-'):
            checkpoints.append(item)
    
    # Sort by step number
    checkpoints.sort(key=extract_step_number)
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {base_path}")
        return
    
    print(f"🔍 Found {len(checkpoints)} checkpoints to evaluate")
    print(f"📁 Base path: {base_path}")
    print()
    
    # Load tokenizer once
    print("🔧 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoints[0], use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(str(base_path), use_fast=False)
    
    results = []
    
    print("🧪 Evaluating checkpoints...")
    print("=" * 100)
    print(f"{'Step':<8} {'Epoch':<8} {'Word Acc':<10} {'Sent Acc':<10} {'Perfect':<8} {'Status':<12} {'Sample Result'}")
    print("=" * 100)
    
    for i, checkpoint in enumerate(checkpoints):
        step = extract_step_number(checkpoint)
        
        # Try to get epoch from trainer_state.json if available
        epoch = 0.0
        trainer_state_file = checkpoint / "trainer_state.json"
        if trainer_state_file.exists():
            try:
                with open(trainer_state_file) as f:
                    state = json.load(f)
                    epoch = state.get('epoch', 0.0)
            except:
                pass
        
        start_time = time.time()
        result = quick_evaluate(str(checkpoint), tokenizer, prefix)
        eval_time = time.time() - start_time
        
        if result:
            word_acc = result['word_accuracy']
            sent_acc = result['sentence_accuracy']
            perfect = f"{result['perfect_sentences']}/{result['total_sentences']}"
            
            # Determine status
            if word_acc >= 0.8 and sent_acc >= 0.4:
                status = "🏆 EXCELLENT"
            elif word_acc >= 0.6 and sent_acc >= 0.2:
                status = "✅ GOOD"
            elif word_acc >= 0.4:
                status = "🟡 FAIR"
            else:
                status = "❌ POOR"
            
            # Show sample result
            sample = result['sample_results'][0] if result['sample_results'] else ("", "", "")
            sample_text = f"'{sample[0][:30]}...' → '{sample[1][:30]}...'"
            
            print(f"{step:<8} {epoch:<8.2f} {word_acc:<10.1%} {sent_acc:<10.1%} {perfect:<8} {status:<12} {sample_text}")
            
            # Store detailed results
            checkpoint_result = {
                "checkpoint": str(checkpoint),
                "step": step,
                "epoch": epoch,
                "evaluation_time": eval_time,
                **result
            }
            results.append(checkpoint_result)
        else:
            print(f"{step:<8} {epoch:<8.2f} {'ERROR':<10} {'ERROR':<10} {'0/5':<8} {'❌ FAILED':<12} {'Evaluation failed'}")
    
    print("=" * 100)
    
    # Find best checkpoint
    if results:
        # Sort by word accuracy, then sentence accuracy
        best_checkpoint = max(results, key=lambda x: (x['word_accuracy'], x['sentence_accuracy']))
        
        print(f"\n🏆 BEST CHECKPOINT:")
        print(f"   Step: {best_checkpoint['step']}")
        print(f"   Epoch: {best_checkpoint['epoch']:.2f}")
        print(f"   Word Accuracy: {best_checkpoint['word_accuracy']:.1%}")
        print(f"   Sentence Accuracy: {best_checkpoint['sentence_accuracy']:.1%}")
        print(f"   Perfect Sentences: {best_checkpoint['perfect_sentences']}/{best_checkpoint['total_sentences']}")
        print(f"   Path: {best_checkpoint['checkpoint']}")
        
        # Show detailed results for best checkpoint
        print(f"\n📝 Best Checkpoint Sample Results:")
        for i, (orig, gen, exp) in enumerate(best_checkpoint['sample_results'], 1):
            print(f"   {i}. '{orig}' → '{gen}'")
            print(f"      Expected: '{exp}'")
        
        # Save detailed results
        output_file = base_path / "checkpoint_evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "best_checkpoint": best_checkpoint,
                "all_results": results,
                "evaluation_summary": {
                    "total_checkpoints": len(results),
                    "best_word_accuracy": best_checkpoint['word_accuracy'],
                    "best_sentence_accuracy": best_checkpoint['sentence_accuracy'],
                    "best_step": best_checkpoint['step'],
                    "best_epoch": best_checkpoint['epoch']
                }
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        if best_checkpoint['word_accuracy'] >= 0.7:
            print(f"   ✅ This checkpoint looks good for production use!")
            print(f"   💡 Consider using step {best_checkpoint['step']} (epoch {best_checkpoint['epoch']:.2f})")
        elif best_checkpoint['word_accuracy'] >= 0.5:
            print(f"   🟡 Decent performance but could be better")
            print(f"   💡 Try training longer or with different hyperparameters")
        else:
            print(f"   ❌ All checkpoints show poor performance")
            print(f"   💡 Consider adjusting training data or model architecture")

def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints to find the sweet spot")
    parser.add_argument("--model-path", required=True, help="Path to model directory with checkpoints")
    parser.add_argument("--prefix", default="fix typos:", help="Model prefix")
    
    args = parser.parse_args()
    
    evaluate_all_checkpoints(args.model_path, args.prefix)

if __name__ == "__main__":
    main()
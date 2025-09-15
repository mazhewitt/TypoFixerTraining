#!/usr/bin/env python3
"""
Comprehensive ByT5 model testing script for CUDA training server
Tests model performance and creates detailed reports for analysis

Usage:
  # Test a local model
  python3 scripts/test_byt5_on_server.py --model-path ./models/byt5-small-typo-fixer-v2
  
  # Test a HuggingFace model  
  python3 scripts/test_byt5_on_server.py --model-path mazhewitt/byt5-small-typo-fixer-v3
  
  # Test with custom prefix
  python3 scripts/test_byt5_on_server.py --model-path ./models/byt5-small-typo-fixer-v2 --prefix "fix typos:"
"""

import argparse
import json
import time
import os
from typing import List, Tuple, Dict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class ByT5TypoTester:
    def __init__(self, model_path: str, prefix: str = "fix typos:"):
        self.model_path = model_path
        self.prefix = prefix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Loading model from: {model_path}")
        print(f"🎯 Using prefix: '{prefix}'")
        print(f"💻 Device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model parameters: ~{sum(p.numel() for p in self.model.parameters()) / 1_000_000:.1f}M")
        print()

    def fix_typos(self, text: str, max_length: int = 512, num_beams: int = 4) -> str:
        """Fix typos in text using the model"""
        input_text = f"{self.prefix} {text}".strip()
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0,
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()

    def calculate_accuracy(self, original: str, corrected: str, expected: str) -> Dict:
        """Calculate detailed accuracy metrics"""
        orig_words = original.lower().split()
        corr_words = corrected.lower().split()
        exp_words = expected.lower().split()
        
        # Word-level analysis
        total_errors = 0
        fixed_errors = 0
        new_errors = 0
        
        max_len = max(len(orig_words), len(corr_words), len(exp_words))
        
        for i in range(max_len):
            orig_word = orig_words[i] if i < len(orig_words) else ""
            corr_word = corr_words[i] if i < len(corr_words) else ""
            exp_word = exp_words[i] if i < len(exp_words) else ""
            
            # Count errors in original
            if orig_word != exp_word:
                total_errors += 1
                # Check if this error was fixed
                if corr_word == exp_word:
                    fixed_errors += 1
            
            # Check for new errors introduced
            if orig_word == exp_word and corr_word != exp_word:
                new_errors += 1
        
        # Calculate metrics
        word_accuracy = fixed_errors / max(1, total_errors)
        sentence_accuracy = 1.0 if corrected.lower().strip() == expected.lower().strip() else 0.0
        
        return {
            "word_accuracy": word_accuracy,
            "sentence_accuracy": sentence_accuracy,
            "total_errors": total_errors,
            "fixed_errors": fixed_errors,
            "new_errors": new_errors,
            "error_fix_rate": word_accuracy,
        }

    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive test suite"""
        
        # Test cases with expected outputs
        test_cases = [
            # Basic typos
            ("I beleive this is teh correct answr.", "I believe this is the correct answer."),
            ("This is a sentnce with mny typos.", "This is a sentence with many typos."),
            ("The qick brown fox jumps ovr the lazy dog.", "The quick brown fox jumps over the lazy dog."),
            
            # Common misspellings
            ("Please chck your email for futher instructions.", "Please check your email for further instructions."),
            ("I recieved your mesage yesterday.", "I received your message yesterday."),
            ("We need to discus this matter urgently.", "We need to discuss this matter urgently."),
            ("The meetng is schedled for tomorrow.", "The meeting is scheduled for tomorrow."),
            ("Can you plese send me the documnt?", "Can you please send me the document?"),
            
            # Complex cases
            ("This is alredy completd.", "This is already completed."),
            ("I dont understnd what you mean.", "I don't understand what you mean."),
            ("Ther are sevral erors in you're text.", "There are several errors in your text."),
            ("Its importnt to check you're work carefuly.", "It's important to check your work carefully."),
            
            # Challenging cases
            ("The restarant serves excelent food and has grate sevice.", 
             "The restaurant serves excellent food and has great service."),
            ("Acording to the studys, this methd is very efective.", 
             "According to the studies, this method is very effective."),
            ("The sofware updats fixed mny bugs and improvd performace.", 
             "The software updates fixed many bugs and improved performance."),
        ]
        
        print("="*80)
        print("🧪 COMPREHENSIVE BYT5 TYPO FIXER TEST")
        print("="*80)
        print(f"Model: {self.model_path}")
        print(f"Prefix: '{self.prefix}'")
        print(f"Test cases: {len(test_cases)}")
        print()
        
        results = []
        total_time = 0
        total_word_accuracy = 0
        total_sentence_accuracy = 0
        
        for i, (original, expected) in enumerate(test_cases, 1):
            print(f"Test {i:2d}:")
            print(f"  Original:  '{original}'")
            print(f"  Expected:  '{expected}'")
            
            # Run inference
            start_time = time.time()
            corrected = self.fix_typos(original)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            print(f"  Generated: '{corrected}'")
            print(f"  Time:      {inference_time:.3f}s")
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(original, corrected, expected)
            total_word_accuracy += accuracy["word_accuracy"]
            total_sentence_accuracy += accuracy["sentence_accuracy"]
            
            # Status
            if accuracy["sentence_accuracy"] == 1.0:
                status = "✅ Perfect"
            elif accuracy["word_accuracy"] > 0.8:
                status = "🟡 Good"
            elif accuracy["word_accuracy"] > 0.3:
                status = "🟠 Partial"
            else:
                status = "❌ Poor"
            
            print(f"  Accuracy:  {accuracy['word_accuracy']:.1%} words, " +
                  f"{accuracy['sentence_accuracy']:.1%} sentence - {status}")
            print()
            
            results.append({
                "test_id": i,
                "original": original,
                "expected": expected,
                "generated": corrected,
                "inference_time": inference_time,
                **accuracy
            })
        
        # Summary statistics
        avg_word_accuracy = total_word_accuracy / len(test_cases)
        avg_sentence_accuracy = total_sentence_accuracy / len(test_cases)
        avg_time = total_time / len(test_cases)
        
        print("="*80)
        print("📊 FINAL RESULTS")
        print("="*80)
        print(f"Overall Word Accuracy:      {avg_word_accuracy:.1%}")
        print(f"Overall Sentence Accuracy:  {avg_sentence_accuracy:.1%}")
        print(f"Perfect Sentences:          {int(total_sentence_accuracy)}/{len(test_cases)}")
        print(f"Average Inference Time:     {avg_time:.3f}s")
        print(f"Total Test Time:           {total_time:.1f}s")
        print()
        
        # Performance classification
        if avg_word_accuracy >= 0.9:
            performance = "🏆 EXCELLENT"
        elif avg_word_accuracy >= 0.8:
            performance = "✅ GOOD"
        elif avg_word_accuracy >= 0.6:
            performance = "🟡 FAIR"
        elif avg_word_accuracy >= 0.4:
            performance = "🟠 POOR"
        else:
            performance = "❌ FAILED"
        
        print(f"Overall Performance: {performance}")
        print()
        
        return {
            "model_path": self.model_path,
            "prefix": self.prefix,
            "device": str(self.device),
            "test_cases": len(test_cases),
            "avg_word_accuracy": avg_word_accuracy,
            "avg_sentence_accuracy": avg_sentence_accuracy,
            "perfect_sentences": int(total_sentence_accuracy),
            "avg_inference_time": avg_time,
            "total_test_time": total_time,
            "performance_level": performance,
            "detailed_results": results,
        }

def main():
    parser = argparse.ArgumentParser(description="Test ByT5 typo fixer model")
    parser.add_argument("--model-path", required=True, help="Path to model (local or HuggingFace)")
    parser.add_argument("--prefix", default="fix typos:", help="Instruction prefix")
    parser.add_argument("--output-file", default=None, help="Save results to JSON file")
    parser.add_argument("--max-length", type=int, default=512, help="Max generation length")
    parser.add_argument("--num-beams", type=int, default=4, help="Number of beams")
    
    args = parser.parse_args()
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA not available, using CPU")
    print()
    
    # Run tests
    tester = ByT5TypoTester(args.model_path, args.prefix)
    results = tester.run_comprehensive_test()
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {args.output_file}")
    
    # Return appropriate exit code
    if results["avg_word_accuracy"] >= 0.8:
        print("🎉 Model passed quality threshold!")
        exit(0)
    else:
        print("⚠️  Model below quality threshold - needs more training")
        exit(1)

if __name__ == "__main__":
    main()
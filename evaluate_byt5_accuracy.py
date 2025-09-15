#!/usr/bin/env python3
"""
Accuracy evaluation script for the ByT5 small typo fixer model
"""

import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
import difflib
from typing import List, Tuple

def load_model():
    """Load the ByT5 typo fixer model and tokenizer"""
    model_path = "./models/byt5-small-typo-fixer"
    
    print("Loading ByT5 typo fixer model...")
    tokenizer = ByT5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, device

def fix_typos(text, model, tokenizer, device):
    """Fix typos in the given text"""
    input_text = f"fix typos: {text}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def calculate_word_accuracy(original: str, corrected: str, expected: str) -> Tuple[float, List[str]]:
    """Calculate word-level accuracy and return corrections made"""
    original_words = original.lower().split()
    corrected_words = corrected.lower().split()
    expected_words = expected.lower().split()
    
    corrections = []
    correct_fixes = 0
    total_needed_fixes = 0
    
    # Find words that should be different between original and expected
    for i, (orig_word, exp_word) in enumerate(zip(original_words, expected_words)):
        if orig_word != exp_word:
            total_needed_fixes += 1
            # Check if the model corrected this word properly
            if i < len(corrected_words) and corrected_words[i] == exp_word:
                correct_fixes += 1
                corrections.append(f"✅ '{orig_word}' → '{exp_word}'")
            else:
                actual = corrected_words[i] if i < len(corrected_words) else "[missing]"
                corrections.append(f"❌ '{orig_word}' → '{actual}' (expected '{exp_word}')")
    
    accuracy = correct_fixes / total_needed_fixes if total_needed_fixes > 0 else 1.0
    return accuracy, corrections

def run_accuracy_evaluation():
    """Run comprehensive accuracy evaluation"""
    
    # Load model
    model, tokenizer, device = load_model()
    
    # Test cases with expected corrections
    test_cases = [
        ("I beleive this is teh correct answr.", "I believe this is the correct answer."),
        ("This is a sentnce with mny typos.", "This is a sentence with many typos."),
        ("The qick brown fox jumps ovr the lazy dog.", "The quick brown fox jumps over the lazy dog."),
        ("Please chck your email for futher instructions.", "Please check your email for further instructions."),
        ("I recieved your mesage yesterday.", "I received your message yesterday."),
        ("We need to discus this matter urgently.", "We need to discuss this matter urgently."),
        ("The meetng is schedled for tomorrow.", "The meeting is scheduled for tomorrow."),
        ("Can you plese send me the documnt?", "Can you please send me the document?"),
        ("This is alredy completd.", "This is already completed."),
        ("I dont understnd what you mean.", "I don't understand what you mean."),
        ("Ther are sevral erors in you're text.", "There are several errors in your text."),
        ("Its importnt to check you're work carefuly.", "It's important to check your work carefully."),
    ]
    
    print("\n" + "="*80)
    print("BYT5 SMALL TYPO FIXER - ACCURACY EVALUATION")
    print("="*80)
    
    total_accuracy = 0
    sentence_level_correct = 0
    
    for i, (typo_text, expected_text) in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Original:  '{typo_text}'")
        print(f"Expected:  '{expected_text}'")
        
        corrected = fix_typos(typo_text, model, tokenizer, device)
        print(f"Generated: '{corrected}'")
        
        # Calculate accuracy
        accuracy, corrections = calculate_word_accuracy(typo_text, corrected, expected_text)
        total_accuracy += accuracy
        
        # Check sentence-level accuracy
        if corrected.lower().strip() == expected_text.lower().strip():
            sentence_level_correct += 1
            print("✅ Perfect sentence match!")
        else:
            print("❌ Sentence differs from expected")
        
        print(f"Word-level accuracy: {accuracy:.1%}")
        if corrections:
            print("Corrections:")
            for correction in corrections:
                print(f"  {correction}")
    
    # Final results
    avg_word_accuracy = total_accuracy / len(test_cases)
    sentence_accuracy = sentence_level_correct / len(test_cases)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Average word-level accuracy: {avg_word_accuracy:.1%}")
    print(f"Sentence-level accuracy: {sentence_accuracy:.1%}")
    print(f"Model: ByT5-small (~300M parameters)")
    print(f"Perfect sentence matches: {sentence_level_correct}/{len(test_cases)}")

if __name__ == "__main__":
    run_accuracy_evaluation()
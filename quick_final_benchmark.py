#!/usr/bin/env python3
"""Quick final benchmark to see overall performance."""

from fixed_explicit_corrector import FixedExplicitCorrector
import time

def quick_benchmark():
    corrector = FixedExplicitCorrector()
    
    # Test cases with expected results
    test_cases = [
        ("Thi is a test", "This is a test"),
        ("The quikc brown fox", "The quick brown fox"),
        ("She recieved a beutiful gift", "She received a beautiful gift"),
        ("Can you halp me with this problm?", "Can you help me with this problem?"),
        ("Its importnt to chek the speling", "Its important to check the spelling"),
        ("I dont no wat to do", "I don't know what to do"),
        ("The wheather is realy nice", "The weather is really nice"),
        ("We ned to finsh this projct", "We need to finish this project"),
    ]
    
    perfect_count = 0
    total_word_accuracy = 0
    total_time = 0
    
    print("🎯 QUICK BENCHMARK RESULTS")
    print("=" * 50)
    
    for original, expected in test_cases:
        start_time = time.time()
        corrected, metadata = corrector.correct_text(original)
        end_time = time.time()
        
        # Calculate metrics
        correction_time = (end_time - start_time) * 1000
        total_time += correction_time
        
        # Word accuracy
        corrected_words = corrected.lower().split()
        expected_words = expected.lower().split()
        
        if len(corrected_words) == len(expected_words):
            correct_words = sum(1 for c, e in zip(corrected_words, expected_words) if c == e)
            word_accuracy = correct_words / len(expected_words)
        else:
            word_accuracy = 0.0
        
        total_word_accuracy += word_accuracy
        
        # Sentence accuracy
        is_perfect = corrected.lower().strip() == expected.lower().strip()
        if is_perfect:
            perfect_count += 1
            status = "✅ PERFECT"
        else:
            status = f"❌ {word_accuracy*100:.1f}% words"
        
        print(f"Original:  {original}")
        print(f"Expected:  {expected}")
        print(f"Got:       {corrected}")
        print(f"Status:    {status} ({correction_time:.1f}ms)")
        print(f"Fixes:     {len(metadata['corrections_made'])}")
        print()
    
    # Summary
    full_sentence_accuracy = (perfect_count / len(test_cases)) * 100
    avg_word_accuracy = (total_word_accuracy / len(test_cases)) * 100
    avg_time = total_time / len(test_cases)
    
    print("📊 SUMMARY")
    print("-" * 30)
    print(f"Full Sentence Accuracy: {full_sentence_accuracy:.1f}% ({perfect_count}/{len(test_cases)})")
    print(f"Average Word Accuracy:  {avg_word_accuracy:.1f}%")
    print(f"Average Time:          {avg_time:.1f}ms per sentence")
    print(f"Throughput:            {1000/avg_time:.1f} sentences/second")
    
    # Rating
    if avg_word_accuracy >= 90:
        accuracy_rating = "🎯 EXCELLENT (TARGET MET!)"
    elif avg_word_accuracy >= 80:
        accuracy_rating = "👍 GOOD"
    elif avg_word_accuracy >= 70:
        accuracy_rating = "⚠️ NEEDS IMPROVEMENT"
    else:
        accuracy_rating = "❌ POOR"
    
    print(f"Accuracy Rating:       {accuracy_rating}")

if __name__ == "__main__":
    quick_benchmark()
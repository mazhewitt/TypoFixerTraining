#!/usr/bin/env python3
"""
Comprehensive benchmark of the explicit format typo corrector.
Measures both full sentence accuracy and timing performance.
"""
import torch
import time
import json
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from fixed_explicit_corrector import FixedExplicitCorrector
import statistics

def load_test_data():
    """Load test data for benchmarking."""
    test_sentences = [
        # Multi-typo sentences (challenging cases)
        "Thi sis a test sentenc with multipl typos that ned correction.",
        "The quikc brown fox jumps over teh lazy dog in thee forest.",
        "I went too the stor to buy som milk and bred for diner.",
        "Can you halp me solv this problm with my computr?",
        "She recieved a beutiful gift form her frend yesterday.",
        "Its importnt to chek you're grammer befor submiting.",
        "The wheather is realy nice todya, lets go outsid.",
        "We ned to mak sure evry thing is workng proparly.",
        "Pleas let me no if you hav any questons or concernes.",
        "There performanc was excelent at the concrt last nite.",
        
        # Common typo patterns
        "I dont no wat to do abut this situaton.",
        "Your definatly goin to lov this new restrant.",
        "We shoud of went to the part earlyer today.",
        "Ther house is loacted neer the shoping center.",
        "He payed alot of mony for that expensiv car.",
        "I recieved you're mesage abut the meating tomorow.",
        "Its posible that we migt hav to chang our planse.",
        "The temprature is to hie for this tim of yer.",
        "We wer abel to finsh the projct on tim.",
        "Pleas chek the speling in you're documnt.",
        
        # Shorter sentences
        "Thi is a test.",
        "Helo world!",
        "How ar you?",
        "Its goin wel.",
        "I hav a questn.",
    ]
    
    # Expected corrections (manual ground truth)
    expected_corrections = [
        "This is a test sentence with multiple typos that need correction.",
        "The quick brown fox jumps over the lazy dog in the forest.",
        "I went to the store to buy some milk and bread for dinner.",
        "Can you help me solve this problem with my computer?",
        "She received a beautiful gift from her friend yesterday.",
        "Its important to check you're grammar before submitting.",
        "The weather is really nice today, lets go outside.",
        "We need to make sure every thing is working properly.",
        "Please let me know if you have any questions or concerns.",
        "There performance was excellent at the concert last night.",
        
        "I dont know what to do about this situation.",
        "Your definitely going to love this new restaurant.",
        "We should of went to the part earlier today.",
        "Their house is located near the shopping center.",
        "He paid alot of money for that expensive car.",
        "I received you're message about the meeting tomorrow.",
        "Its possible that we might have to change our plans.",
        "The temperature is to high for this time of year.",
        "We were able to finish the project on time.",
        "Please check the spelling in you're document.",
        
        "This is a test.",
        "Hello world!",
        "How are you?",
        "Its going well.",
        "I have a question.",
    ]
    
    return test_sentences, expected_corrections

def calculate_sentence_accuracy(corrected, expected):
    """Calculate if the entire sentence was corrected perfectly."""
    return corrected.strip().lower() == expected.strip().lower()

def calculate_word_accuracy(corrected, expected):
    """Calculate percentage of words corrected correctly."""
    corrected_words = corrected.strip().lower().split()
    expected_words = expected.strip().lower().split()
    
    if len(corrected_words) != len(expected_words):
        return 0.0  # Length mismatch = fail
    
    correct_words = sum(1 for c, e in zip(corrected_words, expected_words) if c == e)
    return correct_words / len(expected_words) if expected_words else 0.0

def benchmark_corrector():
    """Run comprehensive benchmark."""
    print("🔍 BENCHMARKING EXPLICIT TYPO CORRECTOR")
    print("=" * 60)
    
    # Initialize corrector
    print("Loading model...")
    corrector = FixedExplicitCorrector()
    
    # Load test data
    test_sentences, expected_corrections = load_test_data()
    
    # Benchmark metrics
    sentence_accuracies = []
    word_accuracies = []
    correction_times = []
    total_sentences = len(test_sentences)
    perfect_sentences = 0
    
    print(f"\nTesting {total_sentences} sentences...\n")
    
    # Process each test case
    for i, (original, expected) in enumerate(zip(test_sentences, expected_corrections), 1):
        print(f"[{i:2d}/{total_sentences}] Testing: {original}")
        
        # Time the correction
        start_time = time.time()
        corrected, metadata = corrector.correct_text(original)
        end_time = time.time()
        
        correction_time = (end_time - start_time) * 1000  # Convert to milliseconds
        correction_times.append(correction_time)
        
        # Calculate accuracies
        sentence_perfect = calculate_sentence_accuracy(corrected, expected)
        word_accuracy = calculate_word_accuracy(corrected, expected)
        
        sentence_accuracies.append(1.0 if sentence_perfect else 0.0)
        word_accuracies.append(word_accuracy)
        
        if sentence_perfect:
            perfect_sentences += 1
            status = "✅ PERFECT"
        else:
            status = f"❌ {word_accuracy*100:.1f}% words"
        
        print(f"     Result: {corrected}")
        print(f"   Expected: {expected}")
        print(f"     Status: {status} ({correction_time:.1f}ms)")
        print()
    
    # Calculate overall metrics
    full_sentence_accuracy = (perfect_sentences / total_sentences) * 100
    avg_word_accuracy = statistics.mean(word_accuracies) * 100
    avg_time = statistics.mean(correction_times)
    median_time = statistics.median(correction_times)
    min_time = min(correction_times)
    max_time = max(correction_times)
    
    # Print comprehensive results
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"📝 ACCURACY METRICS:")
    print(f"   Full Sentence Accuracy: {full_sentence_accuracy:.1f}% ({perfect_sentences}/{total_sentences})")
    print(f"   Average Word Accuracy:  {avg_word_accuracy:.1f}%")
    print()
    print(f"⏱️  TIMING METRICS:")
    print(f"   Average Time:  {avg_time:.1f}ms per sentence")
    print(f"   Median Time:   {median_time:.1f}ms per sentence")
    print(f"   Fastest:       {min_time:.1f}ms")
    print(f"   Slowest:       {max_time:.1f}ms")
    print(f"   Throughput:    {1000/avg_time:.1f} sentences/second")
    print()
    
    # Performance categorization
    if avg_time < 100:
        speed_rating = "🚀 VERY FAST"
    elif avg_time < 500:
        speed_rating = "⚡ FAST"
    elif avg_time < 1000:
        speed_rating = "🐌 MODERATE"
    else:
        speed_rating = "🐢 SLOW"
    
    accuracy_rating = "🎯 EXCELLENT" if full_sentence_accuracy >= 80 else "⚠️ NEEDS IMPROVEMENT"
    
    print(f"🏆 OVERALL PERFORMANCE:")
    print(f"   Accuracy: {accuracy_rating}")
    print(f"   Speed:    {speed_rating}")
    
    # Detailed failure analysis
    failures = []
    for i, (original, expected, word_acc) in enumerate(zip(test_sentences, expected_corrections, word_accuracies)):
        if word_acc < 1.0:
            corrected, _ = corrector.correct_text(original)
            failures.append((i+1, original, expected, corrected, word_acc))
    
    if failures:
        print(f"\n❌ FAILURE ANALYSIS ({len(failures)} sentences need improvement):")
        print("-" * 60)
        for idx, original, expected, corrected, word_acc in failures[:5]:  # Show top 5 failures
            print(f"[{idx}] Word Accuracy: {word_acc*100:.1f}%")
            print(f"    Original:  {original}")
            print(f"    Expected:  {expected}")
            print(f"    Got:       {corrected}")
            print()
    
    return {
        'full_sentence_accuracy': full_sentence_accuracy,
        'avg_word_accuracy': avg_word_accuracy,
        'avg_time_ms': avg_time,
        'median_time_ms': median_time,
        'throughput_per_sec': 1000/avg_time,
        'perfect_sentences': perfect_sentences,
        'total_sentences': total_sentences
    }

if __name__ == "__main__":
    results = benchmark_corrector()
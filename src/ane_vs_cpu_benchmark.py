#!/usr/bin/env python3
"""
Benchmark Apple Neural Engine vs CPU performance for DistilBERT.
This will prove whether ANE acceleration is actually working.
"""

import argparse
import logging
import numpy as np
import coremltools as ct
from transformers import DistilBertTokenizer
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ANECPUBenchmark:
    """
    Benchmark engine to compare ANE vs CPU performance on the same model.
    """
    
    def __init__(self, coreml_model_path: str, tokenizer_path: str, max_sequence_length: int = 128):
        self.max_seq_len = max_sequence_length
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        
        # Load model with ANE enabled
        logger.info(f"Loading Core ML model with ANE enabled...")
        self.ane_model = ct.models.MLModel(coreml_model_path)
        
        # Load model with CPU-only
        logger.info(f"Loading Core ML model with CPU-only...")
        self.cpu_model = ct.models.MLModel(coreml_model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        
        logger.info("✅ Both ANE and CPU models loaded!")
    
    def _prepare_inputs(self, text: str):
        """Prepare inputs for inference."""
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True
        )
        
        return {
            'input_ids': inputs['input_ids'].astype(np.int32),
            'attention_mask': inputs['attention_mask'].astype(np.int32)
        }
    
    def benchmark_model(self, model, model_name: str, test_text: str, num_runs: int = 50):
        """Benchmark a specific model configuration."""
        logger.info(f"Benchmarking {model_name} with {num_runs} runs...")
        
        # Prepare inputs
        inputs = self._prepare_inputs(test_text)
        
        # Warmup runs
        logger.info(f"Warming up {model_name}...")
        for _ in range(5):
            model.predict(inputs)
        
        # Benchmark runs
        times = []
        logger.info(f"Running {model_name} benchmark...")
        
        for i in range(num_runs):
            start_time = time.time()
            prediction = model.predict(inputs)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        throughput = 1000 / avg_time
        
        stats = {
            'model_name': model_name,
            'num_runs': num_runs,
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'throughput_per_sec': throughput,
        }
        
        logger.info(f"📊 {model_name} Results:")
        logger.info(f"   Average time: {avg_time:.2f}ms")
        logger.info(f"   Min time: {min_time:.2f}ms")
        logger.info(f"   Max time: {max_time:.2f}ms")
        logger.info(f"   Std deviation: {std_time:.2f}ms")
        logger.info(f"   Throughput: {throughput:.1f} inferences/second")
        
        return stats
    
    def compare_ane_vs_cpu(self, test_text: str = None, num_runs: int = 50):
        """
        Compare ANE vs CPU performance on the same model.
        This is the key test to prove ANE acceleration works.
        """
        if test_text is None:
            test_text = "This is a test sentence with some potential typos to correct"
        
        logger.info("🚀 Starting ANE vs CPU comparison benchmark...")
        logger.info(f"Test text: '{test_text}'")
        logger.info(f"Sequence length: {self.max_seq_len}")
        logger.info(f"Benchmark runs: {num_runs}")
        
        # Benchmark ANE model
        ane_stats = self.benchmark_model(self.ane_model, "Apple Neural Engine", test_text, num_runs)
        
        # Benchmark CPU model  
        cpu_stats = self.benchmark_model(self.cpu_model, "CPU Only", test_text, num_runs)
        
        # Calculate speedup
        speedup = cpu_stats['avg_time_ms'] / ane_stats['avg_time_ms']
        throughput_improvement = ane_stats['throughput_per_sec'] / cpu_stats['throughput_per_sec']
        
        # Print comparison results
        logger.info("\n" + "="*60)
        logger.info("🏆 ANE vs CPU COMPARISON RESULTS")
        logger.info("="*60)
        logger.info(f"Apple Neural Engine: {ane_stats['avg_time_ms']:.2f}ms avg")
        logger.info(f"CPU Only:            {cpu_stats['avg_time_ms']:.2f}ms avg")
        logger.info(f"")
        logger.info(f"🚀 ANE Speedup:      {speedup:.2f}x faster")
        logger.info(f"⚡ Throughput gain:  {throughput_improvement:.2f}x more inferences/sec")
        logger.info("="*60)
        
        if speedup > 1.5:
            logger.info("✅ ANE ACCELERATION CONFIRMED! Neural Engine is significantly faster!")
        elif speedup > 1.1:
            logger.info("✅ ANE acceleration working, but modest speedup")
        else:
            logger.warning("❌ ANE acceleration not working - similar or slower performance")
        
        return {
            'ane_stats': ane_stats,
            'cpu_stats': cpu_stats,
            'speedup': speedup,
            'throughput_improvement': throughput_improvement,
            'ane_working': speedup > 1.5
        }
    
    def detailed_analysis(self, test_text: str = None, num_runs: int = 100):
        """
        Run detailed analysis to understand performance characteristics.
        """
        if test_text is None:
            test_text = "The quick brown fox jumps over the lazy dog with some typos"
        
        logger.info("📊 Running detailed performance analysis...")
        
        # Test different sequence lengths
        seq_lengths = [32, 64, 128]
        results = {}
        
        for seq_len in seq_lengths:
            logger.info(f"\nTesting sequence length: {seq_len}")
            
            # Temporarily change sequence length
            original_seq_len = self.max_seq_len
            self.max_seq_len = seq_len
            
            # Run comparison
            comparison = self.compare_ane_vs_cpu(test_text, num_runs//2)
            results[seq_len] = comparison
            
            # Restore original sequence length
            self.max_seq_len = original_seq_len
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("📈 DETAILED PERFORMANCE ANALYSIS")
        logger.info("="*80)
        logger.info("Seq Len | ANE (ms) | CPU (ms) | Speedup | ANE Working")
        logger.info("-"*80)
        
        for seq_len, result in results.items():
            ane_time = result['ane_stats']['avg_time_ms']
            cpu_time = result['cpu_stats']['avg_time_ms']
            speedup = result['speedup']
            working = "✅ YES" if result['ane_working'] else "❌ NO"
            
            logger.info(f"{seq_len:7d} | {ane_time:8.2f} | {cpu_time:8.2f} | {speedup:7.2f}x | {working}")
        
        logger.info("="*80)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark ANE vs CPU performance")
    parser.add_argument('--coreml_model', type=str,
                       default='/Users/mazdahewitt/projects/train-typo-fixer/apple-ane-distilbert/DistilBERT_fp16.mlpackage',
                       help='Path to Core ML model')
    parser.add_argument('--tokenizer', type=str,
                       default='/Users/mazdahewitt/projects/train-typo-fixer/apple-ane-distilbert',
                       help='Path to tokenizer')
    parser.add_argument('--text', type=str,
                       default="This is a benchmark test sentence with potential typos to correct",
                       help='Test text for benchmarking')
    parser.add_argument('--num_runs', type=int, default=50,
                       help='Number of benchmark runs per model')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed analysis with different sequence lengths')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    try:
        logger.info("🔥 Starting ANE vs CPU benchmark comparison...")
        
        # Initialize benchmark engine
        benchmark = ANECPUBenchmark(
            coreml_model_path=args.coreml_model,
            tokenizer_path=args.tokenizer,
            max_sequence_length=args.max_seq_len
        )
        
        if args.detailed:
            # Run detailed analysis
            benchmark.detailed_analysis(args.text, args.num_runs)
        else:
            # Run basic comparison
            comparison = benchmark.compare_ane_vs_cpu(args.text, args.num_runs)
            
            # Final verdict
            if comparison['ane_working']:
                logger.info(f"\n🎉 SUCCESS: ANE acceleration is working! {comparison['speedup']:.2f}x speedup confirmed!")
            else:
                logger.info(f"\n😞 ANE acceleration not detected. Only {comparison['speedup']:.2f}x speedup.")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
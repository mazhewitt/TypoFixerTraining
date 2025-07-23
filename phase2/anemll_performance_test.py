#!/usr/bin/env python3
"""
Performance test using anemll's working chat interface.
This validates our ANE model performance using the proven anemll toolkit.
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnemllPerformanceTest:
    def __init__(self, model_path: str, anemll_path: str):
        """Initialize with model and anemll paths."""
        self.model_path = Path(model_path)
        self.anemll_path = Path(anemll_path)
        self.meta_yaml = self.model_path / "meta.yaml"
        
        if not self.meta_yaml.exists():
            raise FileNotFoundError(f"meta.yaml not found at {self.meta_yaml}")
        
        # Test sentences for typo correction
        self.test_cases = [
            "Thi sis a test sentenc with typos",
            "The quikc brown fox jumps over teh lazy dog",
            "I went too the stor to buy som milk",
            "Plese corect thes mistaks in sentenc",
            "Helllo wrold how ar you todya"
        ]
        
    def run_single_inference(self, prompt: str, max_tokens: int = 20) -> Tuple[str, float, float]:
        """Run single inference using anemll's chat interface."""
        
        # Prepare command
        cmd = [
            "python3", 
            str(self.anemll_path / "tests" / "chat.py"),
            "--meta", str(self.meta_yaml),
            "--max-tokens", str(max_tokens),
            "--prompt", prompt,
            "--no-template"  # Disable chat template for direct prompting
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
                cwd=str(self.anemll_path)
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Parse performance info from output
            output_lines = result.stdout.strip().split('\n')
            response_text = ""
            tokens_per_sec = 0.0
            
            # Extract response and performance
            for line in output_lines:
                # Look for performance data
                if "t/s" in line and "Generated" in line:
                    # Extract tokens/second from line like "Total: Generated 20 tokens in 0.25s"
                    match = re.search(r'Generated (\d+) tokens in ([\d.]+)s', line)
                    if match:
                        tokens = int(match.group(1))
                        duration = float(match.group(2))
                        tokens_per_sec = tokens / duration if duration > 0 else 0
                elif "Assistant:" in line:
                    response_text = line.replace("Assistant:", "").strip()
                elif line.strip() and not any(x in line for x in ["scikit-learn", "Torch version", "urllib3", "Warning:", "Loading", "Using", "Created", "Initialized"]):
                    # Clean response extraction
                    if not response_text:
                        response_text = line.strip()
            
            return response_text, total_time, tokens_per_sec
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout for prompt: {prompt}")
            return "", 0.0, 0.0
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running inference: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return "", 0.0, 0.0
    
    def run_performance_benchmark(self, num_runs: int = 5) -> Dict:
        """Run comprehensive performance benchmark."""
        
        logger.info(f"🚀 Running anemll performance test on {len(self.test_cases)} test cases, {num_runs} runs each")
        
        results = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': str(self.model_path),
                'anemll_path': str(self.anemll_path),
                'num_runs_per_case': num_runs,
                'test_cases_count': len(self.test_cases)
            },
            'performance': {
                'test_cases': [],
                'summary': {}
            }
        }
        
        all_inference_times = []
        all_tokens_per_sec = []
        
        for i, test_case in enumerate(self.test_cases):
            logger.info(f"📝 Testing case {i+1}/{len(self.test_cases)}: \"{test_case[:50]}...\"")
            
            case_results = {
                'input': test_case,
                'runs': [],
                'statistics': {}
            }
            
            case_times = []
            case_tokens_per_sec = []
            
            for run in range(num_runs):
                logger.info(f"  Run {run+1}/{num_runs}")
                
                # Create correction prompt
                prompt = f"Correct the typos: {test_case}"
                
                response, inference_time, tokens_per_sec = self.run_single_inference(prompt, max_tokens=30)
                
                run_data = {
                    'run_number': run + 1,
                    'prompt': prompt,
                    'response': response,
                    'inference_time': inference_time,
                    'tokens_per_second': tokens_per_sec
                }
                
                case_results['runs'].append(run_data)
                
                if inference_time > 0:
                    case_times.append(inference_time)
                    all_inference_times.append(inference_time)
                
                if tokens_per_sec > 0:
                    case_tokens_per_sec.append(tokens_per_sec)
                    all_tokens_per_sec.append(tokens_per_sec)
                
                # Brief pause between runs
                time.sleep(0.5)
            
            # Calculate case statistics
            if case_times:
                case_results['statistics'] = {
                    'mean_inference_time': sum(case_times) / len(case_times),
                    'median_inference_time': sorted(case_times)[len(case_times)//2],
                    'min_inference_time': min(case_times),
                    'max_inference_time': max(case_times),
                    'successful_runs': len(case_times),
                    'failed_runs': num_runs - len(case_times)
                }
                
                if case_tokens_per_sec:
                    case_results['statistics'].update({
                        'mean_tokens_per_second': sum(case_tokens_per_sec) / len(case_tokens_per_sec),
                        'median_tokens_per_second': sorted(case_tokens_per_sec)[len(case_tokens_per_sec)//2]
                    })
            
            results['performance']['test_cases'].append(case_results)
        
        # Overall summary
        if all_inference_times:
            results['performance']['summary'] = {
                'overall_mean_inference_time': sum(all_inference_times) / len(all_inference_times),
                'overall_median_inference_time': sorted(all_inference_times)[len(all_inference_times)//2],
                'overall_min_inference_time': min(all_inference_times),
                'overall_max_inference_time': max(all_inference_times),
                'total_successful_runs': len(all_inference_times),
                'total_failed_runs': (len(self.test_cases) * num_runs) - len(all_inference_times)
            }
            
            if all_tokens_per_sec:
                results['performance']['summary'].update({
                    'overall_mean_tokens_per_second': sum(all_tokens_per_sec) / len(all_tokens_per_sec),
                    'overall_median_tokens_per_second': sorted(all_tokens_per_sec)[len(all_tokens_per_sec)//2]
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Performance test using anemll's chat interface")
    parser.add_argument('--model_path', type=str, default="../models/qwen-ane-test",
                       help='Path to ANE model directory')
    parser.add_argument('--anemll_path', type=str, default="../anemll",
                       help='Path to anemll toolkit directory')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs per test case')
    parser.add_argument('--output_file', type=str, default="anemll_performance_results.json",
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    try:
        tester = AnemllPerformanceTest(args.model_path, args.anemll_path)
        results = tester.run_performance_benchmark(args.num_runs)
        
        # Save results
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        logger.info("\\n" + "="*60)
        logger.info("🎯 ANEMLL PERFORMANCE TEST RESULTS")
        logger.info("="*60)
        
        summary = results['performance']['summary']
        if summary:
            logger.info(f"📊 Total successful runs: {summary.get('total_successful_runs', 0)}")
            logger.info(f"📊 Total failed runs: {summary.get('total_failed_runs', 0)}")
            logger.info(f"📊 Mean inference time: {summary.get('overall_mean_inference_time', 0)*1000:.1f}ms")
            logger.info(f"📊 Median inference time: {summary.get('overall_median_inference_time', 0)*1000:.1f}ms")
            
            if 'overall_mean_tokens_per_second' in summary:
                logger.info(f"📊 Mean tokens/second: {summary['overall_mean_tokens_per_second']:.1f}")
                logger.info(f"📊 Median tokens/second: {summary['overall_median_tokens_per_second']:.1f}")
        
        logger.info(f"💾 Full results saved to {args.output_file}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error running performance test: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
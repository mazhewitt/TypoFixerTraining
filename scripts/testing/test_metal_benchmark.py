#!/usr/bin/env python3
"""
Metal GPU benchmark for Qwen typo correction model on macOS.
Tests performance with Apple's Metal Performance Shaders acceleration.
"""

import torch
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import platform
import subprocess

def get_system_info():
    """Get detailed system information for the benchmark."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_mps_available": torch.backends.mps.is_available(),
        "torch_mps_built": torch.backends.mps.is_built(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
    }
    
    # Try to get GPU info
    try:
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Chipset Model:' in line:
                    info["gpu"] = line.split('Chipset Model:')[1].strip()
                    break
    except:
        info["gpu"] = "Unknown"
    
    return info

def conservative_inference_metal(model, tokenizer, prompt: str, device) -> tuple:
    """Conservative inference with timing on specified device."""
    start_time = time.time()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tokenize_time = time.time()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
    generate_time = time.time()
    
    # Decode
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    decode_time = time.time()
    
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
    
    end_time = time.time()
    
    # Timing breakdown
    timings = {
        "total": end_time - start_time,
        "tokenize": tokenize_time - start_time,
        "generate": generate_time - tokenize_time,
        "decode": decode_time - generate_time,
        "cleanup": end_time - decode_time
    }
    
    return corrected, timings

def benchmark_device(model, tokenizer, device_name, test_cases, warmup_runs=3):
    """Benchmark model performance on specified device."""
    print(f"\n🔥 Benchmarking on {device_name}")
    print("-" * 50)
    
    # Move model to device
    if device_name == "MPS (Metal)":
        device = torch.device("mps")
        model = model.to(device)
    elif device_name == "CPU":
        device = torch.device("cpu")
        model = model.to(device)
    else:
        raise ValueError(f"Unknown device: {device_name}")
    
    # Warmup runs
    print(f"🔄 Warming up with {warmup_runs} runs...")
    for i in range(warmup_runs):
        try:
            corrected, _ = conservative_inference_metal(model, tokenizer, test_cases[0], device)
            if i == 0:
                print(f"   Warmup output: {corrected}")
        except Exception as e:
            print(f"❌ Warmup failed: {e}")
            return None
    
    # Clear memory
    if device_name == "MPS (Metal)":
        torch.mps.empty_cache()
    gc.collect()
    
    # Benchmark runs
    print(f"📊 Running benchmark with {len(test_cases)} test cases...")
    
    all_timings = []
    results = []
    
    for i, prompt in enumerate(test_cases):
        try:
            corrected, timings = conservative_inference_metal(model, tokenizer, prompt, device)
            all_timings.append(timings)
            results.append({
                "input": prompt.replace("Fix: ", ""),
                "output": corrected,
                "time": timings["total"]
            })
            
            if i < 3:  # Show first 3 examples
                print(f"   {i+1}. {timings['total']:.3f}s: {corrected}")
        
        except Exception as e:
            print(f"❌ Test {i+1} failed: {e}")
            continue
    
    if not all_timings:
        print("❌ No successful runs!")
        return None
    
    # Calculate statistics
    total_times = [t["total"] for t in all_timings]
    generate_times = [t["generate"] for t in all_timings]
    
    stats = {
        "device": device_name,
        "total_tests": len(all_timings),
        "avg_total_time": sum(total_times) / len(total_times),
        "min_time": min(total_times),
        "max_time": max(total_times),
        "avg_generate_time": sum(generate_times) / len(generate_times),
        "throughput_per_sec": 1.0 / (sum(total_times) / len(total_times)),
    }
    
    # Print detailed results
    print(f"\n📈 {device_name} Results:")
    print(f"   Average total time: {stats['avg_total_time']:.3f}s")
    print(f"   Average generate time: {stats['avg_generate_time']:.3f}s")
    print(f"   Min time: {stats['min_time']:.3f}s")
    print(f"   Max time: {stats['max_time']:.3f}s")
    print(f"   Throughput: {stats['throughput_per_sec']:.1f} corrections/second")
    
    return stats, results

def main():
    print("🚀 Metal GPU Benchmark for Qwen Typo Correction")
    print("=" * 60)
    
    # System info
    sys_info = get_system_info()
    print(f"🖥️  System: {sys_info['platform']}")
    print(f"🧠 Processor: {sys_info['processor']}")
    print(f"🎯 GPU: {sys_info.get('gpu', 'Unknown')}")
    print(f"💾 RAM: {sys_info['memory_gb']}GB")
    print(f"🐍 Python: {sys_info['python_version']}")
    print(f"🔥 PyTorch: {sys_info['torch_version']}")
    print(f"⚡ MPS Available: {sys_info['torch_mps_available']}")
    print(f"🔧 MPS Built: {sys_info['torch_mps_built']}")
    
    if not sys_info['torch_mps_available']:
        print("❌ MPS (Metal) not available on this system!")
        print("💡 Running CPU-only benchmark...")
    
    # Load model
    model_path = "mazhewitt/qwen-typo-fixer"
    print(f"\n🤖 Loading model: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if sys_info['torch_mps_available'] else torch.float32
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test cases for benchmarking
    test_cases = [
        "Fix: I beleive this is teh correct answr.",
        "Fix: She recieved her degre last year.",
        "Fix: The resturant serves excelent food.",
        "Fix: He is studyng for his final examintion.",
        "Fix: We dicussed the importnt details yesterday.",
        "Fix: The begining of the story was excting.",
        "Fix: I definately need to imporve my skils.",
        "Fix: The experiance was chalenging and rewardng.",
        "Fix: Please chekc your emial for informaton.",
        "Fix: This documetn contians importatn points.",
    ]
    
    # Benchmark results
    results = {}
    
    # CPU Benchmark
    cpu_stats, cpu_results = benchmark_device(model, tokenizer, "CPU", test_cases)
    if cpu_stats:
        results["CPU"] = cpu_stats
    
    # Metal Benchmark (if available)
    if sys_info['torch_mps_available']:
        try:
            mps_stats, mps_results = benchmark_device(model, tokenizer, "MPS (Metal)", test_cases)
            if mps_stats:
                results["MPS (Metal)"] = mps_stats
        except Exception as e:
            print(f"❌ Metal benchmark failed: {e}")
    
    # Comparison
    if len(results) > 1:
        print(f"\n🏁 PERFORMANCE COMPARISON")
        print("=" * 60)
        
        cpu_time = results["CPU"]["avg_total_time"]
        mps_time = results["MPS (Metal)"]["avg_total_time"] 
        speedup = cpu_time / mps_time
        
        print(f"CPU Average Time:    {cpu_time:.3f}s")
        print(f"Metal Average Time:  {mps_time:.3f}s")
        print(f"Metal Speedup:       {speedup:.1f}x faster")
        print(f"Metal Throughput:    {results['MPS (Metal)']['throughput_per_sec']:.1f} corrections/sec")
        
        if speedup > 2:
            print("🚀 Excellent Metal acceleration!")
        elif speedup > 1.5:
            print("✅ Good Metal performance boost")
        else:
            print("⚠️ Modest Metal improvement")
    
    else:
        print(f"\n📊 Single Device Results")
        print("=" * 30)
        for device, stats in results.items():
            print(f"{device}: {stats['avg_total_time']:.3f}s avg, {stats['throughput_per_sec']:.1f} corrections/sec")

if __name__ == "__main__":
    main()
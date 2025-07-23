#!/usr/bin/env python3
"""
Quick sanity test to measure original Qwen 0.6B performance with GPU.
Establishes baseline tokens/second for comparison with ANE results.
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qwen_baseline_performance(model_name: str = "Qwen/Qwen3-0.6B", num_runs: int = 3):
    """Test original Qwen performance with GPU acceleration."""
    
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"📥 Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load with appropriate dtype for device
    if device.type == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    elif device.type == 'mps':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model = model.to(device)
    
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"📊 Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1_000_000:.1f}M")
    logger.info(f"📊 Vocabulary size: {tokenizer.vocab_size}")
    
    # Test prompts - same as our ANE tests
    test_prompts = [
        "Correct the typos: Thi sis a test sentenc with typos",
        "Correct the typos: The quikc brown fox jumps over teh lazy dog",
        "Correct the typos: I went too the stor to buy som milk",
        "Correct the typos: Plese corect thes mistaks in sentenc",
        "Correct the typos: Helllo wrold how ar you todya"
    ]
    
    all_results = []
    
    logger.info(f"🚀 Running {num_runs} tests per prompt...")
    
    for prompt_idx, prompt in enumerate(test_prompts):
        logger.info(f"\n📝 Testing prompt {prompt_idx + 1}/{len(test_prompts)}: \"{prompt[:50]}...\"")
        
        prompt_results = []
        
        for run in range(num_runs):
            logger.info(f"  Run {run + 1}/{num_runs}")
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            # Warm up (first run might be slower)
            if run == 0:
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
            
            # Actual timed generation
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,  # Same as ANE test
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            end_time = time.time()
            
            # Calculate metrics
            generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            inference_time = end_time - start_time
            tokens_per_second = generated_tokens / inference_time if inference_time > 0 else 0
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            result = {
                'run': run + 1,
                'prompt': prompt,
                'response': response[:100] + "..." if len(response) > 100 else response,
                'generated_tokens': generated_tokens,
                'inference_time': inference_time,
                'tokens_per_second': tokens_per_second
            }
            
            prompt_results.append(result)
            all_results.append(result)
            
            logger.info(f"    ⚡ {tokens_per_second:.1f} tokens/sec ({generated_tokens} tokens in {inference_time:.3f}s)")
            
            # Brief pause
            time.sleep(0.1)
        
        # Calculate prompt averages
        prompt_tps = [r['tokens_per_second'] for r in prompt_results]
        prompt_times = [r['inference_time'] for r in prompt_results]
        
        logger.info(f"  📊 Average TPS: {sum(prompt_tps)/len(prompt_tps):.1f}")
        logger.info(f"  📊 Average time: {sum(prompt_times)/len(prompt_times):.3f}s")
    
    # Overall summary
    overall_tps = [r['tokens_per_second'] for r in all_results]
    overall_times = [r['inference_time'] for r in all_results]
    overall_tokens = [r['generated_tokens'] for r in all_results]
    
    logger.info("\n" + "="*60)
    logger.info("🎯 QWEN 0.6B BASELINE PERFORMANCE RESULTS")
    logger.info("="*60)
    logger.info(f"🔧 Device: {device}")
    logger.info(f"📊 Total runs: {len(all_results)}")
    logger.info(f"📊 Average tokens/second: {sum(overall_tps)/len(overall_tps):.1f}")
    logger.info(f"📊 Median tokens/second: {sorted(overall_tps)[len(overall_tps)//2]:.1f}")
    logger.info(f"📊 Max tokens/second: {max(overall_tps):.1f}")
    logger.info(f"📊 Min tokens/second: {min(overall_tps):.1f}")
    logger.info(f"📊 Average inference time: {sum(overall_times)/len(overall_times):.3f}s")
    logger.info(f"📊 Average tokens generated: {sum(overall_tokens)/len(overall_tokens):.1f}")
    logger.info("="*60)
    
    return {
        'device': str(device),
        'model_name': model_name,
        'total_runs': len(all_results),
        'average_tps': sum(overall_tps)/len(overall_tps),
        'median_tps': sorted(overall_tps)[len(overall_tps)//2],
        'max_tps': max(overall_tps),
        'min_tps': min(overall_tps),
        'average_inference_time': sum(overall_times)/len(overall_times),
        'average_tokens_generated': sum(overall_tokens)/len(overall_tokens),
        'all_results': all_results
    }

def main():
    parser = argparse.ArgumentParser(description="Test original Qwen 0.6B performance")
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-0.6B",
                       help='Qwen model name/path')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per prompt')
    
    args = parser.parse_args()
    
    try:
        results = test_qwen_baseline_performance(args.model, args.runs)
        return 0
    except Exception as e:
        logger.error(f"❌ Error running baseline test: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
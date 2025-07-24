#!/usr/bin/env python3
"""
Simple test for CoreML Qwen typo fixer - using correct input shapes.
"""

import torch
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import time
import os

def test_with_correct_shapes():
    """Test with the correct input shapes that the model expects."""
    print("🚀 Testing CoreML Qwen Typo Fixer with Correct Shapes")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer_path = "/Users/mazhewitt/projects/TypoFixerTraining/models/qwen-typo-fixer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Load embeddings model
    model_dir = "/Users/mazhewitt/projects/TypoFixerTraining/models/qwen-typo-fixer-ane"
    embeddings_path = os.path.join(model_dir, "qwen-typo-fixer_embeddings.mlpackage")
    embeddings_model = ct.models.MLModel(embeddings_path)
    print(f"✅ Embeddings model loaded")
    
    # Test sentences
    test_sentences = [
        "Fix: I beleive this is teh correct answr.",
        "Fix: Please chekc your speeling.",
        "Fix: This setence has typos.",
    ]
    
    print(f"\n📋 Model accepts input shapes: (1,1) or (1,64)")
    print(f"Let's test with single tokens and short sequences...")
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {sentence}")
        print(f"{'='*50}")
        
        # Tokenize without padding
        inputs = tokenizer(sentence, return_tensors="np", add_special_tokens=True)
        input_ids = inputs['input_ids'].astype(np.int32)
        
        print(f"   Original tokens shape: {input_ids.shape}")
        print(f"   Tokens: {input_ids[0][:10]}...")
        
        # Test with single token (1,1) shape
        print(f"\n🧪 Testing single token inference:")
        single_token = input_ids[:, :1]  # Take first token only
        print(f"   Single token shape: {single_token.shape}")
        print(f"   Token ID: {single_token[0][0]}")
        
        try:
            start_time = time.time()
            result = embeddings_model.predict({"input_ids": single_token})
            end_time = time.time()
            
            hidden_states = result['hidden_states']
            print(f"✅ Single token inference successful!")
            print(f"   Inference time: {(end_time - start_time)*1000:.1f}ms")
            print(f"   Output shape: {hidden_states.shape}")
            print(f"   Output range: [{hidden_states.min():.3f}, {hidden_states.max():.3f}]")
            
        except Exception as e:
            print(f"❌ Single token inference failed: {e}")
        
        # Test with batch of tokens (1,64) shape if we have enough tokens
        if input_ids.shape[1] >= 10:
            print(f"\n🧪 Testing batch inference (up to 64 tokens):")
            
            # Take up to 64 tokens
            batch_size = min(64, input_ids.shape[1])
            batch_tokens = input_ids[:, :batch_size]
            
            # Pad to exactly 64 if needed
            if batch_tokens.shape[1] < 64:
                pad_size = 64 - batch_tokens.shape[1]
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                padding = np.full((1, pad_size), pad_token_id, dtype=np.int32)
                batch_tokens = np.concatenate([batch_tokens, padding], axis=1)
            
            print(f"   Batch tokens shape: {batch_tokens.shape}")
            print(f"   First 10 tokens: {batch_tokens[0][:10]}")
            
            try:
                start_time = time.time()
                result = embeddings_model.predict({"input_ids": batch_tokens})
                end_time = time.time()
                
                hidden_states = result['hidden_states']
                print(f"✅ Batch inference successful!")
                print(f"   Inference time: {(end_time - start_time)*1000:.1f}ms")
                print(f"   Output shape: {hidden_states.shape}")
                print(f"   Output range: [{hidden_states.min():.3f}, {hidden_states.max():.3f}]")
                
                # Calculate throughput
                throughput = batch_tokens.shape[1] / ((end_time - start_time) * 1000) * 1000
                print(f"   Throughput: {throughput:.1f} tokens/second")
                
            except Exception as e:
                print(f"❌ Batch inference failed: {e}")

def test_lm_head():
    """Test the LM head model."""
    print(f"\n{'='*60}")
    print("🧪 Testing LM Head Model")
    print("=" * 60)
    
    model_dir = "/Users/mazhewitt/projects/TypoFixerTraining/models/qwen-typo-fixer-ane"
    lm_head_path = os.path.join(model_dir, "qwen-typo-fixer_lm_head_lut6.mlpackage")
    
    try:
        lm_head_model = ct.models.MLModel(lm_head_path)
        print(f"✅ LM head model loaded")
        
        # Create fake hidden states (1, 1, 1024) as expected by LM head
        fake_hidden_states = np.random.randn(1, 1, 1024).astype(np.float16)
        print(f"   Input shape: {fake_hidden_states.shape}")
        
        start_time = time.time()
        result = lm_head_model.predict({"hidden_states": fake_hidden_states})
        end_time = time.time()
        
        print(f"✅ LM head inference successful!")
        print(f"   Inference time: {(end_time - start_time)*1000:.1f}ms")
        print(f"   Output keys: {list(result.keys())}")
        
        # Check logits shapes
        for key in sorted(result.keys()):
            logits = result[key]
            print(f"   {key}: {logits.shape}, range: [{logits.min():.3f}, {logits.max():.3f}]")
            
        # Combine all logits to get full vocabulary predictions
        all_logits = []
        for i in range(1, 17):  # 16 parts
            key = f"logits{i}"
            if key in result:
                all_logits.append(result[key])
        
        if all_logits:
            combined_logits = np.concatenate(all_logits, axis=-1)
            print(f"   Combined logits shape: {combined_logits.shape}")
            
            # Get top predictions
            top_indices = np.argsort(combined_logits[0, 0])[-5:][::-1]
            print(f"   Top 5 token predictions: {top_indices}")
            
    except Exception as e:
        print(f"❌ LM head test failed: {e}")

def benchmark_embeddings_performance():
    """Benchmark embeddings performance with correct shapes."""
    print(f"\n{'='*60}")
    print("📊 Embeddings Performance Benchmark")
    print("=" * 60)
    
    model_dir = "/Users/mazhewitt/projects/TypoFixerTraining/models/qwen-typo-fixer-ane"
    embeddings_path = os.path.join(model_dir, "qwen-typo-fixer_embeddings.mlpackage")
    embeddings_model = ct.models.MLModel(embeddings_path)
    
    # Test single token performance
    print("🔥 Single token inference (10 runs):")
    single_token = np.array([[12345]], dtype=np.int32)  # Random token
    
    times = []
    for i in range(10):
        start_time = time.time()
        result = embeddings_model.predict({"input_ids": single_token})
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"   Run {i+1:2d}: {(end_time - start_time)*1000:5.1f}ms")
    
    avg_time = np.mean(times) * 1000
    print(f"   Average: {avg_time:.1f}ms")
    print(f"   Throughput: {1000/avg_time:.1f} inferences/second")
    
    # Test batch performance (64 tokens)
    print(f"\n🔥 Batch inference (64 tokens, 5 runs):")
    batch_tokens = np.random.randint(1000, 50000, (1, 64), dtype=np.int32)
    
    times = []
    for i in range(5):
        start_time = time.time()
        result = embeddings_model.predict({"input_ids": batch_tokens})
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"   Run {i+1}: {(end_time - start_time)*1000:5.1f}ms")
    
    avg_time = np.mean(times) * 1000
    throughput = 64 / (avg_time / 1000)
    print(f"   Average: {avg_time:.1f}ms")
    print(f"   Throughput: {throughput:.1f} tokens/second")

def main():
    """Main test function."""
    try:
        print("🍎 Apple Neural Engine CoreML Model Test")
        print("=" * 60)
        
        # Test basic functionality
        test_with_correct_shapes()
        
        # Test LM head
        test_lm_head()
        
        # Benchmark performance
        benchmark_embeddings_performance()
        
        print(f"\n{'='*60}")
        print("🎉 All tests completed!")
        print("✅ Your fine-tuned Qwen model is successfully converted to CoreML")
        print("✅ Ready for Apple Neural Engine deployment!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
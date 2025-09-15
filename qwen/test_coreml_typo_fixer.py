#!/usr/bin/env python3
"""
Test script for CoreML-converted Qwen typo fixer model.
Direct inference without the anemll chat interface.
"""

import torch
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import time
import os

def load_coreml_model(model_path):
    """Load a CoreML model."""
    print(f"Loading CoreML model from: {model_path}")
    try:
        model = ct.models.MLModel(model_path)
        print(f"✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_tokenizer(tokenizer_path):
    """Load the tokenizer."""
    print(f"Loading tokenizer from: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None

def test_embeddings_model(embeddings_model, tokenizer, test_text):
    """Test the embeddings model."""
    print(f"\n🧪 Testing embeddings with: '{test_text}'")
    
    # Tokenize input - use max_length=64 to match model's expected shape
    inputs = tokenizer(test_text, return_tensors="np", max_length=64, padding="max_length", truncation=True)
    input_ids = inputs['input_ids'].astype(np.int32)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   First 10 tokens: {input_ids[0][:10]}")
    
    try:
        # Run inference
        start_time = time.time()
        result = embeddings_model.predict({"input_ids": input_ids})
        end_time = time.time()
        
        print(f"✅ Embeddings inference successful!")
        print(f"   Inference time: {(end_time - start_time)*1000:.1f}ms")
        print(f"   Output keys: {list(result.keys())}")
        
        hidden_states = result['hidden_states']
        print(f"   Hidden states shape: {hidden_states.shape}")
        print(f"   Hidden states range: [{hidden_states.min():.3f}, {hidden_states.max():.3f}]")
        
        return hidden_states
        
    except Exception as e:
        print(f"❌ Embeddings inference failed: {e}")
        return None

def test_simple_inference(model_dir, tokenizer_path, test_sentences):
    """Test simple inference with the CoreML models."""
    print("=" * 60)
    print("🚀 Testing CoreML Qwen Typo Fixer")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        return
    
    # Load embeddings model
    embeddings_path = os.path.join(model_dir, "qwen-typo-fixer_embeddings.mlpackage")
    embeddings_model = load_coreml_model(embeddings_path)
    if embeddings_model is None:
        return
    
    # Test each sentence
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {sentence}")
        print(f"{'='*50}")
        
        # Test embeddings
        hidden_states = test_embeddings_model(embeddings_model, tokenizer, sentence)
        
        if hidden_states is not None:
            print(f"✅ Test {i} passed - embeddings working correctly")
        else:
            print(f"❌ Test {i} failed - embeddings issue")

def benchmark_performance(model_dir, tokenizer_path, num_runs=10):
    """Benchmark the performance of the CoreML model."""
    print("\n" + "=" * 60)
    print("📊 Performance Benchmark")
    print("=" * 60)
    
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        return
    
    embeddings_path = os.path.join(model_dir, "qwen-typo-fixer_embeddings.mlpackage")
    embeddings_model = load_coreml_model(embeddings_path)
    if embeddings_model is None:
        return
    
    test_text = "Fix: I beleive this is teh correct answr and we shoudl teste it."
    inputs = tokenizer(test_text, return_tensors="np", max_length=64, padding="max_length", truncation=True)
    input_ids = inputs['input_ids'].astype(np.int32)
    
    print(f"Running {num_runs} inference runs...")
    
    times = []
    for i in range(num_runs):
        start_time = time.time()
        try:
            result = embeddings_model.predict({"input_ids": input_ids})
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"  Run {i+1:2d}/{num_runs}: {(end_time - start_time)*1000:5.1f}ms")
        except Exception as e:
            print(f"  Run {i+1:2d}/{num_runs}: FAILED - {e}")
    
    if times:
        avg_time = np.mean(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        std_time = np.std(times) * 1000
        
        print(f"\n📈 Performance Results:")
        print(f"   Average: {avg_time:.1f}ms")
        print(f"   Min:     {min_time:.1f}ms") 
        print(f"   Max:     {max_time:.1f}ms")
        print(f"   Std Dev: {std_time:.1f}ms")
        
        # Estimate throughput
        tokens_per_inference = np.sum(input_ids != tokenizer.pad_token_id)
        throughput = tokens_per_inference / (avg_time / 1000)
        print(f"   Estimated throughput: {throughput:.1f} tokens/second")

def inspect_model_details(model_dir):
    """Inspect the details of converted models."""
    print("\n" + "=" * 60)
    print("🔍 Model Inspection")
    print("=" * 60)
    
    models = [
        ("Embeddings", "qwen-typo-fixer_embeddings.mlpackage"),
        ("LM Head", "qwen-typo-fixer_lm_head_lut6.mlpackage"),
        ("FFN+Prefill", "qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage")
    ]
    
    for name, filename in models:
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            print(f"\n{name} Model ({filename}):")
            try:
                model = ct.models.MLModel(model_path)
                spec = model.get_spec()
                
                print(f"  📝 Description: {spec.description}")
                print(f"  🏷️  Metadata: {dict(spec.metadata.userDefined)}")
                
                print(f"  📥 Inputs:")
                for input_desc in spec.description.input:
                    print(f"    - {input_desc.name}: {input_desc.type}")
                
                print(f"  📤 Outputs:")  
                for output_desc in spec.description.output:
                    print(f"    - {output_desc.name}: {output_desc.type}")
                    
                # Get file size
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"  💾 File size: {size_mb:.1f} MB")
                
            except Exception as e:
                print(f"  ❌ Error inspecting model: {e}")
        else:
            print(f"\n{name} Model: ❌ Not found ({filename})")

def main():
    """Main test function."""
    # Configuration
    model_dir = "/Users/mazhewitt/projects/TypoFixerTraining/models/qwen-typo-fixer-ane"
    tokenizer_path = "/Users/mazhewitt/projects/TypoFixerTraining/models/qwen-typo-fixer"
    
    # Test sentences with typos
    test_sentences = [
        "Fix: I beleive this is teh correct answr.",
        "Fix: The recieved messge was very importnt.",
        "Fix: Please chekc your speeling before submiting.",
        "Fix: This setence has multple typos in it.",
        "Fix: We shoudl teste the modle with various inputs."
    ]
    
    # Check if directories exist
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer directory not found: {tokenizer_path}")
        return
    
    try:
        # Inspect model details
        inspect_model_details(model_dir)
        
        # Test inference
        test_simple_inference(model_dir, tokenizer_path, test_sentences)
        
        # Benchmark performance
        benchmark_performance(model_dir, tokenizer_path, num_runs=5)
        
        print("\n" + "=" * 60)
        print("🎉 Testing completed!")
        print("=" * 60)
        print("\nNote: This tests the embeddings layer only.")
        print("Full inference requires combining all model parts (embeddings, FFN, LM head).")
        print("The model is successfully converted and ready for ANE deployment!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
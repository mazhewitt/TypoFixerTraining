#!/usr/bin/env python3
"""
Analyze T5-efficient-tiny model capacity and optimization opportunities.
"""

import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path

def analyze_model_capacity():
    """Analyze the trained model's capacity and potential for improvement."""
    
    print("🔍 Analyzing T5-efficient-tiny Model Capacity")
    print("=" * 50)
    
    # Load the trained model
    model_path = "models/t5-typo-fixer"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"📥 Loading trained model from {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Model architecture analysis
    print(f"\n🏗️ Model Architecture Analysis:")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Layer breakdown
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters()) 
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    
    print(f"   Encoder parameters: {encoder_params:,}")
    print(f"   Decoder parameters: {decoder_params:,}")
    print(f"   LM Head parameters: {lm_head_params:,}")
    
    # Model configuration
    config = model.config
    print(f"\n⚙️ Model Configuration:")
    print(f"   Model size: {config.d_model}")
    print(f"   Feed-forward size: {config.d_ff}")
    print(f"   Attention heads: {config.num_heads}")
    print(f"   Encoder layers: {config.num_layers}")
    print(f"   Decoder layers: {config.num_decoder_layers}")
    print(f"   Vocabulary size: {config.vocab_size}")
    
    # Check training info
    training_info_path = f"{model_path}/training_info.json"
    if Path(training_info_path).exists():
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)
        
        print(f"\n📊 Training Results:")
        print(f"   Training time: {training_info.get('training_time_minutes', 0):.1f} minutes")
        print(f"   Final accuracy: {training_info.get('final_accuracy', 0)*100:.1f}%")
        print(f"   Training examples: {training_info.get('training_examples', 0):,}")
        print(f"   Effective batch size: {training_info.get('batch_size', 0)}")
        print(f"   Total steps: {training_info.get('total_steps', 0):,}")
    
    # GPU utilization analysis
    print(f"\n🖥️ GPU Utilization Analysis:")
    print(f"   Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    
    if torch.backends.mps.is_available():
        # Test memory usage with different batch sizes
        test_inputs = tokenizer("correct typos: I beleive this is a test.", return_tensors="pt")
        
        print(f"   Current model memory footprint:")
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        print(f"     Model parameters: {model_size_mb:.1f} MB")
        
        # Test batch processing capability
        batch_sizes = [1, 4, 8, 16, 32]
        max_working_batch = 1
        
        for batch_size in batch_sizes:
            try:
                # Create batch
                batch_inputs = {
                    'input_ids': test_inputs['input_ids'].repeat(batch_size, 1),
                    'attention_mask': test_inputs['attention_mask'].repeat(batch_size, 1)
                }
                
                # Test forward pass
                with torch.no_grad():
                    outputs = model.generate(**batch_inputs, max_length=64, num_beams=1)
                
                max_working_batch = batch_size
                print(f"     ✅ Batch size {batch_size}: Working")
                
            except Exception as e:
                print(f"     ❌ Batch size {batch_size}: Failed ({str(e)[:50]}...)")
                break
        
        print(f"   Maximum working batch size: {max_working_batch}")
    
    # Learning capacity assessment
    print(f"\n🧠 Learning Capacity Assessment:")
    
    # Calculate model complexity vs task complexity
    vocab_coverage = min(config.vocab_size, 32000)  # Typical vocabulary
    task_complexity = vocab_coverage * config.d_model
    model_capacity = encoder_params + decoder_params
    
    capacity_ratio = model_capacity / task_complexity
    print(f"   Model capacity ratio: {capacity_ratio:.3f}")
    
    if capacity_ratio < 0.1:
        capacity_assessment = "🔴 SEVERELY LIMITED - Model may be too small for complex patterns"
    elif capacity_ratio < 0.3:
        capacity_assessment = "🟡 LIMITED - Model has basic capacity but limited complexity"
    elif capacity_ratio < 0.7:
        capacity_assessment = "🟢 ADEQUATE - Good capacity for the task"
    else:
        capacity_assessment = "🔵 EXCELLENT - High capacity, can learn complex patterns"
    
    print(f"   Assessment: {capacity_assessment}")
    
    # Optimization recommendations
    print(f"\n💡 Optimization Recommendations:")
    
    print(f"   📈 GPU Utilization (currently ~50%):")
    print(f"     - Increase batch size from current to {min(max_working_batch, 32)}")
    print(f"     - Use gradient accumulation for effective larger batches")
    print(f"     - Enable mixed precision (fp16) if supported")
    
    print(f"   🎓 Training Improvements:")
    print(f"     - More epochs (current: 3) → try 5-10 epochs")
    print(f"     - Learning rate scheduling (cosine warmup)")
    print(f"     - Regularization (dropout, weight decay)")
    
    print(f"   🔄 Model Capacity:")
    if capacity_ratio < 0.3:
        print(f"     - Consider T5-small (77M params) for better capacity")
        print(f"     - Current T5-tiny (15.6M) may be hitting fundamental limits")
    else:
        print(f"     - Current model size adequate")
        print(f"     - Focus on training optimization rather than model size")
    
    # Test current model on a few examples
    print(f"\n🧪 Quick Model Test:")
    test_cases = [
        "I beleive this is correct",
        "The qucik brown fox",
        "She recieved her degre"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        prompt = f"correct typos: {test_input}"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64, num_beams=1, do_sample=False)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   {i}. '{test_input}' → '{result}'")
    
    return {
        'total_params': sum(p.numel() for p in model.parameters()),
        'capacity_ratio': capacity_ratio,
        'max_batch_size': max_working_batch,
        'recommendations': {
            'increase_batch_size': True,
            'more_epochs': True,
            'better_model': capacity_ratio < 0.3
        }
    }

if __name__ == "__main__":
    analysis = analyze_model_capacity()
    
    # Save analysis results
    with open('model_capacity_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n💾 Analysis saved to model_capacity_analysis.json")
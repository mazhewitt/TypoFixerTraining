#!/usr/bin/env python3
"""
Quick test script to verify T5 setup before training.
"""

import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

def test_t5_setup():
    print("🧪 Testing T5-efficient-tiny setup...")
    
    # Load model and tokenizer
    print("📥 Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('google/t5-efficient-tiny', legacy=False)
    model = T5ForConditionalGeneration.from_pretrained('google/t5-efficient-tiny')
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test data loading
    print("\n📊 Testing data loading...")
    data_file = "data/enhanced_training_full.jsonl"
    
    try:
        with open(data_file, 'r') as f:
            sample_data = json.loads(f.readline().strip())
        
        print(f"✅ Data file found: {data_file}")
        print(f"📝 Sample data: {sample_data}")
        
        # Test T5 formatting
        source_text = f"correct typos: {sample_data['corrupted']}"
        target_text = sample_data['clean']
        
        print(f"\n🔄 T5 format test:")
        print(f"   Source: {source_text}")
        print(f"   Target: {target_text}")
        
        # Test tokenization
        source_tokens = tokenizer(source_text, return_tensors='pt')
        target_tokens = tokenizer(target_text, return_tensors='pt')
        
        print(f"✅ Source tokens: {source_tokens['input_ids'].shape}")
        print(f"✅ Target tokens: {target_tokens['input_ids'].shape}")
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_file}")
        return False
    
    # Test inference (before training)
    print("\n🚀 Testing pre-training inference...")
    test_input = "correct typos: I beleive this is teh answr."
    inputs = tokenizer(test_input, return_tensors='pt')
    
    try:
        outputs = model.generate(
            **inputs, 
            max_length=64, 
            num_beams=1, 
            do_sample=False,
            early_stopping=True
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"📝 Input: {test_input}")
        print(f"📝 Output: '{result}' (expected to be empty/random before training)")
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False
    
    print("\n✅ All tests passed! Ready for training.")
    return True

if __name__ == "__main__":
    test_t5_setup()
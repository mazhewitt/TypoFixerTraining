#!/usr/bin/env python3
"""
Test script to compare base model vs fine-tuned model performance
and verify what CoreML models we actually need.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_model(model_path, model_name, test_sentences):
    """Test a model on given sentences."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"✅ {model_name} loaded successfully")
        
        # Test each sentence
        for i, sentence in enumerate(test_sentences, 1):
            prompt = f"Fix: {sentence}"
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
            
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
            
            generated = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            if '.' in generated:
                corrected = generated.split('.')[0].strip() + '.'
            else:
                corrected = generated.strip()
            
            print(f"  {i}. Input:  {sentence}")
            print(f"     Output: {corrected}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return False

def main():
    """Compare base vs fine-tuned models."""
    test_sentences = [
        "I beleive this is teh correct answr.",
        "She recieved her degre yesterday.", 
        "The resturant serves excelent food."
    ]
    
    print("🎯 Model Comparison Test")
    print("This will help us understand what CoreML models we need to generate.")
    
    # Test base model (local)
    base_success = test_model(
        "models/qwen-0.6b", 
        "Base Qwen-0.6B (Local)", 
        test_sentences
    )
    
    # Test fine-tuned model (HuggingFace)
    finetuned_success = test_model(
        "mazhewitt/qwen-typo-fixer",
        "Fine-tuned Qwen Typo Fixer (HF)",
        test_sentences
    )
    
    print(f"\n{'='*60}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Base model working: {'✅' if base_success else '❌'}")
    print(f"Fine-tuned model working: {'✅' if finetuned_success else '❌'}")
    
    if base_success and finetuned_success:
        print("\n🎯 CONCLUSION:")
        print("- Both models work, but fine-tuned should perform better")
        print("- Current CoreML models in /models/qwen-ane-test/ are from BASE model")
        print("- We need to convert FINE-TUNED model to CoreML for optimal performance")
        print("- Current typo_fixer_complete.py uses base model CoreML + fine-tuned tokenizer (mismatch)")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
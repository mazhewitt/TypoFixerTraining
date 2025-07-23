#!/usr/bin/env python3
"""
Diagnose why the explicit MLM works in isolation but fails in the pipeline.
Compare individual predictions vs full pipeline corrections.
"""
import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from final_explicit_corrector import FinalExplicitCorrector

def test_individual_predictions():
    """Test the explicit MLM on individual typos."""
    print("🔬 TESTING INDIVIDUAL MLM PREDICTIONS")
    print("=" * 50)
    
    # Load model directly
    model_path = "models/explicit_typo_mlm"
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForMaskedLM.from_pretrained(model_path)
    model.eval()
    
    # Test cases from benchmark failures
    test_cases = [
        ("Thi is a test", "thi", "This is a test"),
        ("The quikc brown fox", "quikc", "The quick brown fox"),
        ("Can you halp me", "halp", "Can you help me"),
        ("She recieved a gift", "recieved", "She received a gift"),
        ("Its importnt to check", "importnt", "Its important to check"),
    ]
    
    for sentence, typo, expected in test_cases:
        # Create explicit format
        masked_text = sentence.replace(typo, '[MASK]', 1)
        explicit_input = f"CORRUPT: {typo} SENTENCE: {masked_text}"
        
        print(f"\nTesting: {sentence}")
        print(f"Typo: {typo}")
        print(f"Explicit format: {explicit_input}")
        
        # Get prediction
        inputs = tokenizer(explicit_input, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Find [MASK] token position
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
        
        if len(mask_positions) > 0:
            mask_pos = mask_positions[0]
            mask_logits = logits[0, mask_pos]
            top_predictions = torch.topk(mask_logits, 5)
            
            print("Top predictions:")
            for i, (score, token_id) in enumerate(zip(top_predictions.values, top_predictions.indices)):
                word = tokenizer.decode(token_id).strip()
                confidence = torch.softmax(mask_logits, dim=0)[token_id].item()
                print(f"  {i+1}. {word} ({confidence*100:.1f}%)")
                
            # Check if top prediction matches expected correction
            top_word = tokenizer.decode(top_predictions.indices[0]).strip()
            expected_word = expected.split()[sentence.split().index(typo)]
            
            if top_word.lower() == expected_word.lower():
                print(f"✅ CORRECT: {top_word}")
            else:
                print(f"❌ WRONG: got '{top_word}', expected '{expected_word}'")
        else:
            print("❌ No [MASK] token found!")

def test_pipeline_vs_individual():
    """Compare pipeline results with individual MLM predictions."""
    print("\n🔬 COMPARING PIPELINE VS INDIVIDUAL PREDICTIONS")
    print("=" * 50)
    
    corrector = FinalExplicitCorrector()
    
    test_sentence = "Thi sis a test sentenc with multipl typos"
    print(f"Test sentence: {test_sentence}")
    
    # Get pipeline result
    pipeline_result, metadata = corrector.correct_text(test_sentence)
    print(f"Pipeline result: {pipeline_result}")
    print(f"Corrections made: {metadata.get('corrections_made', [])}")
    
    # Test each detected typo individually
    detected_typos = ['thi', 'sis', 'sentenc', 'multipl']
    
    for typo in detected_typos:
        if typo in test_sentence.lower():
            print(f"\n--- Testing individual typo: {typo} ---")
            
            # Test with corrector's method
            candidates = corrector.correct_word_with_explicit_mlm(test_sentence, typo, top_k=3)
            print(f"Corrector candidates: {candidates}")
            
            # Test direct MLM
            masked_text = test_sentence.replace(typo, '[MASK]', 1)
            explicit_input = f"CORRUPT: {typo} SENTENCE: {masked_text}"
            
            inputs = corrector.tokenizer(explicit_input, return_tensors="pt", truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = corrector.model(**inputs)
                logits = outputs.logits
            
            mask_positions = (inputs['input_ids'] == corrector.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            
            if len(mask_positions) > 0:
                mask_pos = mask_positions[0]
                mask_logits = logits[0, mask_pos]
                top_predictions = torch.topk(mask_logits, 3)
                
                print("Direct MLM predictions:")
                for i, (score, token_id) in enumerate(zip(top_predictions.values, top_predictions.indices)):
                    word = corrector.tokenizer.decode(token_id).strip()
                    confidence = torch.softmax(mask_logits, dim=0)[token_id].item()
                    print(f"  {i+1}. {word} ({confidence*100:.1f}%)")

def test_tokenization_issues():
    """Check if tokenization is causing problems."""
    print("\n🔬 TESTING TOKENIZATION ISSUES")
    print("=" * 50)
    
    tokenizer = DistilBertTokenizer.from_pretrained("models/explicit_typo_mlm")
    
    test_cases = [
        "CORRUPT: thi SENTENCE: [MASK] is a test",
        "CORRUPT: quikc SENTENCE: The [MASK] brown fox",
        "CORRUPT: halp SENTENCE: Can you [MASK] me",
    ]
    
    for text in test_cases:
        print(f"\nText: {text}")
        tokens = tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        
        # Check if [MASK] is properly tokenized
        if '[MASK]' in tokens:
            mask_idx = tokens.index('[MASK]')
            print(f"[MASK] position in tokens: {mask_idx}")
        else:
            print("❌ [MASK] not found in tokens!")
        
        # Check encoding/decoding
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")

if __name__ == "__main__":
    test_individual_predictions()
    test_pipeline_vs_individual()
    test_tokenization_issues()
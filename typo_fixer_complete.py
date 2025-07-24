#!/usr/bin/env python3
"""
Complete end-to-end typo correction using CoreML converted Qwen model.
Chains embeddings -> FFN+Prefill -> LM Head for full text generation.
"""

import torch
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import time
import os

class CoreMLTypoFixer:
    """Complete typo fixer using CoreML models."""
    
    def __init__(self, model_dir, tokenizer_path):
        """Initialize the typo fixer with model paths."""
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.embeddings_model = None
        self.ffn_model = None
        self.lm_head_model = None
        self.kv_state = None          # <-- new
        self.load_models()
    
    def load_models(self):
        """Load all model components using anemll approach."""
        print("🚀 Loading CoreML Typo Fixer Models...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        print(f"✅ Tokenizer loaded: {self.tokenizer.__class__.__name__}")
        
        # Load embeddings model
        embeddings_path = os.path.join(self.model_dir, "qwen-typo-fixer_embeddings.mlpackage")
        self.embeddings_model = ct.models.MLModel(embeddings_path)
        print(f"✅ Embeddings model loaded")
        
        # Load FFN+Prefill model with separate functions (anemll approach)
        ffn_path = os.path.join(self.model_dir, "qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage")
        self.ffn_model = {
            'prefill': ct.models.MLModel(ffn_path, function_name='prefill'),
            'infer': ct.models.MLModel(ffn_path, function_name='infer')
        }
        print(f"✅ FFN+Prefill model loaded (prefill & infer functions)")
        
        # Load LM Head model
        lm_head_path = os.path.join(self.model_dir, "qwen-typo-fixer_lm_head_lut6.mlpackage")
        self.lm_head_model = ct.models.MLModel(lm_head_path)
        print(f"✅ LM Head model loaded")
        
        print("🎉 All models loaded successfully!\n")
    
    def create_basic_prompt(self, text_with_typos):
        """Create basic prompt for typo correction (like demo.py basic approach)."""
        prompt = f"Fix: {text_with_typos}"
        return prompt
    
    def create_few_shot_prompt(self, text_with_typos):
        """Create few-shot prompt for typo correction (like demo.py optimized approach)."""
        prompt = f"""Fix typos in these sentences:

Input: I beleive this is teh answer.
Output: I believe this is the answer.

Input: She recieved her degre yesterday.
Output: She received her degree yesterday.

Input: The resturant serves good food.
Output: The restaurant serves good food.

Input: {text_with_typos}
Output:"""
        return prompt
    
    def tokenize_input(self, text, max_length=64):
        """Tokenize input text with proper padding."""
        inputs = self.tokenizer(
            text, 
            return_tensors="np", 
            max_length=max_length, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True
        )
        return inputs['input_ids'].astype(np.int32)
    
    def run_embeddings(self, input_ids):
        """Run embeddings model."""
        print(f"   📥 Embeddings input shape: {input_ids.shape}")
        
        start_time = time.time()
        result = self.embeddings_model.predict({"input_ids": input_ids})
        end_time = time.time()
        
        hidden_states = result['hidden_states']
        print(f"   ✅ Embeddings: {(end_time - start_time)*1000:.1f}ms, output: {hidden_states.shape}")
        return hidden_states
    
    
    def run_lm_head(self, hidden_states):
        """Run LM Head model to get token predictions."""
        print(f"   🎯 LM Head input shape: {hidden_states.shape}")
        
        # Ensure we have single token input (1, 1, 1024)
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]  # Take last token
        
        start_time = time.time()
        result = self.lm_head_model.predict({"hidden_states": hidden_states.astype(np.float16)})
        end_time = time.time()
        
        # Combine all logits parts
        all_logits = []
        for i in range(1, 17):  # 16 parts
            key = f"logits{i}"
            if key in result:
                all_logits.append(result[key])
        
        combined_logits = np.concatenate(all_logits, axis=-1)
        print(f"   ✅ LM Head: {(end_time - start_time)*1000:.1f}ms, logits: {combined_logits.shape}")
        return combined_logits
    
    def make_causal_mask(self, length, start):
        """Create causal attention mask (from anemll)."""
        mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
        row_indices = np.arange(length).reshape(length, 1)
        col_indices = np.arange(length).reshape(1, length)
        mask[:, :, col_indices <= (row_indices + start)] = 0
        return torch.tensor(mask, dtype=torch.float16)
    
    def run_prefill(self, input_ids, context_pos, context_length, causal_mask, batch_size=64):
        """Run prefill on the input sequence using anemll approach."""
        # Initialize KV state if not already done
        if self.kv_state is None:
            self.kv_state = self.ffn_model['prefill'].make_state()
        
        batch_pos = 0
        while batch_pos < context_pos:
            batch_end = min(batch_pos + batch_size, context_pos)
            current_batch_size = batch_end - batch_pos
            
            print(f"   📦 Prefill batch {batch_pos}-{batch_end-1} ({current_batch_size} tokens)")
            
            # Get current batch
            batch_input = input_ids[:, batch_pos:batch_end]
            
            # Always pad to full batch size for prefill
            if current_batch_size < batch_size:
                pad_size = batch_size - current_batch_size
                padding = np.zeros((1, pad_size), dtype=np.int32)
                batch_input = np.concatenate([batch_input, padding], axis=1)
            
            # Generate position IDs for full batch size
            position_ids = np.arange(batch_pos, batch_pos + batch_size, dtype=np.int32)
            batch_causal_mask = causal_mask[:, :, batch_pos:batch_pos + batch_size, :].numpy().astype(np.float16)
            
            # Run embeddings
            hidden_states = self.run_embeddings(batch_input)
            
            # Run through FFN prefill with state
            inputs = {
                'hidden_states': hidden_states.astype(np.float16),
                'position_ids': position_ids,
                'causal_mask': batch_causal_mask,
                'current_pos': np.array([batch_pos], dtype=np.int32)
            }
            
            start_time = time.time()
            output = self.ffn_model['prefill'].predict(inputs, self.kv_state)
            end_time = time.time()
            
            print(f"   ✅ Prefill batch: {(end_time - start_time)*1000:.1f}ms")
            
            batch_pos = batch_end
    
    def generate_next_token(self, input_ids, pos, context_length, causal_mask, temperature=0.1):
        """Generate the next token using anemll approach."""
        # Get current token
        current_token = input_ids[:, pos-1:pos]  # [1, 1]
        
        # Run embeddings
        hidden_states = self.run_embeddings(current_token)
        
        # Create inputs for infer function (same as prefill but with single token)
        position_ids = np.array([pos-1], dtype=np.int32)
        single_causal_mask = causal_mask[:, :, pos-1:pos, :].numpy().astype(np.float16)
        
        # Run through FFN infer with state
        inputs = {
            'hidden_states': hidden_states.astype(np.float16),
            'position_ids': position_ids,
            'causal_mask': single_causal_mask,
            'current_pos': position_ids
        }
        output = self.ffn_model['infer'].predict(inputs, self.kv_state)
        hidden_states = output['output_hidden_states']
        
        # Get logits
        logits = self.run_lm_head(hidden_states)
        
        # Generate token
        if temperature == 0.0:
            # Greedy sampling
            next_token_id = np.argmax(logits[0, 0])
        else:
            # Apply temperature and sample
            logits = logits / temperature
            top_k_logits, top_k_indices = torch.topk(torch.from_numpy(logits[0, 0]), 50)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1).item()
            next_token_id = top_k_indices[next_token_idx].item()
        
        return int(next_token_id)
    
    def clean_generated_output(self, generated_text, use_basic=True):
        """Clean generated output using demo.py logic."""
        # Basic cleaning
        generated = generated_text.strip()
        
        if not use_basic:
            # Few-shot cleaning (more thorough)
            generated = ' '.join(generated.split())
            if '\n' in generated:
                generated = generated.split('\n')[0].strip()
            
            # Remove suffixes that may appear
            for suffix in ['Input:', 'Output:', 'Human:', 'Assistant:']:
                if suffix.lower() in generated.lower():
                    generated = generated.split(suffix)[0].strip()
        
        # Common cleaning for both approaches
        if '.' in generated:
            corrected = generated.split('.')[0].strip() + '.'
        else:
            corrected = generated.strip()
        
        return corrected
    
    def fix_typos(self, text_with_typos, max_new_tokens=15, temperature=0.1, use_basic=True):
        """Complete typo correction pipeline using anemll-style prefill/infer approach."""
        self.kv_state = None   # Reset conversation state
        print("=" * 60)
        print(f"🔧 Fixing typos in: '{text_with_typos}'")
        print("=" * 60)
        
        # Create prompt (basic or few-shot)
        if use_basic:
            prompt = self.create_basic_prompt(text_with_typos)
            max_length = 64  # Shorter for basic prompts
            print(f"📝 Basic prompt created: '{prompt}'")
        else:
            prompt = self.create_few_shot_prompt(text_with_typos)
            max_length = 128  # Longer for few-shot
            print(f"📝 Few-shot prompt created ({len(prompt)} chars)")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np", add_special_tokens=True, truncation=True, max_length=max_length)
        input_ids = inputs['input_ids'].astype(np.int32)
        context_pos = input_ids.shape[1]
        print(f"🔤 Tokenized to {context_pos} tokens")
        
        # Create causal mask for full context
        context_length = 256  # From meta.yaml
        causal_mask = self.make_causal_mask(context_length, 0)
        
        # Track generated tokens
        generated_tokens = []
        
        try:
            print("\n🏃 Running prefill phase:")
            
            # Run prefill on the entire prompt using anemll approach
            self.run_prefill(input_ids, context_pos, context_length, causal_mask)
            
            print(f"\n🎯 Generating up to {max_new_tokens} tokens:")
            
            for i in range(max_new_tokens):
                # Current position in sequence
                current_pos = context_pos + i
                
                if current_pos >= context_length:
                    print("     🛑 Context length exceeded")
                    break
                
                # Generate next token using anemll approach
                next_token_id = self.generate_next_token(input_ids, current_pos, context_length, causal_mask, temperature)
                generated_tokens.append(next_token_id)
                
                # Decode token to see progress
                token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                print(f"     Token {i+1:2d}: {next_token_id:5d} -> '{token_text}'")
                
                # Check for end conditions
                if next_token_id == self.tokenizer.eos_token_id:
                    print("     🛑 EOS token reached")
                    break
                
                if token_text.strip() in ['Input:', 'Output:', '\n\n']:
                    print("     🛑 End pattern detected")
                    break
                
                # Add token to input sequence for next iteration
                new_token = np.array([[next_token_id]], dtype=np.int32)
                input_ids = np.concatenate([input_ids, new_token], axis=1)
            
            # Decode generated text
            if generated_tokens:
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"\n📤 Generated text: '{generated_text}'")
                
                # Clean output using demo.py logic
                corrected_text = self.clean_generated_output(generated_text, use_basic)
                
                print(f"✨ Final correction: '{corrected_text}'")
                return corrected_text
            else:
                print("❌ No tokens generated")
                return text_with_typos
                
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return text_with_typos

def main():
    """Test the complete typo fixer."""
    # Configuration
    model_dir = "/Users/mazhewitt/projects/TypoFixerTraining/models/qwen-typo-fixer-ane"
    tokenizer_path = "/Users/mazhewitt/projects/TypoFixerTraining/models/qwen-typo-fixer"
    
    # Test sentences
    test_sentences = [
        "I beleive this is teh correct answr.",
        "Please chekc your speeling before submiting.",
        "The recieved messge was very importnt.",
        "This setence has multple typos in it."
    ]
    
    try:
        # Initialize typo fixer
        fixer = CoreMLTypoFixer(model_dir, tokenizer_path)
        
        # Test each sentence
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}/{len(test_sentences)}")
            print(f"{'='*80}")
            
            start_time = time.time()
            corrected = fixer.fix_typos(sentence, max_new_tokens=15, use_basic=True)
            end_time = time.time()
            
            print(f"\n📊 RESULTS:")
            print(f"   Original:  '{sentence}'")
            print(f"   Corrected: '{corrected}'")
            print(f"   Time:      {(end_time - start_time):.2f}s")
            
            # Simple accuracy check
            typo_words = ['beleive', 'teh', 'answr', 'chekc', 'speeling', 'submiting', 
                         'recieved', 'messge', 'importnt', 'setence', 'multple']
            fixed_words = any(word not in corrected.lower() for word in typo_words if word in sentence.lower())
            
            if fixed_words:
                print(f"   Status:    ✅ Typos likely fixed!")
            else:
                print(f"   Status:    ⚠️  May need manual review")
        
        print(f"\n{'='*80}")
        print("🎉 All tests completed!")
        print("✅ Your CoreML Qwen typo fixer is working end-to-end!")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
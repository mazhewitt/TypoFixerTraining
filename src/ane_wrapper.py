#!/usr/bin/env python3
"""
Anemll-inspired ANE wrapper for DistilBERT typo correction.
Handles tensor format conversions and provides clean interface for ANE optimization.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add ANE models to path
sys.path.append(str(Path(__file__).parent))
from ane_models import DistilBertForMaskedLM as ANEDistilBertForMaskedLM
from ane_models import DistilBertConfig as ANEDistilBertConfig

class ANEDistilBertWrapper(nn.Module):
    """
    ANE-optimized DistilBERT wrapper following Anemll patterns.
    Handles all tensor format conversions internally to provide a clean interface.
    """
    
    def __init__(self, ane_model):
        super().__init__()
        self.model = ane_model
        self.config = ane_model.config
        
        # Set model to eval mode for ANE
        self.model.eval()
        
        # Freeze all parameters (ANE is inference-only)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _convert_inputs_to_ane_format(self, input_ids, attention_mask):
        """
        Convert standard tokenizer outputs to ANE-compatible format.
        Following Anemll pattern of internal tensor conversion.
        """
        batch_size, seq_len = input_ids.shape
        
        # For ANE DistilBERT, we need to handle the embedding conversion
        # The model expects input_ids and attention_mask in standard format,
        # but internally uses BC1S format for computations
        
        # Ensure correct data types
        input_ids = input_ids.to(torch.int32)
        attention_mask = attention_mask.to(torch.int32)
        
        return input_ids, attention_mask
    
    def _convert_outputs_from_ane_format(self, outputs):
        """
        Convert ANE model outputs back to standard format.
        """
        if isinstance(outputs, tuple):
            # ANE model returns tuple, extract logits
            logits = outputs[0]
        else:
            logits = outputs
        
        # Ensure logits are in the right format [batch, seq_len, vocab_size]
        if len(logits.shape) == 2:
            # Already in [batch, vocab_size] format - this is what we want for MLM
            pass
        elif len(logits.shape) == 4:
            # Need to squeeze dimensions from BC1S format
            logits = logits.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        
        return logits
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass with automatic tensor format conversion.
        
        Args:
            input_ids: [batch_size, seq_len] - standard tokenizer format
            attention_mask: [batch_size, seq_len] - standard tokenizer format
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] - standard MLM format
        """
        # Convert inputs to ANE format
        ane_input_ids, ane_attention_mask = self._convert_inputs_to_ane_format(input_ids, attention_mask)
        
        # Run ANE model (handles BC1S conversion internally)
        try:
            ane_outputs = self.model(
                input_ids=ane_input_ids,
                attention_mask=ane_attention_mask,
                return_dict=False  # ANE requires tuple output
            )
        except Exception as e:
            # If ANE format fails, provide detailed error info
            raise RuntimeError(f"ANE model forward pass failed: {e}\n"
                             f"Input shapes - input_ids: {ane_input_ids.shape}, "
                             f"attention_mask: {ane_attention_mask.shape}")
        
        # Convert outputs back to standard format
        logits = self._convert_outputs_from_ane_format(ane_outputs)
        
        return logits

class ANEDistilBertForInference(nn.Module):
    """
    Complete ANE DistilBERT inference pipeline following Anemll architecture patterns.
    This provides the cleanest interface for Core ML conversion.
    """
    
    def __init__(self, ane_model_path: str):
        super().__init__()
        
        # Load ANE model following our successful transfer approach
        self.ane_model = self._load_ane_model(ane_model_path)
        self.wrapper = ANEDistilBertWrapper(self.ane_model)
        
        # Store config for reference
        self.config = self.ane_model.config
        
    def _load_ane_model(self, model_path: str):
        """Load the ANE-optimized model."""
        import os
        import json
        from safetensors.torch import load_file
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create ANE config and model
        ane_config = ANEDistilBertConfig(**config_dict)
        model = ANEDistilBertForMaskedLM(ane_config)
        
        # Load converted weights
        state_dict_path = os.path.join(model_path, "model.safetensors")
        state_dict = load_file(state_dict_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        return model
    
    def forward(self, input_ids, attention_mask):
        """
        Clean forward interface for Core ML tracing.
        
        Args:
            input_ids: torch.Tensor [1, seq_len] - tokenized input
            attention_mask: torch.Tensor [1, seq_len] - attention mask
            
        Returns:
            logits: torch.Tensor [1, seq_len, vocab_size] - MLM predictions
        """
        return self.wrapper(input_ids, attention_mask)
    
    @torch.no_grad()
    def predict_text(self, input_text: str, tokenizer, max_length: int = 128):
        """
        High-level text prediction interface.
        
        Args:
            input_text: str - text with potential typos
            tokenizer: transformers tokenizer
            max_length: int - maximum sequence length
            
        Returns:
            predicted_text: str - corrected text
        """
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        
        # Get predictions
        logits = self.forward(inputs['input_ids'], inputs['attention_mask'])
        
        # Convert to predicted tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        
        return predicted_text.strip()

def create_zero_tensor_inputs(seq_len: int = 128, device: str = 'cpu'):
    """
    Create zero tensor inputs following Anemll pattern for model tracing.
    This avoids tokenizer format issues during torch.jit.trace().
    """
    input_ids = torch.zeros((1, seq_len), dtype=torch.int32, device=device)
    attention_mask = torch.ones((1, seq_len), dtype=torch.int32, device=device)
    
    return input_ids, attention_mask

def prepare_ane_model_for_tracing(ane_model_path: str, seq_len: int = 128):
    """
    Prepare ANE model for torch.jit.trace following Anemll best practices.
    
    Returns:
        model: ANEDistilBertForInference ready for tracing
        sample_inputs: tuple of zero tensors for tracing
    """
    # Create inference model
    model = ANEDistilBertForInference(ane_model_path)
    model.eval()
    
    # Create sample inputs for tracing (Anemll approach)
    sample_inputs = create_zero_tensor_inputs(seq_len)
    
    # Test forward pass to ensure everything works
    with torch.no_grad():
        try:
            test_output = model(*sample_inputs)
            print(f"✅ ANE model ready for tracing. Output shape: {test_output.shape}")
        except Exception as e:
            print(f"❌ ANE model test failed: {e}")
            raise
    
    return model, sample_inputs

if __name__ == "__main__":
    # Quick test of the wrapper
    print("Testing ANE wrapper...")
    
    try:
        model, inputs = prepare_ane_model_for_tracing('models/ane_typo_fixer', seq_len=128)
        print("✅ ANE wrapper created successfully!")
        
        # Test with real text
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('models/ane_typo_fixer')
        
        test_text = "Thi sis a test sentenc with typos"
        result = model.predict_text(test_text, tokenizer)
        print(f"Test correction: '{test_text}' → '{result}'")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
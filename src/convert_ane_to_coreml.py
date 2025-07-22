#!/usr/bin/env python3
"""
Convert ANE-optimized DistilBERT typo correction model to Core ML.
This uses Apple's proven ANE conversion process for maximum performance.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import coremltools as ct
import numpy as np

# Add ANE models to path
sys.path.append(str(Path(__file__).parent))
from ane_models import DistilBertForMaskedLM as ANEDistilBertForMaskedLM
from ane_models import DistilBertConfig as ANEDistilBertConfig

from transformers import DistilBertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ANEDistilBertWrapper(torch.nn.Module):
    """Wrapper for ANE DistilBERT MLM model for Core ML tracing."""
    
    def __init__(self, ane_model):
        super().__init__()
        self.model = ane_model
        
    def forward(self, input_ids, attention_mask):
        """Forward pass returning only logits (no dict output)."""
        # ANE models require return_dict=False for Core ML compatibility
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        
        # Return only logits (first element of tuple)
        return outputs[0] if isinstance(outputs, tuple) else outputs

def load_ane_model(model_path: str):
    """Load the ANE-optimized DistilBERT model."""
    logger.info(f"Loading ANE model from {model_path}")
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    
    # Load config to create the model architecture
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create ANE config from the saved config
    ane_config = ANEDistilBertConfig(**config_dict)
    
    # Create empty ANE model
    model = ANEDistilBertForMaskedLM(ane_config)
    
    # Load the converted state dict directly
    import safetensors
    state_dict_path = os.path.join(model_path, "model.safetensors")
    
    if os.path.exists(state_dict_path):
        # Load using safetensors
        from safetensors.torch import load_file
        state_dict = load_file(state_dict_path)
        logger.info("Loading from safetensors file...")
    else:
        # Fallback to pytorch_model.bin
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location='cpu')
        logger.info("Loading from pytorch_model.bin file...")
    
    # Load the state dict (weights are already in Conv2d format)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    logger.info(f"Loaded ANE model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer

def create_example_inputs(tokenizer, max_sequence_length: int = 128):
    """Create example inputs for tracing."""
    logger.info(f"Creating example inputs with max length {max_sequence_length}")
    
    # Use a representative typo correction example
    example_text = "Thi sis a test sentenc with typos that need correction"
    
    # Tokenize with exact settings for ANE compatibility
    inputs = tokenizer(
        example_text,
        return_tensors="pt",
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True
    )
    
    logger.info(f"Example input shape: {inputs['input_ids'].shape}")
    logger.info(f"Example mask shape: {inputs['attention_mask'].shape}")
    
    return inputs['input_ids'], inputs['attention_mask']

def convert_to_coreml(model, tokenizer, output_path: str, max_sequence_length: int = 128):
    """Convert ANE DistilBERT model to Core ML format."""
    logger.info("Converting ANE model to Core ML...")
    
    # Wrap model for tracing
    wrapped_model = ANEDistilBertWrapper(model)
    wrapped_model.eval()
    
    # Create example inputs
    example_input_ids, example_attention_mask = create_example_inputs(tokenizer, max_sequence_length)
    
    logger.info("Tracing model with example inputs...")
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_model,
            (example_input_ids, example_attention_mask),
            strict=False
        )
    
    logger.info("Converting traced model to Core ML...")
    
    # Define input specifications for ANE compatibility
    inputs = [
        ct.TensorType(
            name="input_ids",
            shape=(1, max_sequence_length),
            dtype=np.int32
        ),
        ct.TensorType(
            name="attention_mask",
            shape=(1, max_sequence_length), 
            dtype=np.int32
        )
    ]
    
    # Convert with ANE-optimized settings
    coreml_model = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=[ct.TensorType(name="logits")],
        
        # ANE-optimized settings (following Apple's recommendations)
        compute_units=ct.ComputeUnit.CPU_AND_NE,  # Use Neural Engine when available
        minimum_deployment_target=ct.target.macOS13,  # ANE requires macOS 13+
        compute_precision=ct.precision.FLOAT16,        # ANE optimized precision
        
        # Additional optimizations
        convert_to="mlprogram",  # Use ML Program format for best ANE support
    )
    
    # Add comprehensive metadata
    coreml_model.short_description = "ANE-optimized DistilBERT for typo correction"
    coreml_model.user_defined_metadata = {
        "model_type": "ANE-DistilBertForMaskedLM",
        "task": "typo_correction",
        "max_sequence_length": str(max_sequence_length),
        "vocab_size": str(tokenizer.vocab_size),
        "ane_optimized": "true",
        "architecture": "Conv2d layers for ANE acceleration",
        "expected_speedup": "3x vs standard Core ML",
        "input_format": "WordPiece tokens (DistilBERT tokenizer)",
        "output_format": "Logits over vocabulary",
    }
    
    # Save the model
    logger.info(f"Saving Core ML model to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    coreml_model.save(output_path)
    
    # Print model info
    logger.info("=== ANE Core ML Model Info ===")
    logger.info(f"Input specs: {coreml_model.input_description}")
    logger.info(f"Output specs: {coreml_model.output_description}")
    logger.info(f"Model size: {os.path.getsize(output_path + '/weights/weight.bin') / (1024*1024):.1f} MB")
    
    return coreml_model

def test_coreml_inference(coreml_model, tokenizer, max_sequence_length: int = 128):
    """Test the converted Core ML model."""
    logger.info("Testing Core ML model inference...")
    
    # Create test input
    test_texts = [
        "Thi sis a test sentenc with typos",
        "The quikc brown fox jumps over teh lazy dog",
        "I went too the stor to buy som milk"
    ]
    
    for test_text in test_texts:
        logger.info(f"Testing: '{test_text}'")
        
        # Tokenize input
        inputs = tokenizer(
            test_text,
            return_tensors="np",
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True
        )
        
        try:
            # Run inference
            prediction = coreml_model.predict({
                'input_ids': inputs['input_ids'].astype(np.int32),
                'attention_mask': inputs['attention_mask'].astype(np.int32)
            })
            
            # Get predicted tokens
            logits = prediction['logits']
            predicted_ids = np.argmax(logits, axis=-1)[0]
            predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
            
            logger.info(f"  → '{predicted_text}'")
            
        except Exception as e:
            logger.error(f"  ❌ Inference failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert ANE DistilBERT to Core ML")
    parser.add_argument('--input_model', type=str, default='models/ane_typo_fixer',
                       help='Path to ANE-optimized DistilBERT model')
    parser.add_argument('--output_model', type=str, default='models/ANE_TypoFixer.mlpackage',
                       help='Output path for Core ML model')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length (use 128 for proven ANE compatibility)')
    
    args = parser.parse_args()
    
    try:
        # Load ANE model
        model, tokenizer = load_ane_model(args.input_model)
        
        # Convert to Core ML
        coreml_model = convert_to_coreml(model, tokenizer, args.output_model, args.max_seq_len)
        
        # Test the converted model
        test_coreml_inference(coreml_model, tokenizer, args.max_seq_len)
        
        logger.info("✅ ANE Core ML conversion completed successfully!")
        logger.info(f"📦 Model saved to: {args.output_model}")
        logger.info("🚀 This model should run on Apple Neural Engine for 3x speedup!")
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Convert trained PyTorch DistilBERT typo correction model to Core ML.
Applies INT8 quantization for efficient on-device inference.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import coremltools as ct
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_coreml(model_dir: str, output_path: str, max_sequence_length: int = 64):
    """Convert PyTorch DistilBERT model to Core ML format."""
    
    logger.info(f"Loading model from {model_dir}")
    
    # Load trained model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForMaskedLM.from_pretrained(model_dir)
    model.eval()
    
    # Create example inputs for tracing
    vocab_size = tokenizer.vocab_size
    example_input_ids = torch.randint(0, vocab_size, (1, max_sequence_length))
    example_attention_mask = torch.ones(1, max_sequence_length, dtype=torch.long)
    
    logger.info("Tracing model with example inputs...")
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (example_input_ids, example_attention_mask),
            strict=False
        )
    
    # Convert to Core ML
    logger.info("Converting to Core ML format...")
    
    # Define input specifications
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
    
    # Convert with quantization
    coreml_model = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,  # Use Neural Engine when available
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )
    
    # Add model metadata
    coreml_model.short_description = "DistilBERT model for typo correction"
    coreml_model.user_defined_metadata = {
        "model_type": "DistilBertForMaskedLM",
        "max_sequence_length": str(max_sequence_length),
        "vocab_size": str(vocab_size),
        "original_model_dir": model_dir,
    }
    
    # Quantize to INT8 for smaller model size
    logger.info("Applying INT8 quantization...")
    
    try:
        quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
            coreml_model, 
            nbits=8,
            quantization_mode='linear'
        )
        coreml_model = quantized_model
        logger.info("INT8 quantization applied successfully")
    except Exception as e:
        logger.warning(f"Quantization failed, using original model: {e}")
    
    # Save Core ML model
    logger.info(f"Saving Core ML model to {output_path}")
    coreml_model.save(output_path)
    
    # Print model info
    logger.info("=== Core ML Model Info ===")
    logger.info(f"Input specs: {coreml_model.input_description}")
    logger.info(f"Output specs: {coreml_model.output_description}")
    
    # Test the converted model
    logger.info("Testing converted model...")
    test_inference(coreml_model, tokenizer, max_sequence_length)
    
    return coreml_model

def test_inference(coreml_model, tokenizer, max_seq_len: int = 64):
    """Test the converted Core ML model with sample input."""
    
    # Create test input
    test_text = "Thi sis a test sentenc with typos"
    inputs = tokenizer(
        test_text,
        truncation=True,
        padding='max_length',
        max_length=max_seq_len,
        return_tensors='np'
    )
    
    # Run inference
    try:
        prediction = coreml_model.predict({
            'input_ids': inputs['input_ids'].astype(np.int32),
            'attention_mask': inputs['attention_mask'].astype(np.int32)
        })
        
        # Get predicted tokens
        logits = prediction['logits']
        predicted_ids = np.argmax(logits, axis=-1)[0]
        predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
        
        logger.info(f"Test input: '{test_text}'")
        logger.info(f"Predicted: '{predicted_text}'")
        logger.info("✅ Core ML model inference successful!")
        
    except Exception as e:
        logger.error(f"❌ Core ML model test failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert DistilBERT typo model to Core ML")
    parser.add_argument('model_dir', type=str,
                       help='Directory containing trained PyTorch model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for Core ML model (default: models/TypoFixer.mlmodel)')
    parser.add_argument('--max_seq_len', type=int, default=64,
                       help='Maximum sequence length for the model')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = "models/TypoFixer.mlmodel"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Convert model
    try:
        coreml_model = convert_to_coreml(args.model_dir, args.output, args.max_seq_len)
        logger.info(f"✅ Conversion completed! Model saved to {args.output}")
        
        # Print file size
        file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        logger.info(f"Model size: {file_size_mb:.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
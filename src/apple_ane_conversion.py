#!/usr/bin/env python3
"""
Apple ANE DistilBERT conversion using Apple's official architecture and weight converter.
This follows Apple's proven approach for maximum ANE acceleration.
"""

import argparse
import json
import logging
import os
import sys
import shutil
from pathlib import Path

import torch
import coremltools as ct
import numpy as np
from safetensors.torch import load_file, save_file
from transformers import DistilBertTokenizer, DistilBertForMaskedLM

# Add ANE models to path
sys.path.append(str(Path(__file__).parent / 'ane_models'))
from modeling_distilbert_ane import DistilBertForMaskedLM as ANEDistilBertForMaskedLM
from configuration_distilbert_ane import DistilBertConfig as ANEDistilBertConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_finetuned_model(model_path: str):
    """Load our fine-tuned typo correction model."""
    logger.info(f"Loading fine-tuned model from {model_path}")
    
    # Load fine-tuned model and tokenizer
    model = DistilBertForMaskedLM.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    
    logger.info(f"Fine-tuned model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer

def create_ane_model_with_finetuned_weights(finetuned_model, output_path: str):
    """Create ANE model with our fine-tuned weights using Apple's architecture."""
    logger.info("Creating ANE model with fine-tuned weights...")
    
    # Create ANE config matching our fine-tuned model
    ane_config = ANEDistilBertConfig(
        vocab_size=finetuned_model.config.vocab_size,
        dim=finetuned_model.config.dim,
        hidden_dim=finetuned_model.config.hidden_dim,
        n_heads=finetuned_model.config.n_heads,
        n_layers=finetuned_model.config.n_layers,
        max_position_embeddings=finetuned_model.config.max_position_embeddings,
        dropout=finetuned_model.config.dropout,
        activation=finetuned_model.config.activation,
        architecture="DistilBertForMaskedLM"
    )
    
    # Create ANE model with Apple's architecture
    ane_model = ANEDistilBertForMaskedLM(ane_config)
    
    # Convert and load weights using Apple's conversion function
    logger.info("Converting Linear weights to Conv2d format using Apple's converter...")
    
    # Get fine-tuned state dict
    finetuned_state_dict = finetuned_model.state_dict()
    
    # Apply Apple's linear_to_conv2d_map function manually
    converted_state_dict = {}
    for k, v in finetuned_state_dict.items():
        # Check if this is a weight that needs conversion (following Apple's logic)
        is_internal_proj = all(substr in k for substr in ['lin', '.weight'])
        is_output_proj = all(substr in k for substr in ['classifier', '.weight'])
        # Also convert vocab layers (transform and projector)
        is_vocab_layer = any(substr in k for substr in ['vocab_transform.weight', 'vocab_projector.weight'])
        
        if (is_internal_proj or is_output_proj or is_vocab_layer) and len(v.shape) == 2:
            # Convert Linear weight [out_features, in_features] to Conv2d [out_channels, in_channels, 1, 1]
            converted_state_dict[k] = v[:, :, None, None]
            logger.info(f"Converted {k}: {v.shape} → {converted_state_dict[k].shape}")
        else:
            converted_state_dict[k] = v
    
    # Load converted weights into ANE model
    missing_keys, unexpected_keys = ane_model.load_state_dict(converted_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    ane_model.eval()
    
    # Save ANE model
    logger.info(f"Saving ANE model to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_path, "config.json"), 'w') as f:
        json.dump(ane_config.__dict__, f, indent=2)
    
    # Save converted weights (handle shared tensors)
    try:
        save_file(converted_state_dict, os.path.join(output_path, "model.safetensors"))
    except RuntimeError as e:
        if "share memory" in str(e):
            logger.warning("Shared memory detected, using PyTorch save instead")
            torch.save(converted_state_dict, os.path.join(output_path, "pytorch_model.bin"))
        else:
            raise
    
    # Copy tokenizer files from fine-tuned model
    tokenizer_files = [
        'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json'
    ]
    
    finetuned_dir = Path(finetuned_model.config._name_or_path) if hasattr(finetuned_model.config, '_name_or_path') else Path('models/test_model')
    
    for filename in tokenizer_files:
        src_path = finetuned_dir / filename
        if src_path.exists():
            shutil.copy2(src_path, os.path.join(output_path, filename))
    
    logger.info(f"✅ ANE model created successfully at {output_path}")
    return ane_model

class ANEDistilBertWrapper(torch.nn.Module):
    """Apple ANE DistilBERT wrapper for Core ML tracing."""
    
    def __init__(self, ane_model):
        super().__init__()
        self.model = ane_model
        
    def forward(self, input_ids, attention_mask):
        """Forward pass returning only logits for Core ML compatibility."""
        # ANE models require return_dict=False for Core ML
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        # Return only logits (first element of tuple)
        return outputs[0] if isinstance(outputs, tuple) else outputs

def convert_ane_to_coreml(ane_model, tokenizer, output_path: str, max_sequence_length: int = 128):
    """Convert ANE model to Core ML using Apple's proven settings."""
    logger.info("Converting ANE model to Core ML with Apple's settings...")
    
    # Wrap model for tracing
    wrapped_model = ANEDistilBertWrapper(ane_model)
    wrapped_model.eval()
    
    # Create example inputs (following Apple's approach)
    example_text = "The Neural Engine is really fast"
    inputs = tokenizer(
        example_text,
        return_tensors="pt",
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True
    )
    
    logger.info(f"Example input shapes: input_ids={inputs['input_ids'].shape}, mask={inputs['attention_mask'].shape}")
    
    # Trace the model
    logger.info("Tracing ANE model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_model,
            (inputs['input_ids'], inputs['attention_mask']),
            strict=False
        )
    
    logger.info("Converting to Core ML with ANE optimization...")
    
    # Convert with Apple's recommended settings for ANE
    coreml_model = ct.convert(
        traced_model,
        inputs=[
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
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        
        # Apple's ANE optimization settings
        compute_units=ct.ComputeUnit.CPU_AND_NE,  # Enable Neural Engine
        minimum_deployment_target=ct.target.macOS13,  # ANE requires macOS 13+
        compute_precision=ct.precision.FLOAT16,        # ANE optimized precision
        convert_to="mlprogram",  # ML Program format for best ANE support
    )
    
    # Add metadata following Apple's pattern
    coreml_model.short_description = "ANE-optimized DistilBERT for typo correction"
    coreml_model.user_defined_metadata = {
        "model_type": "ANE-DistilBertForMaskedLM",
        "task": "typo_correction",
        "max_sequence_length": str(max_sequence_length),
        "vocab_size": str(tokenizer.vocab_size),
        "ane_optimized": "true",
        "architecture": "Apple ANE DistilBERT with Conv2d layers",
        "expected_speedup": "3x vs standard Core ML on ANE",
        "conversion_method": "Apple official ANE architecture",
        "bc1s_format": "supported",
    }
    
    # Save the model
    logger.info(f"Saving ANE Core ML model to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    coreml_model.save(output_path)
    
    return coreml_model

def test_ane_coreml_model(coreml_model, tokenizer, max_sequence_length: int = 128):
    """Test the ANE Core ML model with typo correction examples."""
    logger.info("Testing ANE Core ML model...")
    
    test_texts = [
        "Thi sis a test sentenc with typos",
        "The quikc brown fox jumps over teh lazy dog",
        "I went too the stor to buy som milk",
        "Ther are many mistaks in this sentance",
        "Its a beutiful day outsid today"
    ]
    
    for test_text in test_texts:
        logger.info(f"Testing: '{test_text}'")
        
        # Tokenize input following Apple's approach
        inputs = tokenizer(
            test_text,
            return_tensors="np",
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True
        )
        
        try:
            # Run ANE Core ML inference
            prediction = coreml_model.predict({
                'input_ids': inputs['input_ids'].astype(np.int32),
                'attention_mask': inputs['attention_mask'].astype(np.int32)
            })
            
            # Extract logits and decode
            logits = prediction['logits']
            predicted_ids = np.argmax(logits, axis=-1)[0]
            predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
            
            logger.info(f"  → '{predicted_text.strip()}'")
            logger.info("✅ ANE inference successful!")
            
        except Exception as e:
            logger.error(f"❌ ANE inference failed: {e}")

def benchmark_ane_performance(coreml_model, tokenizer, num_runs: int = 50, max_sequence_length: int = 128):
    """Benchmark ANE Core ML model performance."""
    logger.info(f"Benchmarking ANE performance with {num_runs} runs...")
    
    # Create benchmark input
    test_text = "This is a test sentence with potential typos for benchmarking performance"
    inputs = tokenizer(
        test_text,
        return_tensors="np",
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True
    )
    
    input_dict = {
        'input_ids': inputs['input_ids'].astype(np.int32),
        'attention_mask': inputs['attention_mask'].astype(np.int32)
    }
    
    import time
    
    # Warmup runs
    logger.info("Warming up ANE...")
    for _ in range(5):
        coreml_model.predict(input_dict)
    
    # Benchmark runs
    logger.info("Running benchmark...")
    start_time = time.time()
    for _ in range(num_runs):
        coreml_model.predict(input_dict)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    logger.info(f"🚀 ANE Average inference time: {avg_time*1000:.2f}ms")
    logger.info(f"⚡ ANE Throughput: {1/avg_time:.1f} inferences/second")
    
    return avg_time

def main():
    parser = argparse.ArgumentParser(description="Convert fine-tuned DistilBERT to ANE using Apple's architecture")
    parser.add_argument('--input_model', type=str, default='models/test_model',
                       help='Path to fine-tuned DistilBERT model')
    parser.add_argument('--ane_model_path', type=str, default='models/apple_ane_typo_fixer',
                       help='Output path for ANE model')
    parser.add_argument('--coreml_output', type=str, default='models/Apple_ANE_TypoFixer.mlpackage',
                       help='Output path for Core ML model')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length (128 recommended for ANE)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    try:
        logger.info("🍎 Starting Apple ANE conversion process...")
        
        # Step 1: Load fine-tuned model
        logger.info("Step 1: Loading fine-tuned model...")
        finetuned_model, tokenizer = load_finetuned_model(args.input_model)
        
        # Step 2: Create ANE model with fine-tuned weights
        logger.info("Step 2: Creating ANE model with fine-tuned weights...")
        ane_model = create_ane_model_with_finetuned_weights(finetuned_model, args.ane_model_path)
        
        # Step 3: Convert to Core ML
        logger.info("Step 3: Converting to ANE Core ML...")
        coreml_model = convert_ane_to_coreml(ane_model, tokenizer, args.coreml_output, args.max_seq_len)
        
        # Step 4: Test the model
        logger.info("Step 4: Testing ANE Core ML model...")
        test_ane_coreml_model(coreml_model, tokenizer, args.max_seq_len)
        
        # Step 5: Benchmark if requested
        if args.benchmark:
            logger.info("Step 5: Benchmarking ANE performance...")
            benchmark_ane_performance(coreml_model, tokenizer, max_sequence_length=args.max_seq_len)
        
        logger.info("✅ Apple ANE conversion completed successfully!")
        logger.info(f"🍎 ANE model: {args.ane_model_path}")
        logger.info(f"📦 Core ML model: {args.coreml_output}")
        logger.info("🚀 Model should utilize Apple Neural Engine for maximum performance!")
        
        # Print model size
        if os.path.exists(args.coreml_output):
            model_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(args.coreml_output)
                for filename in filenames
            ) / (1024 * 1024)
            logger.info(f"📊 Core ML model size: {model_size:.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Apple ANE conversion failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Step 2: Generate causal mask
Loads tokens from step 1 and generates causal attention mask.
"""

import json
import numpy as np
import torch
from datetime import datetime

def make_causal_mask(length, start):
    """Create causal attention mask (from typo_fixer_complete.py)."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return torch.tensor(mask, dtype=torch.float16)

def main():
    """Load tokens and generate causal mask."""
    # Load tokens from step 1
    input_file = "test_generation/step_1_tokens.json"
    print(f"Loading tokens from {input_file}...")
    
    with open(input_file, 'r') as f:
        step1_data = json.load(f)
    
    # Extract data
    input_ids = np.array(step1_data["data"]["input_ids"], dtype=np.int32)
    context_pos = step1_data["data"]["context_pos"]
    
    print(f"Loaded input_ids shape: {input_ids.shape}")
    print(f"Context position: {context_pos}")
    print()
    
    # Generate causal mask (same parameters as in typo_fixer_complete.py)
    context_length = 256  # From meta.yaml in original
    causal_mask = make_causal_mask(context_length, 0)
    
    print(f"Generated causal mask shape: {causal_mask.shape}")
    print(f"Causal mask dtype: {causal_mask.dtype}")
    print()
    
    # Show sample of the mask (first few positions)
    print("Sample causal mask (first 8x8 positions):")
    sample_mask = causal_mask[0, 0, :8, :8].numpy()
    print(sample_mask)
    print()
    
    # Create output data with metadata
    output_data = {
        "metadata": {
            "step": "2_causal_mask",
            "input_file": input_file,
            "context_length": context_length,
            "context_pos": context_pos,
            "timestamp": datetime.now().isoformat(),
            "description": "Causal attention mask for transformer inference"
        },
        "data": {
            "input_ids": input_ids.tolist(),
            "input_ids_shape": list(input_ids.shape),
            "input_ids_dtype": str(input_ids.dtype),
            "causal_mask": causal_mask.numpy().tolist(),
            "causal_mask_shape": list(causal_mask.shape),
            "causal_mask_dtype": str(causal_mask.dtype),
            "context_pos": context_pos,
            "context_length": context_length
        }
    }
    
    # Save to file
    output_file = "test_generation/step_2_causal_mask.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved causal mask data to {output_file}")
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Causal mask shape: {causal_mask.shape}")
    print(f"   Context position: {context_pos}")
    print(f"   Context length: {context_length}")

if __name__ == "__main__":
    main()
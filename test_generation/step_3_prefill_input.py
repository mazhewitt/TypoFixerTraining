#!/usr/bin/env python3
"""
Step 3: Generate prefill input test data
Prepares all data needed for the prefill phase including batching, position IDs, and embeddings.
"""

import json
import numpy as np
import torch
import coremltools as ct
import os
from datetime import datetime

def run_embeddings(embeddings_model, input_ids):
    """Run embeddings model (from typo_fixer_complete.py)."""
    result = embeddings_model.predict({"input_ids": input_ids})
    return result['hidden_states']

def main():
    """Generate prefill input test data."""
    # Load causal mask data from step 2
    input_file = "test_generation/step_2_causal_mask.json"
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r') as f:
        step2_data = json.load(f)
    
    # Extract data
    input_ids = np.array(step2_data["data"]["input_ids"], dtype=np.int32)
    causal_mask = np.array(step2_data["data"]["causal_mask"], dtype=np.float16)
    context_pos = step2_data["data"]["context_pos"]
    context_length = step2_data["data"]["context_length"]
    
    print(f"Loaded input_ids shape: {input_ids.shape}")
    print(f"Loaded causal_mask shape: {causal_mask.shape}")
    print(f"Context position: {context_pos}")
    print()
    
    # Load embeddings model to generate hidden states
    model_dir = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane-flex"
    embeddings_path = os.path.join(model_dir, "qwen-typo-fixer_embeddings.mlpackage")
    print(f"Loading embeddings model from {embeddings_path}...")
    embeddings_model = ct.models.MLModel(embeddings_path)
    
    # Prepare prefill data (following the prefill logic from typo_fixer_complete.py)
    batch_size = 128
    batch_pos = 0
    batch_end = min(batch_pos + batch_size, context_pos)
    current_batch_size = batch_end - batch_pos
    
    print(f"Prefill batch: {batch_pos}-{batch_end-1} ({current_batch_size} tokens)")
    
    # Get current batch
    batch_input = input_ids[:, batch_pos:batch_end]
    
    # Always pad to full batch size for prefill (as done in original)
    if current_batch_size < batch_size:
        pad_size = batch_size - current_batch_size
        padding = np.zeros((1, pad_size), dtype=np.int32)
        batch_input = np.concatenate([batch_input, padding], axis=1)
    
    # Generate position IDs for full batch size
    position_ids = np.arange(batch_pos, batch_pos + batch_size, dtype=np.int32)
    batch_causal_mask = causal_mask[:, :, batch_pos:batch_pos + batch_size, :].astype(np.float16)
    
    print(f"Batch input shape: {batch_input.shape}")
    print(f"Position IDs shape: {position_ids.shape}")
    print(f"Batch causal mask shape: {batch_causal_mask.shape}")
    print()
    
    # Run embeddings to get hidden states
    print("Running embeddings model...")
    hidden_states = run_embeddings(embeddings_model, batch_input)
    print(f"Generated hidden_states shape: {hidden_states.shape}")
    print()
    
    # Create complete prefill input data
    prefill_inputs = {
        'hidden_states': hidden_states.astype(np.float16),
        'position_ids': position_ids,
        'causal_mask': batch_causal_mask,
        'current_pos': np.array([batch_pos], dtype=np.int32)
    }
    
    # Create output data with metadata
    output_data = {
        "metadata": {
            "step": "3_prefill_input",
            "input_file": input_file,
            "model_dir": model_dir,
            "batch_size": batch_size,
            "batch_pos": batch_pos,
            "batch_end": batch_end,
            "current_batch_size": current_batch_size,
            "timestamp": datetime.now().isoformat(),
            "description": "Complete input data for prefill phase"
        },
        "data": {
            "batch_input": batch_input.tolist(),
            "batch_input_shape": list(batch_input.shape),
            "batch_input_dtype": str(batch_input.dtype),
            "hidden_states": hidden_states.tolist(),
            "hidden_states_shape": list(hidden_states.shape),
            "hidden_states_dtype": str(hidden_states.dtype),
            "position_ids": position_ids.tolist(),
            "position_ids_shape": list(position_ids.shape),
            "position_ids_dtype": str(position_ids.dtype),
            "causal_mask": batch_causal_mask.tolist(),
            "causal_mask_shape": list(batch_causal_mask.shape),
            "causal_mask_dtype": str(batch_causal_mask.dtype),
            "current_pos": [batch_pos],
            "context_pos": context_pos,
            "context_length": context_length
        }
    }
    
    # Save to file
    output_file = "test_generation/step_3_prefill_input.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved prefill input data to {output_file}")
    print(f"   Batch input shape: {batch_input.shape}")
    print(f"   Hidden states shape: {hidden_states.shape}")
    print(f"   Position IDs shape: {position_ids.shape}")
    print(f"   Causal mask shape: {batch_causal_mask.shape}")
    print(f"   Current position: {batch_pos}")
    
    # Show sample data
    print(f"\nSample hidden states (first 5 values): {hidden_states[0, 0, :5]}")
    print(f"Sample position IDs (first 10): {position_ids[:10]}")

if __name__ == "__main__":
    main()
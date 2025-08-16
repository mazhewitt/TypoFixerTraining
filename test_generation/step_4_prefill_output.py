#!/usr/bin/env python3
"""
Step 4: Generate prefill output hidden states
Runs the prefill model and captures the output hidden states for the infer phase.
"""

import json
import numpy as np
import coremltools as ct
import os
from datetime import datetime

def main():
    """Run prefill and generate output hidden states."""
    # Load prefill input data from step 3
    input_file = "test_generation/step_3_prefill_input.json"
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r') as f:
        step3_data = json.load(f)
    
    # Extract prefill inputs
    hidden_states = np.array(step3_data["data"]["hidden_states"], dtype=np.float16)
    position_ids = np.array(step3_data["data"]["position_ids"], dtype=np.int32)
    causal_mask = np.array(step3_data["data"]["causal_mask"], dtype=np.float16)
    current_pos = step3_data["data"]["current_pos"]
    context_pos = step3_data["data"]["context_pos"]
    
    print(f"Loaded hidden_states shape: {hidden_states.shape}")
    print(f"Loaded position_ids shape: {position_ids.shape}")
    print(f"Loaded causal_mask shape: {causal_mask.shape}")
    print(f"Current position: {current_pos}")
    print(f"Context position: {context_pos}")
    print()
    
    # Load prefill model
    model_dir = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane-flex"
    prefill_path = os.path.join(model_dir, "qwen-typo-fixer_prefill_chunk_01of01.mlpackage")
    print(f"Loading prefill model from {prefill_path}...")
    prefill_model = ct.models.MLModel(prefill_path)
    
    # Initialize KV state
    print("Initializing KV state...")
    kv_state = prefill_model.make_state()
    
    # Prepare prefill inputs (following typo_fixer_complete.py)
    inputs = {
        'hidden_states': hidden_states,
        'position_ids': position_ids,
        'causal_mask': causal_mask,
        'current_pos': np.array(current_pos, dtype=np.int32)
    }
    
    print("Running prefill model...")
    print(f"  Hidden states input: {hidden_states.shape}")
    print(f"  Position IDs: {position_ids.shape}")
    print(f"  Causal mask: {causal_mask.shape}")
    print(f"  Current pos: {current_pos}")
    print()
    
    # Run prefill
    output = prefill_model.predict(inputs, kv_state)
    print(f"Prefill completed. Output keys: {list(output.keys())}")
    
    # Note: Prefill typically doesn't return hidden_states directly,
    # it just updates the KV cache state for the next infer step
    
    # Prepare data for the infer step (next token generation)
    # We need to get the hidden states for the next token position
    next_pos = context_pos  # Position where we'll generate the next token
    
    # For the infer step, we'll need a single token input
    # This would typically be the last token from the context, but for testing
    # we'll prepare the structure needed for single token inference
    
    # Create output data with KV state information
    output_data = {
        "metadata": {
            "step": "4_prefill_output",
            "input_file": input_file,
            "model_dir": model_dir,
            "prefill_completed": True,
            "next_token_pos": next_pos,
            "timestamp": datetime.now().isoformat(),
            "description": "Prefill completed, KV cache initialized for infer phase"
        },
        "data": {
            "prefill_inputs": {
                "hidden_states": hidden_states.tolist(),
                "hidden_states_shape": list(hidden_states.shape),
                "position_ids": position_ids.tolist(),
                "causal_mask": causal_mask.tolist(),
                "current_pos": current_pos
            },
            "prefill_output": {key: val.tolist() if hasattr(val, 'tolist') else val 
                              for key, val in output.items()},
            "kv_state_initialized": True,
            "next_token_pos": next_pos,
            "context_pos": context_pos,
            "ready_for_infer": True
        }
    }
    
    # Save to file
    output_file = "test_generation/step_4_prefill_output.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved prefill output data to {output_file}")
    print(f"   Prefill completed successfully")
    print(f"   KV state initialized")
    print(f"   Next token position: {next_pos}")
    print(f"   Ready for infer phase")
    
    # Show prefill output info
    if output:
        for key, val in output.items():
            if hasattr(val, 'shape'):
                print(f"   Prefill output '{key}': {val.shape}")
            else:
                print(f"   Prefill output '{key}': {type(val)}")

if __name__ == "__main__":
    main()
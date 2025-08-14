#!/usr/bin/env python3
"""
Inspect CoreML model input/output shapes for embeddings, prefill/infer, and LM head.
"""

import os
import sys
import coremltools as ct


def show(io_model, name):
    spec = io_model.get_spec()
    print(f"\n{name} inputs:")
    for i in spec.description.input:
        print(" -", i.name, i.type)
    print(f"{name} outputs:")
    for o in spec.description.output:
        print(" -", o.name, o.type)


def main(model_dir: str):
    emb_path = os.path.join(model_dir, "qwen-typo-fixer_embeddings.mlpackage")
    lm_path = os.path.join(model_dir, "qwen-typo-fixer_lm_head.mlpackage")
    # Prefer combined multifunction if present, else separate parts
    combined_ffn = os.path.join(model_dir, "qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage")
    ffn_only = os.path.join(model_dir, "qwen-typo-fixer_FFN_chunk_01of01.mlpackage")
    pf_only = os.path.join(model_dir, "qwen-typo-fixer_prefill_chunk_01of01.mlpackage")

    if not os.path.exists(emb_path):
        print(f"Embeddings not found: {emb_path}")
        return 1
    # Load FFN/prefill
    ffn_prefill = None
    ffn_infer = None
    if os.path.exists(combined_ffn):
        ffn_prefill = ct.models.MLModel(combined_ffn, function_name="prefill")
        ffn_infer = ct.models.MLModel(combined_ffn, function_name="infer")
    else:
        if os.path.exists(pf_only):
            ffn_prefill = ct.models.MLModel(pf_only)
        if os.path.exists(ffn_only):
            ffn_infer = ct.models.MLModel(ffn_only)
        if ffn_prefill is None and ffn_infer is None:
            print("FFN/Prefill not found: expected combined or separate models in directory")
            return 1
    if not os.path.exists(lm_path):
        print(f"LM head not found: {lm_path}")
        return 1

    emb = ct.models.MLModel(emb_path)
    lm = ct.models.MLModel(lm_path)

    show(emb, "Embeddings")
    show(ffn_prefill, "FFN Prefill")
    show(ffn_infer, "FFN Infer")
    show(lm, "LM Head")
    return 0


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/qwen-typo-fixer-ane"
    raise SystemExit(main(model_dir))

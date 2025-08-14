# Qwen Typo Fixer — CoreML (ANE) Models

Standalone repository containing CoreML `.mlpackage` artifacts and tokenizer for the Qwen typo-fixer.

## Contents

- qwen-typo-fixer_embeddings.mlpackage — Token embeddings → hidden states
- qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage — FFN + attention (prefill/infer)
- qwen-typo-fixer_lm_head_lut6.mlpackage — Final LM head (split logits)
- meta.yaml — Model metadata for pipelines
- tokenizer.json, tokenizer_config.json, vocab.json, merges.txt — Tokenizer files

## Inputs/Outputs (summary)

- Embeddings
  - input: input_ids [batch, seq_len] (int32)
  - output: hidden_states [batch, seq_len, hidden]
- FFN + Prefill/Infer
  - inputs: hidden_states [batch, S, hidden], position_ids [S], causal_mask [1,1,S,context], current_pos [1]
  - outputs: updated hidden_states (prefill may return [batch, 1, hidden] while only updating KV-cache), updated kv-cache (state)
- LM Head
  - input: hidden_states [batch, 1, hidden]
  - output: split logits (concat to vocab_size)

## Notes

- Converted and optimized for Apple Neural Engine (ANE) with LUT quantization.
- See training and reference implementation in the main project.

## Supported shapes and defaults

- Prefill sequence length S supported by this build: 64 (enumerated)
- Default prefill S: 64 (other sizes will raise a shape error)
- Context length used by reference: 256

If you recompile models with different chunking, the supported sizes may change. Probe via a small embeddings run with shape (1, N).

## Troubleshooting shape mismatches

If embeddings output is [1, S, 1024] (e.g., S=64) but your prefill path expects [1, 1, 1024], the prompt will be truncated and generation may leak prior state. Verify shapes directly from the CoreML models:

Python (macOS):

```python
import coremltools as ct

emb = ct.models.MLModel("qwen-typo-fixer_embeddings.mlpackage")
ffn_prefill = ct.models.MLModel("qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage", function_name="prefill")
ffn_infer = ct.models.MLModel("qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage", function_name="infer")
lm = ct.models.MLModel("qwen-typo-fixer_lm_head_lut6.mlpackage")

def show(io_model, name):
  spec = io_model.get_spec()
  print(f"\n{name} inputs:")
  for i in spec.description.input:
    print(" -", i.name, i.type)
  print(f"{name} outputs:")
  for o in spec.description.output:
    print(" -", o.name, o.type)

show(emb, "Embeddings")
show(ffn_prefill, "FFN Prefill")
show(ffn_infer, "FFN Infer")
show(lm, "LM Head")
```

Expectations:
- Embeddings: input_ids [1, 1] and [1, 64] (enumerated); hidden_states [1, S, 1024]
- FFN Prefill: hidden_states [1, S, 1024] with S=64; may output [1, 1, 1024] while updating KV-cache
- FFN Infer: hidden_states [1, 1, 1024]
- LM Head: hidden_states [1, 1, 1024]

If prefill shows S=1, re-export the prefill graph with S=64 (or a RangeDim/EnumeratedShapes) and keep LM head/infer at single-token.

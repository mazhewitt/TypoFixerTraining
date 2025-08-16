# Qwen Typo Fixer — CoreML (ANE) Models

Standalone repository containing CoreML `.mlpackage` artifacts and tokenizer for the Qwen typo-fixer.

## Contents

- qwen-typo-fixer_embeddings.mlpackage — Token embeddings → hidden states
- qwen-typo-fixer_prefill_chunk_01of01.mlpackage — FFN + attention (prefill)
- qwen-typo-fixer_FFN_chunk_01of01.mlpackage — FFN + attention (single‑token infer)
- qwen-typo-fixer_lm_head.mlpackage — Final LM head (split logits)
- meta.yaml — Model metadata for pipelines
- tokenizer.json, tokenizer_config.json, vocab.json, merges.txt — Tokenizer files

## Inputs/Outputs (summary)

- Embeddings (qwen-typo-fixer_embeddings.mlpackage)
  - input: input_ids [1, 1] or [1, 128] (int32)
  - output: hidden_states [1, S, 1024] (float16)
- Prefill (qwen-typo-fixer_prefill_chunk_01of01.mlpackage)
  - inputs:
  - hidden_states [1, 128, 1024] (float16)
  - position_ids [128] (int32)
  - causal_mask [1, 1, 128, 256] (float16)
    - current_pos [1] (int32)
  - output:
    - output_hidden_states [1, 1, 1024] (float16)
  - notes: prefill updates the KV‑cache in model state; the returned hidden state is not used downstream.
- Infer (qwen-typo-fixer_FFN_chunk_01of01.mlpackage)
  - inputs:
    - hidden_states [1, 1, 1024] (float16)
    - position_ids [1] (int32)
    - causal_mask [1, 1, 1, 256] (float16)
    - current_pos [1] (int32)
  - output:
    - output_hidden_states [1, 1, 1024] (float16)
- LM Head (qwen-typo-fixer_lm_head.mlpackage)
  - input: hidden_states [1, 1, 1024] (float16)
  - output: split logits across multiple tensors (concat along last dimension to vocab_size)

## Notes

- Converted and optimized for Apple Neural Engine (ANE) with LUT quantization.
- See training and reference implementation in the main project.
 - Minimum deployment target: iOS 18; compute precision float16; compute units CPU_AND_NE.

## Supported shapes and defaults

- Prefill sequence length S supported by this build: 128 (fixed)
- Embeddings accept [1, 1] and [1, 128]; infer is single‑token [1, 1, 1024].
- Context length: 256

If you recompile models with different chunking, the supported sizes may change. Probe via a small embeddings run with shape (1, N).

## Troubleshooting shape mismatches

If embeddings output is [1, S, 1024] (e.g., S=128) but your prefill path expects [1, 1, 1024], the prompt will be truncated and generation may leak prior state. Verify shapes directly from the CoreML models:

Python (macOS):

```python
import coremltools as ct

emb = ct.models.MLModel("qwen-typo-fixer_embeddings.mlpackage")
ffn_prefill = ct.models.MLModel("qwen-typo-fixer_prefill_chunk_01of01.mlpackage")
ffn_infer = ct.models.MLModel("qwen-typo-fixer_FFN_chunk_01of01.mlpackage")
lm = ct.models.MLModel("qwen-typo-fixer_lm_head.mlpackage")

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
- Embeddings: input_ids [1, 1] and [1, 128]; hidden_states [1, S, 1024]
- FFN Prefill: hidden_states [1, 128, 1024]; output_hidden_states [1, 1, 1024] (KV‑cache updated)
- FFN Infer: hidden_states [1, 1, 1024]
- LM Head: hidden_states [1, 1, 1024]

If prefill shows S=1, re-export the prefill graph with S=128 (or a RangeDim/EnumeratedShapes) and keep LM head/infer at single-token.

## Constraints and gotchas

- CoreML enumerated shapes must be symmetrical across flexible inputs. If you export prefill with enumerated S values, every flexible input (hidden_states, position_ids, causal_mask, and current_pos if made flexible) must expose the same number of enumerated shapes, or CoreML will raise AsymmetricalEnumeratedShapesException. This repo’s prefill uses a fixed S=128 to avoid that runtime constraint.
- Types: ids (input_ids, position_ids, current_pos) are int32; masks/hidden_states are float16.
- Causal mask shape is [1, 1, S, context_length] with zeros on allowed positions and -inf elsewhere.
- LM head emits multiple logits tensors (e.g., logits1..logits16). Concatenate them along the last axis to form full vocab logits; chunk sizes can vary slightly across heads.

## Examples

Few-shot prompt with a longer sentence (S=128 prefill):

```text
Fix typos in these sentences:

Input: I beleive this is teh answer.
Output: I believe this is the answer.

Input: She recieved her degre yesterday.
Output: She received her degree yesterday.

Input: The resturant serves good food.
Output: The restaurant serves good food.

Input: Thiss is a longg sentnce with severl typoes, includng mispellngs and grammer errros; it shoud stil be corected propery by the fixr even if it goes a bit longer than usual.
Output:
```

Expected output snippet (greedy, temperature 0.0):

```text
This is a long sentence with several typos, including mispellings and grammar errors; it should
```

Notes:
- The example uses a few-shot style prompt to encourage corrections; the model may truncate after a clause depending on max_new_tokens.
- With S=128, the prefill can accommodate longer prompts without truncation.

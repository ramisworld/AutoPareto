# Protected candidate and checkpoint interface

Only candidate copies of `train.py` are editable. Training behavior remains defined by
that file, while checkpointing and inference measurement are protected.

## Required training definitions

Before the unchanged setup marker, a compatible candidate exposes:

- `GPTConfig(**checkpoint_config)`;
- `GPT(config)`;
- `model.init_weights()`;
- a strict `state_dict` that can be saved and loaded;
- the canonical block fields consumed by `protected/inference_adapter.py`.

The setup marker must remain exactly
`# Setup: tokenizer, model, optimizer, dataloader`. After execution the protected
launcher requires `model`, `config`, `val_bpb`, `total_tokens`,
`total_training_time`, `step`, `TOTAL_BATCH_SIZE`, `peak_vram_mb`, and
`num_params`. The protected validation evaluator must be called exactly once. Its
result and the protected 300–315 second measured-training window are checked outside
the candidate's reporting logic.

The protected `save_training_checkpoint` and `load_checkpoint_file` functions save and
load configuration, seed, train filename, and state dict. `load_checkpoint_model`
constructs on `meta`, materializes on the evaluation device, restores intended parameter
dtypes through `init_weights`, and loads the checkpoint strictly.

## Inference adapter

The protected `CandidateInferenceAdapter` contract is:

```python
prefill(input_ids) -> (logits, decode_state)
decode_one(one_token, decode_state) -> (logits, new_decode_state)
cache_nbytes(decode_state) -> int
```

The canonical protected adapter implements per-layer rotated K/V caches, value-embed
gating, grouped-query head expansion, causal/sliding-window masks, residual scalars,
MLPs, final normalization, LM head, and logit soft-capping.

An architecture incompatible with the canonical fields may define
`build_inference_adapter(model)` in candidate `train.py` above the setup marker. That
factory returns the same three-method interface. Candidate cache implementation is
model code and is included in the recorded diff; candidate-supplied timing,
correctness, prompts, thresholds, repetition control, or benchmark-aware branching
remain forbidden. The protected evaluator checks common-prefix cached and uncached
logits at multiple prefix lengths before it accepts any timing or memory measurement.
It also traverses the returned decode state and rejects `cache_nbytes(state)` values
that do not equal the unique tensor storage reachable from that state.
No candidate cache is valid until that protected A40 correctness sequence passes.

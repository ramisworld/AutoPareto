# AutoPareto campaign

Read `RESEARCH_PROTOCOL.md` and `docs/candidate_interface.md` completely. Run one
independent 25-attempt AutoPareto replicate using the registered campaign ID, training
seed, and baseline experiment supplied by the user.

## Scientific contract

- Start from the shared A40 baseline source commit
  `5ea452c9d86a43ad4bebd507c5872f18249627ce`.
- Modify only one candidate copy of `train.py` per attempt.
- Use `scripts/run_runner.sh` for every runner command; never invoke `runner.py`
  directly.
- Use exactly the registered seed for every candidate.
- Submit attempts 1 through 25 exactly. Every crash, OOM, timeout, interruption,
  inference incompatibility, or protection failure consumes its attempt. If training
  completed and protected inference, cache correctness, or inference-memory
  measurement then fails, preserve that candidate checkpoint and stop the campaign
  immediately after recording the exact error and current attempt. Do not propose or
  reserve the next attempt; the runner pauses the campaign for maintainer review.
- Use only `scripts/run_runner.sh campaign-view --campaign-id <campaign-id>` to inspect
  research history. Do not inspect the global database, global frontier exports,
  AutoResearch, confirmation, ablation, classical-search, or another AutoPareto
  campaign.
- Do not modify protected infrastructure, benchmark semantics, prompts, datasets,
  tokenizer, validation split, precision, evaluator, metric calculations, or timing.

## Complete candidate research space

The complete candidate `train.py` is the research surface, not merely its scalar
hyperparameters. Evidence may motivate changes to:

- transformer depth, width, attention, head layout, KV heads, windowing, RoPE,
  embeddings, unembeddings, MLPs, normalization, residual paths, parameter sharing,
  or architectural simplification;
- optimizer algorithms and settings, learning-rate schedules, warm-up, warm-down,
  weight decay, batching, gradient accumulation, and training math;
- inference-oriented model structure and a candidate cache adapter that satisfies the
  protected interface.

Do not force architectural novelty for its own sake. Choose each hypothesis from the
current campaign evidence and its intended quality–speed–memory trade-off.

The following protected-launcher contract must remain intact:

- keep the marker `# Setup: tokenizer, model, optimizer, dataloader`;
- definitions required for checkpoint loading must appear above that marker;
- expose `GPTConfig`, `GPT`, `model.init_weights()`, and a strict loadable
  `state_dict`;
- finish with `model`, `config`, `val_bpb`, `total_tokens`,
  `total_training_time`, `step`, `TOTAL_BATCH_SIZE`, `peak_vram_mb`, and
  `num_params`;
- call the protected `evaluate_bpb` exactly once and do not replace, wrap, bypass, or
  reinterpret its result;
- keep measured training time inside the protected 300–315 second window.

The canonical architecture uses the protected fallback KV-cache adapter. An
architecture that changes the canonical fields may define
`build_inference_adapter(model)` above the setup marker. The returned object must
provide:

```python
prefill(input_ids) -> (logits, state)
decode_one(one_token, state) -> (logits, new_state)
cache_nbytes(state) -> int
```

This adapter is candidate model code, not benchmark code. It must perform the same
model computation as the uncached forward path, report the exact live tensor storage
used by its cache, and must not branch on the protected prompt, call order, repetition,
timer, or evaluator. The protected evaluator owns correctness, prompts, timing, memory
measurement, repetitions, and acceptance.

## Three-objective decision rule

AutoPareto does not advance only the lowest-quality model. It minimizes validation
BPB, maximizes protected median cached-decode tokens/second, and minimizes protected
peak allocated inference CUDA memory.

A candidate is `retained` exactly when it is non-dominated within the registered
same-seed baseline plus this campaign. It may be retained with slightly worse BPB when
it provides a real speed or memory advantage. Cache correctness is a mandatory gate,
not a tradeable objective. An inference-incompatible candidate preserves its BPB as
quality-only evidence but is `dominated` for the three-objective archive.

Before attempt 1, verify the immutable registration:

```bash
scripts/run_runner.sh register-campaign \
  --campaign-id <campaign-id> --method autopareto --phase campaign \
  --attempt-limit 25 --seed <42|0|1> \
  --baseline-experiment-id <same-seed-baseline-id> \
  --program programs/program_autopareto.md \
  --coding-model gpt-5.6-sol --reasoning-level high
```

Stop if the seed, baseline, method, budget, model, reasoning level, or program hash
does not match.

## Genuine sequential loop

For each attempt:

1. Read the campaign-isolated view.
2. Select one explicit current frontier parent and state which trade-off region it
   represents.
3. Form one testable hypothesis and record its category, intended change, exact
   candidate diff, and visible rationale.
4. Create only the current candidate. Do not create or encode future proposals,
   candidate files, parameter tables, queues, controllers, or hidden retries.
5. Run the candidate exactly once:

```bash
scripts/run_runner.sh run \
  --candidate queue/<campaign-id>/<N>-train.py \
  --campaign-id <campaign-id> --experiment-number <N> \
  --attempt-limit 25 --method autopareto --phase campaign --seed <registered-seed> \
  --baseline-experiment-id <same-seed-baseline-id> \
  --program programs/program_autopareto.md \
  --coding-model gpt-5.6-sol --reasoning-level high \
  --parent-experiment-id <current-frontier-id> \
  --hypothesis "<testable quality-speed-memory claim>" \
  --intended-change "<precise change>" \
  --change-category "<architecture|optimizer|schedule|batching|systems|other>" \
  --description "<hypothesis and parent rationale>"
```

6. Record quality and inference compatibility separately. When compatible, record
   cached speed, speed variation, allocated/reserved memory, prefill latency, and
   KV-cache growth.
7. Record the mechanically correct `retained` or `dominated` decision and a concise
   interpretation with `scripts/run_runner.sh record-decision`.
8. Only after that decision is stored may the next hypothesis be formulated.

The runner rejects wrong-seed baselines, cross-campaign or dominated parents, missing
previous decisions, incorrect attempt numbers, changed program hashes, and incorrect
Pareto decisions.

Stop after attempt 25. Do not run confirmation, ablation, another seed, a classical
method, deployment characterization, or a hybrid in this task.

# Downstream evaluation protocol and audit

This document describes what was actually run and persisted. The evaluator is the
project-local [`downstream_evaluator.py`](../downstream_evaluator.py),
not `lm-eval-harness`.

The frozen specification is
[`evaluation/downstream_suite.json`](../evaluation/downstream_suite.json).
It defines post-search characterization only: downstream scores do not influence
search proposals, Pareto retention, or the headline training objectives.

## Runs and reproducibility

The original five-minute panel contains 11 selected representatives: 99 task metrics
and 198 qualitative generations. Its raw bundles are under
`results/downstream_evaluations/`, and its suite SHA is:

```text
3dce2c736a7108a855d1150e1f39e2d3d13ca50cfadfff4f568dfed80ed72ef3
```

The equal-token campaign contains 21 new checkpoints (experiments 301–321; downstream
runs 14–34), producing 189 task metrics and 378 generations. Its database is
`results/phase1/downstream_200m.db`; raw bundles are under
`results/phase1/downstream_200m_evaluations/`.

Those database and raw bundles were used for the local audit but are intentionally not
included in this public branch. The checked-in
[`results/phase1/public_downstream_metrics.json`](../results/phase1/public_downstream_metrics.json)
contains the aggregate means and sample standard deviations used by the publication
figures.

The companion `phase1_200m_downstream_summary.json` reports 32 runs, 288 task rows,
and 576 generations because it includes the 11 historical downstream rows copied into
the Phase 1 database. The campaign-scoped counts above are the correct counts for the
new equal-token evaluation.

All 21 equal-token runs used the same suite SHA, the same pinned datasets, and the
same runtime settings. The equal-token evaluator SHA was:

```text
696f9c4677d3f6538c8b92cedec08691175aa30e8b8e48e96b080f2d43c107d5
```

The original five-minute evaluator SHA was:

```text
97a59f9344aaf9472899aa5bbb2e922d446e887aa54f38e92e2ee010676d9c40
```

The SHA difference is real. The later evaluator materialized contiguous causal input
and target tensors to fix a `.view()` failure; the benchmark definitions were not
intentionally changed, but the evaluator artifacts are not byte-identical.

Each run also records checkpoint SHA-256, candidate SHA-256, suite SHA-256, evaluator
SHA-256, environment JSON, status, raw-result path, and timestamps. The schema is in
[`autopareto_db.py`](../autopareto_db.py).

Recorded runtime:

| Setting | Value |
|---|---|
| Hardware | NVIDIA A40 |
| Device | CUDA |
| Precision | BF16 autocast |
| PyTorch | `2.9.1+cu128` |
| CUDA | `12.8` |
| Python | `3.10.18` |
| `datasets` | `5.0.1` |
| Maximum sequence length | 2,048 tokens |
| Scoring batch size | 8 |
| Held-out bootstrap seed | `20260731` |

These values are specified at
[`downstream_suite.json`](../evaluation/downstream_suite.json)
and recorded by `environment_record()` in
[`downstream_evaluator.py`](../downstream_evaluator.py).

## Datasets and sample counts

| Task | Dataset and pinned revision | Config | Split | Examples per checkpoint | Subsampling/limit |
|---|---|---|---|---:|---|
| LAMBADA | `EleutherAI/lambada_openai` — `900124bf3b8235c6daf21033af9948b3f07346c4` | `default` | `test` | 5,153 | None; full split iterated |
| BLiMP | `nyu-mll/blimp` — `877fba0801ffb7cbd8c39c1ff314a46f053f6036` | `all` | `train` | 67,000 pairs | None; all 67 configs, 1,000 pairs each |
| PIQA | `baber/piqa` — `142f6d7367fd9877f0fb3b5734ea6a545f54cdd1` | None | `validation` | 1,838 | None; full split iterated |
| ARC-Easy | `allenai/ai2_arc` — `210d026faf9955653af8916fad021475a3f00453` | `ARC-Easy` | `test` | 2,376 | None; full split iterated |
| Held-out BPB | `karpathy/climbmix-400b-shuffle` — `915333b4f8b8684f39aeaafea600fea6f43fb703` | `shard_06541.parquet` | `heldout` | 5,242,880 scored tokens / 2,560 segments | Fixed sequential prefix of one shard |

The four standard datasets are obtained through `datasets.load_dataset()` with the
specified revision and split. BLiMP additionally calls
`get_dataset_config_names()`, sorts the names, and loads every configuration:
[`downstream_evaluator.py`](../downstream_evaluator.py).

The held-out parquet shard is downloaded directly from the pinned Hugging Face URL,
cached under `~/.cache/autopareto/downstream/`, and read with PyArrow:
[`downstream_evaluator.py`](../downstream_evaluator.py).

The databases confirm these counts for every original and equal-token run. The
equal-token campaign has 21 completed runs, 21 × 9 = 189 metric rows, and 21 × 18 =
378 generation rows.

## Common likelihood scoring

For a context `C` and continuation `K`:

1. The project tokenizer encodes `C` and `C + K`.
2. If the context is empty, the evaluator prepends the BOS token.
3. The causal input is `full_tokens[:-1]`; the target is `full_tokens[1:]`.
4. The model receives padded token tensors, with padding filled by BOS, in sorted
   input-length batches of eight.
5. The evaluator selects logits at the continuation positions, applies
   `log_softmax`, gathers the target-token log-probabilities, and sums them.

The implementation is
[`downstream_evaluator.py`](../downstream_evaluator.py)
and
[`downstream_evaluator.py`](../downstream_evaluator.py).

Requests longer than the maximum context are left-trimmed to fit. No per-example
token sequence, logits, or log-probabilities are persisted.

## LAMBADA

For each row, the code performs literal space splitting:

```python
parts = row["text"].split(" ")
context = " ".join(parts[:-1])
continuation = " " + parts[-1]
```

The final whitespace-delimited word, including its leading space, is scored. It may
contain multiple tokenizer tokens.

Perplexity is corpus-level:

```text
PPL = exp(-sum(example_log_likelihoods) / sum(target_token_counts))
```

Lower is better. Exact match is the fraction of examples for which every target
subtoken is the greedy top-1 prediction:

```text
exact_match = count(all greedy target subtokens correct) / 5,153
```

This is token-level greedy exact match of the final word, not string matching of a
sampled completion. The implementation is
[`downstream_evaluator.py`](../downstream_evaluator.py).

## PIQA and ARC-Easy

PIQA receives:

```text
Question: {goal}
Answer:
```

ARC-Easy receives:

```text
Question: {question}
Answer:
```

Each candidate continuation is prefixed with exactly one space. The ordinary choice
is the candidate with the greatest summed continuation log-likelihood:

```text
prediction = argmax_candidate(sum target-token log-probabilities)
accuracy = correct predictions / number of questions
```

The length-normalized choice divides each candidate’s log-likelihood by its number
of tokenizer tokens before taking the argmax:

```text
prediction_normalized = argmax_candidate(log_likelihood / max(1, token_count))
```

The evaluator compares the selected candidate index with the dataset label. The
implementation is
[`downstream_evaluator.py`](../downstream_evaluator.py).

## BLiMP

The evaluator confirms all 67 configurations and loads each configuration’s `train`
split. Each configuration contributes 1,000 minimal pairs, for 67,000 total pairs.
For each pair it scores the full good and bad sentences from an empty context and
counts the pair correct when:

```text
log_likelihood(good) > log_likelihood(bad)
```

Ties are incorrect. Per-configuration `accuracy`, `correct`, and `count` are stored
in the `raw_json` field of the BLiMP task rows. The database schema stores the full
JSON payload in `downstream_task_results.raw_json`:
[`autopareto_db.py`](../autopareto_db.py).

Macro accuracy is the unweighted mean of the 67 category accuracies. Micro accuracy
is the mean of all 67,000 binary pair decisions. Because each category has exactly
1,000 pairs, the two values are numerically equal in the current runs.

## Held-out bits per byte

The evaluator reads only the parquet `text` column, tokenizes sequentially, and
constructs the first 2,560 non-overlapping segments of 2,049 tokens. Each segment
produces 2,048 next-token losses:

```text
2,560 × 2,048 = 5,242,880 scored tokens
```

Token byte sizes come from the project tokenizer. Zero-byte tokens are masked. The
reported metric is:

```text
BPB = sum(losses in nats) / (ln(2) × sum(token byte sizes))
```

Lower is better. The evaluator also computes 1,000 deterministic segment-bootstrap
ratios with seed `20260731`; the interval is formed from the sorted bootstrap
estimates at the 2.5th and 97.5th percentile indices:
[`downstream_evaluator.py`](../downstream_evaluator.py).

Every equal-token run recorded the same shard SHA-256:

```text
51da1a9bc052dfd5a77988d6ca4d1cbe62b89eedb247b56850fc1b9b45e56c73
```

## Qualitative generation

Each checkpoint received the same six prompts, each at generation seeds 0, 1, and 2:

1. `The scientist opened the old notebook and discovered that`
2. `In simple terms, the reason the sky appears blue is`
3. `A key-value cache makes transformer inference faster because`
4. `The strongest argument for using renewable energy is`
5. A Fibonacci function docstring prompt
6. `When the power failed across the city, Maya knew she had only one option:`

Each generation used:

```text
128 new tokens
temperature = 0.8
top_k = 50
```

The decoder samples from the top-k distribution with the stored generation seed.
Prompt text, output text, token IDs, elapsed time, token count, settings, and cache
correctness are stored in `downstream_generations`. The fixed generations are
illustrative only and are not an optimization metric or a substitute for aggregate
benchmark results.

## What cannot be reconstructed

The current artifacts do not preserve per-example benchmark traces. They preserve
aggregate LAMBADA totals/correct counts, aggregate PIQA/ARC correct counts, BLiMP
per-category aggregates, and the full qualitative generations. They do not preserve
the individual dataset prompt, candidate token IDs, candidate log-probabilities, or
winning choice. Consequently, an exact individual benchmark decision cannot be
replayed from SQLite alone without rerunning or changing the evaluator.

## Caveats for interpretation

- This is a project-local evaluator, not `lm-eval-harness`.
- BLiMP uses its `train` split; this must be disclosed.
- Held-out BPB is one fixed sequential prefix of one shard, not the complete
  ClimbMix distribution.
- The original and equal-token evaluator SHAs differ because of the tensor-layout
  fix. Comparisons should report the two provenance scopes separately.
- Standard datasets are pinned by revision, but dataset rows are not copied into the
  database, so row-level provenance is not independently inspectable from SQLite.
- The 200M training campaign is recorded as 201,326,592 tokens, not exactly
  200,000,000.
- LAMBADA uses literal ASCII-space splitting and exact-match requires every final-word
  subtoken to be greedy-correct.
- Accuracy standard errors are computed by the project’s sample-SD-over-square-root-n
  helper, while the suite prose calls for binomial standard errors.
- Qualitative outputs are stochastic, only 18 samples per checkpoint, and
  explicitly illustrative.

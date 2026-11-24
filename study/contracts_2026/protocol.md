# AutoPareto contract study protocol

This file is the frozen protocol for Pair A. It is not injected into the agent
context; the agent receives one of the two byte-level program snapshots in
`programs/` and a campaign-local view. No paper, audit, historical result, or
expected conclusion is part of either program.

## Scientific question

Does changing only the resource contract presented to an autonomous Transformer
researcher change the research trajectory and the models it discovers?

The initial matched pair has 25 attempts per condition in a pre-registered,
single-GPU same-index block-balanced interleaving. Within block `n`, the resource
contract order is randomized, but T always advances `T1,T2,...,T25` and N always
advances `N1,N2,...,N25`. The resource contract is the planned
intervention. No direction of effect is assumed.

## Common constants and starting state

- Pair: `PAIR_A`.
- Search seed: `3`, selected because it is not one of the registered historical
  AutoPareto seeds (`0`, `1`, `42`). A fresh seed-3 baseline must be generated and
  registered before the pair is runnable.
- Starting source: Git commit
  `5ea452c9d86a43ad4bebd507c5872f18249627ce`, `train.py` SHA256
  `b2fb8788f02517af8bc2fcfa059fb799a6974c9050d00e322168ec414fac4fab`.
- Dataset and tokenizer are the protected baseline sources. Their exact hashes are
  written into every future attempt manifest; no attempt may silently change them.
- Agent model: `gpt-5.6-sol`, reasoning level `high`, subject to the same client/API
  configuration in both campaigns.
- Hardware class and software image are the same for both campaigns. The current
  local macOS/CPU checkout is not a valid campaign execution environment.
- The evaluator exposes absolute candidate cached-decode tok/s, bracketed
  registered-baseline tok/s before and after each candidate window, the linearly
  interpolated baseline at each candidate timestamp, the median candidate/
  interpolated-baseline ratio and its uncertainty interval, diagnostic prefill/TTFT, and peak allocated
  inference memory during search. Prefill and TTFT never enter selection.

## Contract T: fixed wall-clock training

Before each scientific execution, the trusted launcher may run exactly eleven
disposable compilation/cache-warm-up updates in a separate process. That process's
model, optimizer, RNG/CUDA-RNG state, dataloader position, counters, and outputs are
discarded. It is not part of the candidate's scientific training. The scientific
execution then starts a fresh process from the registered source and seed, so its
first update is not preceded by untimed learning.

Each candidate receives a `300.000`-second protected training budget.
The same protected timing boundary is used for every candidate:

```text
start: immediately before the first scientific training iteration
stop:  after the first completed optimizer update at or beyond 300.000 seconds,
       before final validation
```

The timing boundary includes synchronized training forward/backward/optimizer work
and any candidate code executed inside the measured loop. It excludes final
validation, checkpoint serialization, protected inference, inference memory
measurement, and post-run provenance serialization. Compilation and warm-up are
performed only in the disposable pre-run process; the fresh scientific process may
reuse its persistent compiler cache but contributes all of its own updates to the
scientific boundary. Steps 0–10 are scientifically executed but excluded from the
historical steady-state timing boundary. Steady-state timing begins at step index
11. Under the repaired protocol those eleven updates occur only in the disposable
process and contribute no scientific tokens, updates, timing, or final state. The
schedule variable is exactly:

```text
progress = min(measured_training_time / 300, 1)
```

The stop check occurs after a complete optimizer update, so the raw elapsed value
may include the final update's bounded boundary overrun. Raw elapsed time and the
budget-boundary time are both recorded; the protected maximum raw overrun is 15
seconds. A result beyond that bound is a failed attempt. The stop check is never
performed in the middle of an optimizer update.

The resulting token count and optimizer-update count are recorded observations, not
selection targets.

The study launcher enforces this boundary outside the candidate process. The parent
process starts a monotonic clock before the fresh process's first training forward,
authorizes continuation only after each completed optimizer update, and closes the
boundary on the first completed update at or beyond 300 seconds. Candidate events
cross only a typed IPC boundary; the parent reconstructs counts and retains
candidate-exported counters as diagnostics. A custom optimizer may use the staged
`contract_update_completed()` hook after its real update. The candidate's elapsed-
time export is diagnostic only. The final update may overrun the boundary by at
most 15 seconds.

## Contract N: fixed counted-token exposure

`N_STAR` is not hard-coded in this repository. It is chosen once by the calibration
procedure below, logged, and then immutable for both campaign registration and all
future attempts under CONTRACT N. Each candidate must process exactly `N_STAR`
counted training tokens, with no partial optimizer step. The same disposable
eleven-update compilation warm-up and fresh scientific process rule applies; warm-up
tokens are never part of the N contract:

```text
progress = min(total_tokens / N_STAR, 1)
```

Final validation, checkpoint serialization, protected inference, and inference
memory measurement are outside the resource contract. The fixed-token loop must
stop after the exact optimizer update that brings `total_tokens` to `N_STAR`.

## Counting and batching

One counted training token is one next-token prediction target in a packed training
sequence. For sequence length `S=2048`, device batch `B`, and accumulation factor
`A`, one microbatch counts `B*S` tokens, one optimizer update counts:

```text
tokens_per_microbatch = DEVICE_BATCH_SIZE * 2048
tokens_per_optimizer_update = DEVICE_BATCH_SIZE * 2048 * accumulation_steps
                           = TOTAL_BATCH_SIZE
microbatches = optimizer_updates * accumulation_steps
```

The common allowed `TOTAL_BATCH_SIZE` set is exactly
`{32768, 65536, 131072, 262144, 524288}`. These are the five distinct values
observed in the registered AutoPareto candidate sources, including the baseline and
the smaller-batch regimes used in the seed-42 campaign. Their least common multiple
is `524288`, so a single `N_STAR` divisible by that quantum is exactly divisible by
every legal total batch size. A candidate may choose `DEVICE_BATCH_SIZE` and
`accumulation_steps` only when their product with 2048 equals a legal total batch
size. Gradient accumulation can therefore change optimizer-update count while the
counted-token exposure remains identical.

The last partial microbatch or optimizer step is forbidden under CONTRACT N. Under
CONTRACT T, all completed microbatches and the observed total token count are
recorded by the protected worker from actual `[B, 2048]` model-forward tensors and
observed optimizer calls; candidate-exported counters are diagnostics only. The
protected stopping boundary decides when the measured clock closes.

## Candidate failure and attempt consumption

Every crash, OOM, timeout, interruption, malformed output, protected evaluator
failure, inference incompatibility, cache-correctness failure, timer/synchronization
failure, resource-validity failure, or other benchmark-invalid state consumes exactly
one attempt. Sampling variance is persisted as an uncertainty interval and never
turns a scientifically valid candidate into an eligibility failure. A training result
that exists but fails a protected post-training gate is preserved as quality-only/error
evidence and pauses the campaign for review. No hidden retries are permitted.

## Protected validation and inference metrics

Validation uses the common protected validation shard, tokenizer, evaluator, and
precision. Validation BPB is the only quality metric visible during search.
The validation shard is excluded from the candidate training data and is not
available to the Scientist during reasoning or candidate training. After the
candidate process terminates, the trusted parent loads the validation examples and
sends them only to the disposable model host over the narrow validation IPC. The
model host returns model outputs; the trusted parent applies the target-byte mask,
cross-entropy sum, BPB formula, acceptance, and provenance. No validation inputs,
raw model-host outputs, or protected evaluator internals are returned to the
Scientist.

Candidate model definitions are loaded only in the disposable inference model
host, never in the trusted evaluator Python process. The evaluator communicates
with that host over an anonymous Unix IPC socketpair and independently owns the
fixed prompts, correctness comparisons, monotonic timing, repetitions,
stability gates, acceptance, and Pareto arithmetic. The model host returns only
typed execution observations. The Design-B semantic workload is unchanged; the
parent decode interval necessarily includes the small IPC request/response
overhead and records this as an explicit timing-boundary limitation. Ordinary
processes cannot conceal input tensors from the process executing the model, and
CUDA allocator values remain model-host observations rather than hardware-
attested facts under the registered non-malicious-CUDA threat model. See
`study/contracts_2026/inference_ipc.md`.

Inference uses protocol `autopareto_search_inference_bracketed_v2`. The canonical
workload is BF16 greedy batch-1, one request at a time, eight fixed protected
prompts of exactly 1024 input tokens and exactly 128 generated tokens. Each request
prefills a fresh state, obtains `y1 = argmax(prefill_logits)`, then times exactly
127 cached `decode_one(y)` transitions from `y1` through `y128`. The authoritative
clock is synchronized host monotonic time; CUDA events are diagnostic only.

Candidate and registered baseline each receive a fixed, discarded 20 seconds of
pure cached decode warm-up. The measured phase has exactly eight temporal brackets:
`baseline_before → candidate → baseline_after`. Each timing window cycles all eight
prompts in order until at least eight seconds of pure cached decode time have
accumulated. For each candidate window, its baseline is linearly interpolated at
the candidate timestamp; the search speed objective is the median of the eight
candidate/interpolated-baseline ratios. Absolute candidate and baseline tok/s,
whole-block bootstrap uncertainty, CV, IQR drift, and raw range remain recorded.
An unstable normalized speed measurement is retained as speed-uncertain evidence,
but cannot claim a throughput Pareto advantage.

Pure prefill latency/input throughput and local TTFT are measured diagnostically on
the same eight prompts. Memory is measured in a clean BF16-only request after FP32
adjudication has been destroyed. Peak allocated CUDA memory is the Pareto objective;
reserved memory, parameter bytes, autoregressive-state bytes, and KV bytes are
diagnostic.

The training launcher invokes this protected benchmark after every successful
training worker and supplies the frozen candidate checkpoint plus candidate-local
definitions. The checkpoint is hashed, the training process is terminated, and a
fresh inference process verifies source/config/hash pairing before validation or
timing. A candidate-defined `build_inference_adapter(model)` is supported; the
protected adapter validator checks its three methods and exact reachable cache
storage before any velocity value can enter the search record.

The inference memory metric is peak allocated CUDA memory during the protected
measurement. Reserved CUDA memory and exact live KV-cache bytes are also persisted,
but reserved memory is not substituted for the Pareto objective.

## Pareto rule and parent selection

The search archive contains the registered same-seed baseline and results from the
current campaign only. A valid candidate is retained exactly when no archive member
is no worse in all three objectives—validation BPB (lower), normalized paired
cached-decode throughput (higher), and peak allocated inference memory (lower)—and it is strictly
better in at least one. Correctness and benchmark validity are mandatory gates;
sampling uncertainty is handled by the conservative speed comparison below. A
candidate parent must be a current frontier member of the same campaign, or the
registered same-seed baseline for attempt 1. A dominated, failed, or other-campaign
result may not be selected as a parent.

## Campaign isolation and sealed metrics

The two campaigns have separate history namespaces, frontier files, attempt
directories, and agent contexts. Each agent sees only its own program, baseline
source, protected candidate interface/runtime, and campaign-local view. It cannot
read existing results, databases, paper/audit Markdown, downstream files, another
campaign, or protected evaluator internals. Held-out BPB, BLiMP, PIQA, ARC-Easy,
LAMBADA perplexity/exact match, transfer results, and all other downstream metrics
are post-search sealed evaluations and never enter the original search history.

The scientist process is launched through a deny-by-default operating-system
sandbox with only the staged context readable and writable. Linux execution uses
the registered Landlock/seccomp/setpriv process boundary inside the provider's
outer container; if that boundary is unavailable, launch fails closed. Candidate
training is additionally isolated in a fresh restricted process created by
`execve`; the trusted launcher never executes candidate source. The exact single-GPU order is
frozen in `provenance/pair_a_interleaved_same_index_schedule_20260902.json`; T and N histories,
views, caches, and feedback remain separate, and the two attempts in one block are
never simultaneous.

## Calibration and freezing N_STAR

After a seed-3 baseline source is registered, run the exact baseline three independent
times under CONTRACT T on the campaign hardware/software image. Record measured
training seconds, total tokens, optimizer updates, training throughput, validation
BPB, and errors for each run. Let `m` be the median of the three successful token
counts and `Q=524288`. Freeze:

```text
N_STAR = floor(m / Q) * Q
```

The command refuses to produce a choice when any run fails or when `N_STAR` would be
zero. It logs the three raw calibration records, the median, `Q`, and the selected
value before either campaign can be registered. No later campaign command can
change it. The registered historical baseline is the only execution permitted to use
the baseline-calibration compatibility flag: its historical schedule predates the
protected progress hook, while all search candidates must call the protected
contract-progress function before each optimizer update.

## Post-search crossover and downstream sealing

Only after both campaigns are complete, an evaluator may take a frozen candidate and
run it under 300-second training, `N_STAR` tokens, and—when legal—a token-matched
condition with matched optimizer-update count. These results live in a separate
post-search namespace and never return to either frontier. Downstream benchmarks are
also separate post-search artifacts.

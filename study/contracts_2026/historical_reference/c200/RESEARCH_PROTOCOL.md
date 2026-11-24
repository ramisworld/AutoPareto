# AutoPareto research protocol

## Question

Can an autonomous coding researcher discover useful transformer designs when it
preserves trade-offs among validation quality, cached inference speed, and inference
memory instead of advancing only a single quality winner?

This is the one authoritative current protocol. Git history, SQLite program snapshots,
hashes, and the completed diagnostic program preserve the exact historical rules used
by earlier campaigns. Completed results are never reinterpreted or rewritten.

## Fixed benchmark

- one NVIDIA A40;
- candidate copies of `train.py` are the only editable scientific source;
- protected dataset, tokenizer, splits, evaluator, prompt, and runner;
- bfloat16 training from scratch;
- 300 measured training seconds;
- fixed training seed within each campaign;
- validation `val_bpb` is minimized;
- official inference uses protected KV-cached greedy decoding;
- cached and full-prefix inference must pass protected common-prefix compatibility
  checks at multiple prefix lengths before any speed or memory value is accepted.

Every crash, OOM, timeout, interruption, incompatibility, or protection failure counts
as an attempt. Training quality and inference compatibility are separate statuses. A
post-training inference failure cannot erase an already produced `val_bpb`.

## Search budget and seeds

Every new search campaign has exactly 25 attempts. Search replicates use seeds 42, 0,
and 1. Each replicate is an independent clean task and may inspect only the frozen
baseline plus its own campaign history.

Confirmation and ablation use seeds 0, 1, and 2. The existing baseline measurements
for those seeds are reused rather than rerun.

## Objectives and methods

The headline deployment objectives are:

1. minimize `val_bpb`;
2. maximize median protected cached-decode tokens/second;
3. minimize protected peak allocated inference CUDA memory in MiB.

AutoResearch edits candidate `train.py`, always branches from its current strict
quality winner, and retains only strict `val_bpb` improvements.

AutoPareto edits candidate `train.py`, chooses an explicit current non-dominated
parent, and preserves the complete non-dominated archive on the three headline
objectives.

Random Search, TPE, CMA-ES, MOTPE, and NSGA-II operate over the same frozen fixed
parameter space. TPE and CMA-ES propose from quality only. MOTPE and NSGA-II use all
three objectives. Fixed-space methods and arbitrary-code agents are reported as
separate tracks; unequal search spaces are never presented as a direct capability
comparison.

No method may inspect another method's proposals or results. All methods use the same
attempt budget, benchmark, hardware, failure accounting, and fixed training seed per
replicate.

Before attempt 1, each campaign is registered in SQLite with an immutable method,
phase, attempt limit, training seed, algorithm seed when applicable, coding model,
reasoning level, program hash, and frozen search-space hash when applicable. Resume
must reject missing or conflicting optimizer state rather than silently reconstructing
different proposals. Research-agent parents may come only from the shared baseline or
the same campaign.

Each AutoPareto campaign additionally registers exactly one valid compatible baseline
experiment with the same training seed. Its frontier, parent eligibility, and
retention decisions contain only that baseline and the campaign's own valid results.
Attempt `N+1` cannot be reserved until attempt `N` is terminal and has a recorded
Pareto decision.

The classical search space fixes `DEVICE_BATCH_SIZE=64` and varies depth, width, head
dimension, `L`/`SL`/`SSSL`/`SSSS` attention patterns, KV-head ratio, total training
batch, embedding/unembedding/matrix/scalar learning rates, weight decay, warm-up,
warm-down, final learning-rate fraction, and Muon Newton–Schulz steps. Its canonical
serialization and SHA-256 are recorded before any classical attempt.

## Sequential agent research

For each AutoResearch or AutoPareto attempt, the agent must:

1. inspect only its permitted history;
2. record one hypothesis, category, exact parent, rationale, and candidate diff;
3. run that candidate exactly once through `scripts/run_runner.sh`;
4. record quality, compatibility, speed and memory when available, failure details,
   visible interpretation, and decision;
5. formulate the next proposal only after the current result is terminal.

Automation may execute and log one current proposal, but cannot pre-plan or queue
future LLM hypotheses or diffs. Hidden chain-of-thought is never requested or stored.
AutoPareto agents inspect history only through the campaign-isolated runner view,
which exposes the registered baseline, that campaign's attempts, and that campaign's
frontier.

## Confirmation and ablation before AutoPareto

The existing study is followed by a preregistered three-seed check of:

- experiment 65 exactly: lowest observed quality, inference-incompatible on seed 42;
- experiment 68 exactly: best fully compatible observed candidate;
- batch-only: the root baseline with only `TOTAL_BATCH_SIZE=2**17`;
- non-batch-only: experiment 68 with the original `TOTAL_BATCH_SIZE=2**19`.

These twelve runs estimate the batch contribution, the non-batch contribution, their
interaction, winner repeatability, and compatibility repeatability. They do not alter
the historical discovery decisions.

## Deployment characterization

The fixed headline benchmark remains unchanged. A separate inference-only workload
matrix characterizes selected compatible checkpoints at context lengths 256, 896,
1,024, and 1,536; inference batch sizes 1, 2, 4, and 8; and 128 generated tokens.
Each workload uses one warm-up and five measured repetitions. It records prefill
latency, cached decode throughput, allocated and reserved VRAM, KV-cache growth,
correctness, evaluator hash, and repetitions. Workload failures and OOMs are recorded
and do not prevent later workloads.

Representatives are selected mechanically: intact baseline seed 1; the median-quality
seed of confirmed experiments 68 and 65 only when that median checkpoint is
inference-compatible; the median seed of the better mean-improving compatible isolated
ablation; and fixed historical experiments 61 and 19. These are supporting
measurements, not additional optimization objectives, and never overwrite the headline
measurements.

## Post-search AutoPareto confirmation selection

Before AutoPareto results are visible, final confirmation representatives are fixed as:

1. the lowest-BPB frontier candidate;
2. the fastest frontier candidate whose BPB is within 1% of the campaign's
   lowest-BPB candidate;
3. the lowest-memory frontier candidate within the same 1% quality budget;
4. the remaining candidate with the largest exclusive three-objective hypervolume
   contribution after objectives are normalized to the registered baseline and the
   observed campaign ranges.

Identical candidate source hashes are deduplicated. If one candidate fills multiple
roles, it is confirmed only once. Representatives are pooled across the three
AutoPareto replicates, deduplicated again, and trained on confirmation seeds 0, 1, and
2. This rule selects important models without inspecting their narrative appeal and
prevents post-hoc cherry-picking. Confirmation and expanded deployment
characterization occur only after all three 25-attempt searches finish.
The frozen implementation is
`scripts/select_autopareto_representatives.py`; its offline validation must pass
before attempt 1.

## Reporting

Report every attempt, failure, exclusion, exact diff, parent, hypothesis, seed,
individual inference measurement, evaluator hash, GPU time, available LLM usage and
cost, raw frontier, confirmed evidence, and limitations. Default presentation may
hide unimportant points, but the complete audit history remains available.

The completed 50-attempt AutoResearch campaign remains valid exploratory history. It
used the earlier cache-compatibility rule and therefore is not treated as a perfectly
matched replicate of new campaigns using the corrected evaluator.

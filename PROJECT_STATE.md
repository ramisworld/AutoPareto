# AutoPareto project state

## Purpose

This opening summary and the newest dated completion sections are authoritative.
Earlier dated sections are retained as historical state transitions and may describe
work that was pending at that earlier time.

Compare frozen AutoResearch, code-editing AutoResearch, multi-objective AutoPareto,
Random Search, TPE, CMA-ES, MOTPE, and NSGA-II under one immutable five-minute A40
training benchmark.
The headline objectives are validation bits per byte (minimize), generated tokens per
second (maximize), and peak inference CUDA memory (minimize).

The controlled scientific question and frozen Phase 1 rules are specified in
`docs/PHASE1_200M_RESULTS.md` and `evaluation/phase1_equal_token_preflight.json`. The
controlled AutoResearch campaign completed all 50 attempts;
its conclusions remain frozen. The separate corrected five-attempt AutoResearch
diagnostic and the preregistered confirmation/batch-ablation study are complete.
AutoPareto discovery campaigns for seeds 42, 0, and 1 are complete. All five seed-42
classical controls are complete: Random Search, TPE, CMA-ES, MOTPE, and NSGA-II. The
registered seed-0/1 classical repeats have not run. The four-recipe, three-seed
AutoPareto primary confirmation is complete. The frozen downstream language suite is
complete for the historical 11-model panel and the 21-checkpoint equal-token panel.
The equal-token panel trained seven recipes at exactly 201,326,592 tokens on seeds
3–5. Stabilized deployment characterization exists with explicit unresolved workloads
preserved. Quantization, cross-hardware replication, and larger-scale transfer remain
unstarted; the canonical forward plan is the roadmap below.
Historical full-prefix inference is archived as `uncached_legacy`; protected realistic
`cached_decode` is official for all measurements made after the July 2026 A40
correctness validation.

This public branch contains the reproducibility source, frozen manifests, aggregate
results, and figures. Paths to private SQLite databases, raw run artifacts, telemetry,
and checkpoint weights are retained in historical notes for provenance but those files
are intentionally not published here.

# AutoPareto Remaining Research Roadmap

This is the canonical current-state and forward research roadmap. Update this section
when a phase changes status or finishes; do not create a competing roadmap. The dated
sections below and `docs/RESEARCH_LEDGER.md` remain evidence/history, not a replacement
for this forward plan. The older quantization-first note in `docs/NEXT_PHASE_PLAN.md`
is retained as an auxiliary options note and is superseded for sequencing by this
roadmap.

## Current scientific state

Established measured facts:

- Three AutoPareto discovery campaigns are complete (training seeds 42, 0, and 1),
  under the original fixed five-minute A40 regime.
- Four selected recipes have confirmation runs: the primary panel retrained the
  exact recipes on confirmation seeds 0, 1, and 2. The important confirmation rows
  include experiment IDs 292, 294, 299, and 289, corresponding to the selected
  quality, trade-off, fast-quality, and low-memory roles. The selection and aggregate
  evidence are frozen in `results/autopareto_primary_confirmation_plan.json` and
  `results/autopareto_primary_confirmation_analysis.json`.
- Downstream evaluation exists for the frozen 11-model historical panel (99 aggregate
  task metrics and 198 fixed-prompt generations) and the 21-checkpoint equal-token
  panel (189 aggregate task metrics and 378 fixed-prompt generations). Stabilized
  deployment characterization also exists for the five final representatives: 66 of
  80 workload pairs are stable and 14 are explicitly unresolved, with those unresolved
  pairs excluded from final speed/memory claims. See
  `docs/PHASE1_200M_RESULTS.md` and
  `results/phase1/public_downstream_metrics.json` in this public branch; the raw
  database and deployment bundles remain private.
- The original experiments used a fixed approximately 300-second training budget on
  one NVIDIA A40. The fixed-time result is therefore a valid **quality-per-A40-time /
  compute-budget result** under that protocol.
- The current important candidates are confirmation experiments 292, 294, 299, and
  289. The most interesting current hypothesis is the shallow-wide trajectory
  represented by discovery experiment `136 -> 137` and confirmation experiment `299`.
  The discovery parent/child relationship and architecture diffs are preserved in
  `results/pareto_frontier.json` and the confirmation analysis.
- The equal-token evidence removes unequal training-token exposure. For the central
  AP292-versus-AutoResearch comparison, total batch size (131,072), device batch size
  (64), and gradient accumulation (1) are also matched. The remaining primary
  confounds are architecture/depth, parameter count, and optimizer/schedule or other
  training-recipe differences; batch size remains unmatched only for comparisons that
  include the baseline.
- The unequal-token exposure is measured, not assumed. In `results/results.db`, same-
  session baseline experiment 4 processed 92,798,976 training tokens. Confirmation
  rows 289, 292, 294, and 299 processed 179,568,640, 159,252,480, 209,715,200, and
  196,345,856 tokens respectively: approximately 1.94x, 1.72x, 2.26x, and 2.12x the
  baseline during equal five-minute runs. Across the selected candidates this is the
  approximately 1.72x-2.26x exposure range that must be controlled before making a
  clean architecture claim.

Interpretation boundary: AutoPareto has discovered reproducible quality/speed/memory
trade-offs under a fixed A40-time budget, and the shallow-wide path is a credible
hypothesis. It is not yet established that the architecture itself is superior.

## Intended final scientific contribution

The target hypothesis is:

> AutoPareto autonomously discovered a shallow-wide Transformer design trajectory that
> may improve the training/inference efficiency frontier. We want to determine whether
> this advantage survives equal-token, parameter-matched, training-recipe-controlled,
> multi-seed, and larger-scale evaluation.

This is a research target, not a current conclusion. The strongest result we are trying
to earn would establish that:

1. the agent autonomously discovered the design trajectory;
2. the result is not merely caused by extra token exposure;
3. it is not merely caused by having fewer parameters;
4. the architecture effect can be separated from the training recipe;
5. the result reproduces across seeds;
6. the principle transfers outside the original approximately 25-50M parameter
   discovery regime; and
7. the deployment advantage survives relevant hardware and workloads.

These are research targets, not current conclusions. Any final paper must state what
was directly demonstrated, what is mechanistic interpretation, what remains limited or
unknown, which hypotheses/parent selections/code changes/frontier discoveries came
from the agent, and which research questions, evaluator protections, verification,
interventions, and later causal-validation designs came from the human researchers.

## CURRENT PHASE

**Phase 1 — Fair equal-token comparison: COMPLETE.**

All 21 preregistered runs (seven frozen models × seeds 3, 4, and 5) completed at
exactly 201,326,592 training tokens. Training, checkpoint, held-out BPB, timing,
stabilized inference, and cache-correctness integrity checks pass. The canonical
machine-readable and paper-readable summaries are
`results/phase1/phase1_equal_token_results.json` and
`results/phase1/phase1_equal_token_results.md`.

The reproducible preflight is recorded in
`evaluation/phase1_equal_token_preflight.json` and
`docs/PHASE1_EQUAL_TOKEN_PREFLIGHT.md`. Exact token-stream equivalence is statically
audited in `evaluation/phase1_token_stream_audit.json`; a CPU runtime hash must be
written before Stage B.

The pre-Stage-B deployment-timing correction is recorded separately in
`evaluation/phase1_stabilized_inference_rebenchmark.json` and
`results/phase1/phase1_stabilized_inference_rebenchmark.md`. Phase 1's old
post-training evaluator used the legacy fixed-two-warmup path. The corrected
inference-only sessions reused the frozen `autopareto_inference_stabilized_v2` gate:
five-run adaptive warm-up blocks, adjacent median drift at most 2%, within-block CV at
most 5%, a 30-warm-up cap, and three ten-repetition stabilized passes. Historical
discovery/confirmation values remain unchanged and are not relabeled as stabilized.
The first session and its versioned retry are preserved under
`results/phase1/stabilized_inference/`; only MOTPE completed all three passes in the
retry. AutoPareto 299 showed reproducible high-throughput stable passes but failed its
final pass gate, so those pre-Stage-B observations remain historical diagnostics. The
correction changed no model or training recipe. It is now superseded for Phase 1
deployment claims by the completed per-seed equal-token checkpoint measurements in
`results/phase1/phase1_equal_token_results.json`.

## Phase 1 — Fair equal-token comparison

**Priority:** ESSENTIAL — next phase.
**Status:** `COMPLETE`

**Question:** When important models process exactly the same number of training tokens,
what happens to quality, training time, and deployment efficiency? This controls the
major confound in the original five-minute experiments.

**Frozen preflight panel (human approval required before launch):**

- untouched original `train.py` baseline, source experiment 4;
- strongest frozen AutoResearch-compatible representative, source experiment 68;
- strongest frozen TPE quality representative, source experiment 183;
- strongest frozen MOTPE quality representative, source experiment 234;
- AutoPareto confirmed-quality recipe corresponding to confirmation 292, discovery
  source 134; and
- AutoPareto shallow-wide / fast-quality recipe corresponding to confirmation 299,
  discovery source 137; and
- AutoPareto efficiency-boundary recipe corresponding to confirmation 294, discovery
  source 136.

All seven models are frozen into Stage B before observing any Stage B result. The exact
common budget is **201,326,592 training tokens**. Common immutable milestones are
`50,331,648`, `100,663,296`, `150,994,944`, and `201,326,592` tokens. Fresh
pre-registered seeds are 3, 4, and 5; the preflight verified that these seeds were
not used to select the AutoPareto recipes.

**Method:** Frozen architectures and optimizer hyperparameters are retrained under a
common token-normalized training horizon. The original wall-clock LR/weight-decay
progress is translated deterministically to token progress while preserving each
recipe's LR values, warmdown ratios, final-LR fractions, optimizer groups, and other
frozen hyperparameters. The original step-based Muon momentum ramp remains frozen as
an explicit remaining recipe difference. All seven sources use the same deterministic
training-token loader sequence; this is statically audited and runtime-hashed before
Stage B. Immutable milestone checkpoints record byte sizes, SHA-256 hashes, exact token
counts, optimizer steps, and pure measured training time. Milestone BPB is evaluated
after training; the final checkpoint additionally receives held-out BPB and protected
cached-decode correctness/throughput/VRAM/KV measurements.

The preregistered primary outcomes are final held-out BPB at exactly 201,326,592
tokens, steady-state-equivalent pure measured A40 seconds normalized to that token
count, and final protected cached-decode throughput under the frozen headline
protocol. The observed sum from the first through final optimizer step is retained as
a secondary full-horizon training-time metric. Supporting outcomes are milestone
validation BPB, allocated/reserved VRAM, incremental KV memory, cache correctness, a
same-excluded-token timing sensitivity, and only pre-declared non-cherry-picked
time-to-quality thresholds.

Scientific timing disables per-step CUDA-event instrumentation because it would add
unequal overhead to the 384-step baseline and 1,536-step accumulation-1 regimes.
Telemetry never imports PyTorch or calls CUDA runtime APIs and uses a fixed
2.0-second NVML/host sampling interval. The initial 0.5-second smoke was retained as
an excluded diagnostic and failed the preregistered telemetry-overhead gate for the
short accumulation-1 workload. The revised interval was validated before the
scientific panel ran. Training and BPB evaluation are separated from stabilized
inference; inference instability leaves training/BPB valid but blocks deployment
claims.

The historical execution plan was staged: a clearly excluded non-scientific smoke
test, then the complete seven-model seed-3 panel (Stage B), then—only after integrity
review—seeds 4 and 5 (Stage C). That plan has now completed; see
`results/phase1/phase1_equal_token_results.json`.

Equal tokens control data exposure. In this panel, AP292 and AutoResearch also share
total batch size 131,072, device batch size 64, and gradient accumulation 1. The
baseline uses total batch size 524,288 and accumulation 4, so batch size remains a
known limitation for baseline-inclusive comparisons. Architecture/depth, parameter
count, gradient noise, learning rates, and other recipe differences remain to be
isolated causally.

**Why this phase exists:** It tests whether the observed fixed-time advantage survives
when every model receives the same amount of training data, while preserving the
quality-per-time question through wall-clock measurements.

**Stop condition:** Do not proceed automatically to Phase 2. Review and interpret the
Phase 1 results before fixing the Phase 2 design.

## Phase 2 — Causal architecture / recipe control

**Priority:** ESSENTIAL if Phase 1 leaves an interesting AutoPareto advantage.
**Status:** `NOT STARTED`

**Main question:** Is the shallow-wide result caused by architecture itself, or simply
by smaller parameter count and/or the changed training recipe?

Construct a conventional Transformer control approximately parameter-matched to the
important shallow-wide AutoPareto candidate, likely 299. Compare a conventional,
deeper-narrower model with the AutoPareto shallow-wide model at comparable parameter
count and the same token budget.

Separate architecture from training recipe with this pre-registered 2x2 design:

| Architecture | Baseline-style recipe | AutoPareto-style recipe |
| --- | --- | --- |
| Conventional parameter-matched | A | B |
| Shallow-wide parameter-matched | C | D |

Use multiple registered seeds. Do not tune one treatment after seeing another
treatment's results.

**Why this phase exists:** It estimates the causal contribution of shallow-wide
allocation, tests whether comparable parameter count preserves quality, decomposes BPB
changes into architecture versus batch/LR/training-recipe effects, and tests whether
systems improvements exceed what parameter reduction alone explains.

## Phase 3 — Scale-transfer experiment

**Priority:** HIGH — only run if Phase 2 supports a real architectural principle.
**Status:** `NOT STARTED`

**Question:** Does the discovered shallow-wide principle work outside the small
approximately 25-50M discovery regime?

Start with approximately 100M parameters, not 300M immediately. Construct a
conventional approximately 100M control and a parameter-matched shallow-wide
approximately 100M treatment. Keep constant training tokens, dataset, tokenizer,
optimizer/training recipe, seed, KV-head policy unless explicitly studied, and the
primary A40 environment.

First run a small canary to verify A40 memory fit, compilation, training throughput,
checkpointing, and cached-inference correctness. Only if promising should the properly
registered multi-seed experiment run. Do not automatically proceed to 300M; that scale
is optional/high priority only after a meaningful 100M result and if GPU budget permits.

**Why this phase exists:** It tests whether the candidate principle transfers beyond the
regime in which it was discovered, rather than treating a small-model result as a
general scaling law.

## Phase 4 — Cross-hardware deployment validation

**Priority:** OPTIONAL / MEDIUM-HIGH.
**Status:** `NOT STARTED`

No retraining should normally be required. Benchmark identical frozen checkpoints on
available hardware: A40 as the primary scientific hardware, RTX 3090 as an available
secondary, and H100 only if a substantially different GPU architecture is accessible.
Measure prefill latency, cached decode throughput, allocated/reserved VRAM, KV-cache
growth, and workload stability.

**Why this phase exists:** It tests whether the architecture ranking survives different
hardware. Do not claim hardware-independent speedup unless this phase supports it.

## Phase 5 — Final downstream/statistical validation

**Priority:** HIGH.
**Status:** `NOT STARTED`

Run only on the final scientifically important models, not every experimental
checkpoint. Reuse the existing frozen downstream suite where appropriate: held-out BPB,
LAMBADA, BLiMP, PIQA, and ARC-Easy. Calculate appropriate uncertainty and multi-seed
summaries. Do not introduce a large benchmark expansion unless a specific scientific
question requires it.

**Why this phase exists:** The existing downstream evaluation is useful evidence for
the current fixed panel, but final causal/scaled models still need statistical and
capability validation. The goal is to determine whether efficiency improvements preserve
useful language capabilities rather than merely improving the search objective.

## Phase 6 — Paper and public artifact

**Priority:** ESSENTIAL.
**Status:** `NOT STARTED`

Once the decisive experiments are complete, stop searching for improvements. Produce
the AutoMLR paper around the evidence that survived controls. Clearly distinguish:

- demonstrated results;
- mechanistic interpretation;
- unresolved limitations;
- the autonomous contribution (hypotheses, parent selections, code changes, and
  frontier discoveries from the agent); and
- the human contribution (research question, protected evaluator, verification,
  interventions, experiment design, and later causal validation).

Potential final story, only if supported by the results:

> Under a constrained single-A40 autonomous research budget, AutoPareto discovered a
> shallow-wide Transformer design trajectory that shifted the
> quality-training-efficiency-inference-efficiency frontier. Controlled equal-token and
> parameter-matched experiments isolated the effect, with subsequent scale and hardware
> tests determining where the principle transfers.

This wording is conditional and must not be presented as established before the
experiments establish it.

## Deferred / optional research

### Full missing classical-search campaigns

Low priority for the current submission. They would require approximately 250
additional five-minute search attempts at high GPU cost, and AutoPareto and classical
methods currently have unequal search spaces. Completing all missing seeds therefore
would not cleanly prove that AutoPareto is a universally better search algorithm.

### Quantization

Low priority for the current scientific story. It is a useful deployment extension but
is unlikely to resolve the central equal-token, parameter, recipe, or architecture
questions.

### 300M scale transfer

Conditional. Run only after a successful 100M result and only if the budget supports it.

### Full H100 retraining/search

Not currently necessary. Cross-hardware inference benchmarking is much cheaper and more
scientifically targeted.

## Progress tracking and evidence discipline

Use only these phase statuses: `NOT STARTED`, `PREPARING / PREFLIGHT`, `DESIGNING`,
`RUNNING`, `COMPLETE`, and
`DEFERRED`. The expected starting statuses are:

| Phase | Status |
| --- | --- |
| Phase 1 — fair equal-token comparison | `COMPLETE` |
| Phase 2 — causal architecture / recipe control | `NOT STARTED` |
| Phase 3 — scale transfer | `NOT STARTED` |
| Phase 4 — cross-hardware deployment validation | `NOT STARTED` |
| Phase 5 — final downstream/statistical validation | `NOT STARTED` |
| Phase 6 — paper and public artifact | `NOT STARTED` |

Every update must label statements as one of: existing measured fact, current
hypothesis, planned experiment, or conditional experiment. Link or name the relevant
repository artifact, experiment ID, database table, or documentation section where
useful. In particular, update this canonical roadmap when future work finishes rather
than creating a competing roadmap document.

## Locations and provenance

- Canonical local Git repository: `/Users/Rami/Documents/dev/autopareto`
- Remote CUDA working directory: `/workspace/autopareto`
- Last recovery SSH endpoint: `1y7yjg1u969jup-64411310@ssh.runpod.io`; it is not a
  permanent endpoint and must remain stopped after artifact synchronization.
- SSH identity path: `~/.ssh/id_ed25519` (never inspect, copy, or modify the key)
- Original repository: `https://github.com/karpathy/autoresearch.git`
- Original AutoResearch commit: `228791fb499afffb54b46200aca536f79142f117`
- Current branch at the time of the Phase 1 work: `autopareto/research-infrastructure`
- Shared A40 starting baseline commit: `5ea452c9d86a43ad4bebd507c5872f18249627ce`
- Seed-0 project-state snapshot commit: `c119eeb235e2556d9bb956443398feb33108ec95`
- New-Pod endpoint update commit: `fe87792b9632bd9ec3b7d1b0b9e46f132cb24169`
- Cached evaluator and scientific schema commit: `da4da1c`
- Frozen controlled research protocol commit: `6761228`
- Live/replay dashboard commit: `6b4af1d`
- Editorial dashboard redesign commit: `ad5884bb4209097c11862010c8130e9000217598`
- Presentation exit and stale-state fixes: `8b873c6640b8e4fb0edda2387331facb64227518`
- Premium research-console visual system: `7cb805a7fcf03700567f79b5b9e8f47895939eb2`
- Guarded cached checkpoint remeasurement commit: `a0c5412`
- A40 correctness-tolerance calibration commit: `357c390d2feae916b418e258464cd152439a6523`
- Cached rebenchmark repetition-storage fix: `c1af0ef8ec22a2e9338d021f48aae9137aae4576`
- Synchronized evaluator-commit recording fix: `bf0c1f6894cdb2fe9cb7f84612e7456e650abb92`
- Fixed evaluator-byte preservation and runner-side metadata fallback: `9e1650df29104a7ff6015f1a6bfcac1e694d4389`
- Pre-AutoResearch readiness and truthful metadata normalization: `29dde211`
- Active A40 endpoint and guarded fresh-Pod result restoration: `e657a6739e3e9bbb268a84d8de47c0682250d380`
- Attempt-6 readiness checkpoint: `a2680be0096698894f54a22d60f0dc2d4eeff104`
- Five-attempt diagnostic preparation checkpoint: `75af7b7ec8d170e59aaac5b68fa154c5a5117a5b`

## Architecture

`runner.py` is the authoritative orchestrator and SQLite writer. Each attempt is
reserved in SQLite before GPU work, copied into `worktrees/<campaign>-<number>`, guarded
by a filesystem GPU lock, run through the protected checkpoint launcher, evaluated by
the protected inference benchmark, and saved immediately. Candidate isolates are
removed even after failure. Detached remote campaigns are launched with tmux through
`scripts/run_remote.sh --tmux` so an SSH disconnect does not terminate them.

For future `runner.py run` invocations, use the non-protected
`scripts/run_runner.sh` wrapper. It registers the standard SQLite conversion from
the argparse `pathlib.PosixPath` received by `--program` to text before loading the
unchanged protected runner. `scripts/run_runner.sh --metadata-preflight` performs the
same metadata write against a disposable database copy and never uses SSH, CUDA, or a
GPU. This prevents the pre-GPU SQLite binding failure that interrupted attempt 1.

The shared A40 starting `train.py` differs from upstream only in
`DEVICE_BATCH_SIZE = 64` instead of 128; `TOTAL_BATCH_SIZE` and the architecture are
unchanged. `protected/train_launcher.py` overrides its hard-coded seed call at runtime,
then saves the final model state and training metrics.
`inference_benchmark.py` reconstructs the candidate architecture through a stable
interface. The starting model uses the protected canonical adapter; an architecture
that changes that layout may define `build_inference_adapter(model)` in its candidate
`train.py`, while the protected harness retains control of the prompt, correctness
test, timing, repetitions, and metrics. Correctness compares 16 deterministic
teacher-forced next-token steps on identical prefixes, requires greedy-token agreement,
and bounds mean absolute logit error at 0.03. This avoids treating harmless bfloat16
kernel drift as 256 cascading failures. Official measurement still performs two full
warm-ups and ten measured cached repetitions and records prefill, median cached speed,
variation, allocated/reserved memory, and incremental KV cache.

The additive SQLite schema now separates campaigns, proposals, training runs,
individual inference measurements, confirmation, ablation, visible agent actions, and
timestamped replay events. Sparse training telemetry is queued every five steps and
written asynchronously so the measured loop does not perform synchronous SQLite work.
Training metrics, `val_bpb`, checkpoint path, and candidate archive are committed
immediately after training, before inference begins. Top-level training and inference
statuses are separate. An inference failure therefore leaves quality evidence intact
but excludes the incomplete three-objective row from Pareto analysis. AutoResearch
decision logging rejects any decision that violates strict parent-relative `val_bpb`
retention.
The dashboard merges committed SQLite events with live JSONL events. Its current warm
editorial UI has four modes: focused live research, completed-result exploration,
plain-language evidence, and distraction-free 1920x1080 presentation replay. Replay
has labelled phase segments and a phase jump selector instead of one unstructured
global slider. Live and replay share one deterministic event reducer, and the synthetic
demonstration remains JSON-only and visibly labelled as non-research data.

Classical methods share only the frozen discrete knobs in `search/common.py`. They
materialize candidate copies by changing those assignments and cannot rewrite
arbitrary code. Optuna supplies genuine Random, TPE, CMA-ES, multi-objective TPE, and
NSGA-II samplers. TPE and CMA-ES minimize quality only; MOTPE and NSGA-II optimize
quality, cached speed, and allocated inference memory.

## Protected files

- `prepare.py`
- `benchmark_config.py`
- `inference_benchmark.py`
- `autopareto_db.py`
- `event_log.py`
- `runner.py`
- `protected/inference_adapter.py`
- `protected/train_launcher.py`
- `protected/smoke_gpu.py`

The runner hashes these files in the canonical tree and every isolate immediately
before and after GPU execution. Only an isolated `train.py` is replaced. A local
mutation probe has passed: an altered disposable `prepare.py` was rejected while the
canonical SHA-256 remained `4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150`.

## Fixed benchmark

- GPU: one NVIDIA A40 (46,068 MiB reported)
- Upstream dataset/tokenizer/splits/evaluator: protected `prepare.py`
- Training budget: 300 measured training seconds
- Precision: bfloat16
- CUDA allocator: `cudaMallocAsync` for all measured subprocesses (fixed shared
  fragmentation control; it does not change candidate source or numerical precision)
- Inference input: deterministic protected sequence of exactly 256 tokens
- Generation: greedy, exactly 256 new tokens
- Warmup: two complete 256-token cached generations
- Inference timing: official mode is protected KV-cached decode, with CUDA
  synchronization around prefill and decode, two full warm-ups, ten repetitions, and
  a deterministic full-prefix/cached compatibility check on common prefixes
- Quality: upstream validation `val_bpb`; generated text is inspection-only

## Experiment schema

SQLite table `experiments` records: campaign ID, experiment number, method, phase,
seed, pipeline status, separate training/inference statuses, `valid_result`,
`exclude_from_analysis`, description, Git commit,
unified code diff, `val_bpb`, inference tokens/s,
peak inference CUDA MB, training tokens/s, total training tokens, peak training CUDA MB,
parameter count, training/inference/total runtimes, checkpoint/sample/log paths, failure
reason, protected-manifest digest, and start/finish timestamps. `(campaign_id,
experiment_number)` is unique. `campaign_state` stores attempt limits and resume state.
Crashes, OOMs, timeouts, protection failures, and interrupted campaign rows are terminal
attempts. Setup rows have `valid_result=0` and `exclude_from_analysis=1`; Pareto and
scientific queries require valid, non-excluded rows.

Additive normalized tables record campaign/model/program metadata, parent hypotheses
and visible agent actions, per-seed training runs, every inference repetition,
confirmation and ablation mappings, timestamped live/replay events, and immutable
legacy metrics. `results/pareto_frontier.json` is now the raw seed-42 cached frontier;
`results/confirmed_pareto_frontier.json` requires completed seeds 0–2;
`results/quality_budgets.json` reports 0.5%, 1%, 2%, and 5% confirmed budgets relative
to the confirmed AutoResearch quality winner.

## Preregistered next study

- Every new search campaign has 25 counted attempts.
- Independent search seeds are 42, 0, and 1; no method may inspect another replicate.
- Confirmation and ablation use seeds 0, 1, and 2.
- The twelve pre-AutoPareto runs are experiment 65 exact, experiment 68 exact,
  batch-only, and experiment 68 with the original batch restored.
- AutoPareto remains a standalone LLM code-editing method. No hybrid optimizer is
  included in the primary study.
- The dashboard remains three-dimensional but defaults to the non-dominated frontier;
  all audit rows remain available.
- Supporting inference workloads vary context length and batch size without changing
  the three headline objectives or overwriting official measurements.

## 2026-07-26 local completion repair

- Campaign registration now freezes method, phase, attempt budget, fixed discovery
  seed, algorithm seed, coding model, reasoning level, program hash, and search-space
  hash before attempt 1. Conflicting resumes and skipped experiment numbers are rejected.
- AutoResearch and AutoPareto research parents are restricted to the shared baseline or
  their own campaign. AutoResearch quality decisions and AutoPareto campaign-local
  Pareto decisions are mechanically checked.
- Confirmation and ablation mapping rows now receive terminal status and failure
  details. The exact experiment-65, experiment-68, batch-only, and non-batch-only source
  hashes are checked before execution.
- The frozen classical space now includes `SSSS`, KV-head ratio, final learning-rate
  fraction, and Muon Newton–Schulz steps. Optuna study metadata is immutable, interrupted
  trials are reconciled, missing optimizer state is rejected, and NSGA-II uses an
  eight-trial population so evolution occurs within 25 attempts.
- Deployment measurements are exposed through the dashboard with context and
  inference-batch selectors, supporting latency, decode speed, allocated memory, and
  KV-cache growth. Experiment 65 has a separate quality-only/incompatible presentation.
- The pre-repair authoritative database is preserved at
  `results/archive/2026-07-26-before-local-completion-repair.db`.

## Completed work

- Local target created and upstream cloned.
- Exact upstream commit captured and new branch created.
- RunPod SSH, A40, OS, Python, CUDA, storage, Git, uv, and `/workspace` verified.
- SQLite runner, checkpoint launcher, fixed inference benchmark, Pareto export, locking,
  isolated copies, timeouts, failure recording, resume behavior, and protection hashes implemented.
- The current fixed classical parameter space and offline validation cover Random
  Search, TPE, CMA-ES, MOTPE, and NSGA-II. New campaigns use exactly 25 attempts.
- Current controlled AutoResearch and AutoPareto instructions require exactly 25
  attempts; historical 50-attempt instructions remain preserved by Git and SQLite.
- PTY-safe push, pull, remote command, tmux, and five-seed baseline scripts prepared.
- Local protection mutation probe passed.
- First remote smoke attempt failed before CUDA because `protected/smoke_gpu.py` lacked
  the repository root on `sys.path`; the import path was corrected. The rerun passed on
  the NVIDIA A40 with a `[1, 64, 8192]` output and 228.3 MiB peak allocated memory.
- Remote protection mutation probe passed with the same canonical `prepare.py` hash.
- The first unchanged seed-0 baseline attempt used the upstream native allocator and
  failed as an OOM after 132.7 seconds (43.98 GiB in use, 3.91 GiB reserved but unused,
  then a 4 GiB request). It is retained in SQLite as an OOM experiment. A fixed shared
  `cudaMallocAsync` allocator was then selected to test whether the failure was only
  fragmentation. The unchanged retry also OOMed after 125.2 seconds. Both are setup
  records and will not count as baseline seeds or campaign attempts.
- The only shared training-source compatibility change is lowering
  `DEVICE_BATCH_SIZE` from 128 to 64 while retaining `TOTAL_BATCH_SIZE = 2**19`.
- The project was restored onto the replacement RunPod. Its NVIDIA A40 (46,068 MiB),
  Ubuntu 24.04, Python environment, CUDA 12.8, Git, uv, `/workspace`, protected hashes,
  and authoritative SQLite state were reverified before GPU work.
- The replacement Pod had no dataset/tokenizer cache, so protected `prepare.py` ran once:
  all 10 training shards plus the pinned validation shard were downloaded, the fixed
  8,192-token tokenizer was built, and its sanity check passed.
- A fresh remote protected-file mutation probe passed and rejected an altered disposable
  `prepare.py`. The CUDA smoke test also passed on the replacement A40.
- All five A40-compatible baseline seeds (0 through 4) have completed the full training,
  validation, checkpoint, fixed 256-input/256-output inference, and SQLite pipeline.
  Seeds 1 through 4 were run sequentially in the resumable tmux session
  `baseline-a40-seeds-1-4`; no research or classical-search campaign was started.
- A local-only scientific refactor added the frozen research protocol, additive schema,
  protected KV-cache adapter/evaluator, raw/confirmed analysis, visible agent/event
  records, and live/replay dashboard. The five original uncached speed/memory/timing
  values were copied into `legacy_metrics` and tracked audit snapshots before the
  original columns were labelled `uncached_legacy`; no value or validity flag changed.
- A local-only dashboard redesign replaced the crowded single dark page with four
  progressively disclosed modes. It added deterministic replay selection, recorded
  hypothesis typewriter presentation, six event-driven stages, noisy raw plus smoothed
  training loss, all ten cached inference repetitions, offline 3D/2D frontier views,
  confirmation/ablation evidence, and a 16:9 presentation mode. No scientific backend,
  result row, protected benchmark, training behavior, or campaign instruction changed.
- A follow-up dashboard usability audit added persistent Presentation-mode exit via a
  visible button, `Escape`, and browser history; disabled replay controls when no events
  exist; removed misleading cached-speed units and stage labels from the pre-validation
  authoritative state; preserved the synthetic source across navigation; and made
  Explore selection follow active filters. This was UI-only and changed no results.
- The same audit found that authoritative dashboard polling used the general
  migration-capable database connector and did not close it. Dashboard snapshots now
  use short-lived SQLite `mode=ro` plus `query_only=ON` connections; the authoritative
  experiment writer and schema remain unchanged.
- A presentation-only visual pass restyled every dashboard mode as a minimal research
  console: warm technical grid paper, black system chrome, monospace navigation,
  telemetry and headlines, sharp low-radius geometry, a dark agent-proposal surface,
  and restrained orange/green signal states. Information architecture, event handling,
  SQLite reads, scientific measurements, and campaign behavior were not changed.
- Before replacing the baseline display, the seven historical rows were backed up to
  `results/archive/2026-07-22-before-fresh-baselines.db`. Their three campaigns were
  marked `dashboard_visible=0` with an archive reason. No experiment row, metric,
  validity flag, exclusion flag, log, or checkpoint reference was deleted or changed.
  The dashboard API now omits archived campaigns and their events. Replay repetition
  counts reset per attempt, baseline labels use seeds, baseline lineage has one frozen
  root, and the orange dashed 3D line is explicitly labelled as the non-dominated
  frontier. The baseline command now stops the suite on the first failed seed and
  records candidate provenance separately from evaluator provenance.
- Post-run migration tests found that 12 cached headline values (three fields for fresh
  seeds 0--3) had been duplicated into the audit table under the wrong
  `uncached_legacy` label. No headline or experiment row was wrong. The pre-correction
  database is preserved at
  `results/archive/2026-07-22-before-legacy-label-correction.db`; those duplicate audit
  records are retained as `cached_decode_migration_duplicate`, the 15 genuine legacy
  records remain `uncached_legacy`, and cached rows are now excluded from future legacy
  archival.
- Live telemetry was corrected without modifying `train.py` or any scientific rule.
  Future events include cumulative tokens, peak CUDA allocation, and the exact final
  step. The chart plots the trainer's debiased EMA cross-entropy as measured dots instead
  of calling it raw loss or smoothing it twice. Gradient norm remains uninstrumented.
  Future aggregate training throughput now matches post-warm-up tokens to the existing
  post-warm-up timer; recorded baseline rows remain unchanged. A PTY-safe read-only
  snapshot relay supports genuine local live viewing despite rejected TCP forwarding.
- The loss chart now has fixed axes (`0–5` timed minutes, loss `2–10`) so replay never
  changes its visual scale. Compilation/warm-up steps 0–10 are summarized separately,
  out-of-range timed values remain explicit, and the remaining-time formatter now uses
  correct `minutes:seconds` arithmetic rather than rounding minutes upward.
- A local readiness audit corrected two operational labels without changing any metric:
  warm-up inference rows now use null (“not evaluated”) rather than false for correctness,
  and completed `campaign_state` records synchronize the descriptive campaign status to
  completed. All 50 measured cached baseline repetitions passed correctness.
- The pre-AutoResearch local preflight passed ten consecutive full-suite test loops, Python/JS/
  shell syntax checks, SQLite integrity and foreign-key checks, protected-mutation
  detection, exact `train.py`/`prepare.py` shared-baseline hashes, and runner rejection
  of seed 41, attempt 51, and missing proposal metadata. No GPU or SSH was used.
- The final GPU gate restored the repository, authoritative 12-row database, locked
  dependencies, and fixed dataset/tokenizer cache on the active NVIDIA A40 Pod. All
  14 remote tests, the protected mutation probe, and the CUDA smoke test passed; the
  smoke output was `[1, 64, 8192]` with 228.3 MiB peak allocated memory.
- One unchanged seed-42 discovery-reference baseline completed the full protected
  pipeline. It is a baseline row and consumes no AutoResearch or AutoPareto proposal.
  Real append-only events were relayed to the local dashboard throughout. A gateway-only
  relay bug was fixed by invoking the bridge as a Python module and queueing `exit`
  after its long-running command; this does not touch GPU timing.
- AutoResearch campaign `autoresearch-seed42-20260723` completed exactly 50 counted
  attempts using discovery seed 42 and the unchanged protected runner/evaluator.
  Nineteen attempts completed with valid `val_bpb`, 29 crashed, and two were
  interrupted. Every row records a terminal `retained` or `discarded` decision.
- The greedy retained chain is attempts 4, 5, 6, 9, 13, 15, 16, 19, 21, and 48.
  Attempt 48 (database id 61) is the final quality winner at
  `val_bpb=1.096754145494427`. Its only new change over retained attempt 21 is Muon
  `ns_steps=6`; the stored candidate also contains all prior retained changes.
- Attempt 1 remains an excluded pre-GPU metadata interruption. Attempt 23 reached its
  final training-progress event but was interrupted when checkpoint accumulation
  exhausted the project-volume quota; it has no valid result and was discarded. The
  database and WAL were recovered without dropping a row, the checkpoint tree was
  moved intact to `/tmp/autoresearch-checkpoints`, and
  `results/checkpoints` is a remote symlink to that directory. The final authoritative
  database is again a regular project-volume file. At finalization the remote working
  directory no longer contained `.git` metadata after the quota incident; the canonical
  local Git repository is intact, and attempts 6 through 50 retain readiness commit
  `a2680be0096698894f54a22d60f0dc2d4eeff104` in SQLite. Attempt 23's top-level
  experiment is terminal, but its subordinate `training_runs` row preserves the
  pre-interruption `running` value; no post-hoc scientific metadata rewrite was made.
- Of the 29 crashes, 27 were protected cached/uncached correctness failures and attempts
  41 and 42 were early assertion failures. Failed attempts consumed their numbers and
  were never eligible for retention. Speed and memory were recorded for valid results
  but were not used in any decision.
- SQLite marks the campaign `completed`, `next_experiment_number=51`, and
  `attempt_limit=50`. Integrity and foreign-key checks pass; all 50 decisions and
  visible proposal/action records are synchronized locally. No AutoPareto, Random
  Search, TPE, CMA-ES, confirmation, ablation, or other campaign was started.
- Corrected diagnostic campaign `autoresearch-diagnostic5-seed42-20260724` completed
  exactly five sequential proposals from retained experiment 61 under
  `programs/program_autoresearch_diagnostic_5.md`. Every attempt used seed 42, one A40,
  exactly one runner execution, and a one-line `FINAL_LR_FRAC` change. All five
  completed training and recorded quality. Attempts 1 and 2 were retained by strict
  parent-relative `val_bpb`; attempts 3 through 5 were discarded.
- Diagnostic attempt 1 completed training at log-rounded `val_bpb=1.095215`, then the
  project-volume quota truncated its checkpoint at exactly 134,217,728 bytes, zeroed
  the metrics JSON, and caused a SQLite WAL disk-I/O failure before inference. It was
  not rerun. The intact WAL was checkpointed off-volume, the original DB/WAL/SHM were
  archived remotely, the auditable training log and telemetry were used to recover the
  terminal row at their recorded precision, and the strict quality rule retained it.
  The local pre-metadata-recovery snapshot is
  `results/archive/2026-07-24-diagnostic-attempt1-before-metadata-recovery.db`.
  The checkpoint directory was restored as the documented symlink to
  `/tmp/autoresearch-checkpoints`.
- Diagnostic attempt 2 completed training at `val_bpb=1.0948561434742032` and was
  retained. Its corrected teacher-forced cache check preserved greedy-token agreement,
  but maximum mean absolute logit error was `0.0341237`, above the preregistered `0.03`,
  so inference is `incompatible` and speed/memory are unavailable.
- Diagnostic attempts 3, 4, and 5 completed compatible cached inference but had worse
  quality than retained attempt 2, so all were discarded. Their median cached speeds
  were respectively `163.95254015579656`, `163.11501283527514`, and
  `161.4235396941305` tok/s; each used `249.88385009765625` MiB peak allocated
  inference memory. These secondary measurements did not affect decisions.
- The diagnostic campaign records three visible actions for each attempt and terminal
  decisions for all five. Available token/API cost fields are null; hidden
  chain-of-thought was neither requested nor stored. No confirmation, ablation,
  AutoPareto, Random Search, TPE, CMA-ES, or other experiment was started.

## Current results

SQLite has exactly 68 experiment rows. Rows 1 and 2 remain invalid, excluded setup
OOMs. Rows 3 through 7 are the preserved historical baseline suite. Those seven rows
belong to three archived campaigns and are intentionally absent from normal dashboard
views. Rows 8 through 12 are the current visible fresh baseline campaign
`baseline-a40-cached-fresh-20260722`; all five completed with `valid_result=1`,
`exclude_from_analysis=0`, empty code diffs, and frozen candidate Git commit
`5ea452c9d86a43ad4bebd507c5872f18249627ce`. Their evaluator commit is
`cc029fdb4381b32012d01bb4df25c08183f03cf7`.

| Seed | `val_bpb` | Inference mode | Median cached tok/s | Speed stdev | Prefill ms | Allocated MiB | Reserved MiB | KV growth MiB |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.2137349928434058 | `cached_decode` | 164.11263064330285 | 1.897115207107037 | 6.193792447447777 | 249.88385009765625 | 352.0 | 3.984375 |
| 1 | 1.2189262480600007 | `cached_decode` | 151.87496681853105 | 2.740314276608005 | 6.667674519121647 | 249.88385009765625 | 352.0 | 3.984375 |
| 2 | 1.2169696961309011 | `cached_decode` | 161.99222320432617 | 3.2760288989580095 | 6.22141920030117 | 249.88385009765625 | 352.0 | 3.984375 |
| 3 | 1.1966832149933828 | `cached_decode` | 160.81916285396 | 15.254324076324787 | 6.630244664847851 | 249.88385009765625 | 352.0 | 3.984375 |
| 4 | 1.213109342994092 | `cached_decode` | 153.90597321109982 | 3.2862487543208285 | 6.852209568023682 | 249.88385009765625 | 352.0 | 3.984375 |

All five fresh models have 50,332,176 parameters and exactly two warm-ups plus ten
measured cached repetitions. Mean +/- sample standard deviation: `val_bpb`
1.2118846990043566 +/- 0.008823959785837629; cached decode 158.54099134624397 +/-
5.3400259205954885 tok/s; prefill 6.513068079948425 +/- 0.291395972228771 ms;
training throughput 300270.7055742931 +/- 1019.3496156715676 tok/s; total tokens
90387251.2 +/- 287164.3642292685; pipeline runtime 492.4855269908905 +/-
17.485026046899637 seconds. Peak allocated inference memory is 249.88385009765625
MiB, reserved memory 352.0 MiB, KV growth 3.984375 MiB, and peak training memory
22804.445083618164 MiB for every seed. Total recorded fresh-suite pipeline time is
2462.4276349544525 seconds (0.6840076763762368 GPU-hours). The database, exports,
logs, samples, candidates, and event streams are synchronized locally. Checkpoints
remain remote; seed 0 was SHA-256 verified after creation.

Row 13 is the visible unchanged seed-42 discovery reference campaign
`discovery-reference-seed42-20260722`. It completed with `val_bpb`
1.1892983259238852, median cached decode 157.342638264525 tok/s (stdev
7.52920026686155), 6.60345703363419 ms prefill, 38,770.914643083 prefill tok/s,
249.883850097656 MiB peak allocated inference memory, 352 MiB reserved memory,
3.984375 MiB incremental KV-cache memory, 294,172.437099101 training tok/s,
94,371,840 total training tokens, 22,804.4450836182 MiB peak training memory,
50,332,176 parameters, 301.199775457382 training seconds, 1.62066723965108
median decode seconds, and 585.80659532547 seconds total pipeline runtime. Both
warm-ups and all ten measured repetitions completed; all measured cached/uncached
correctness checks passed. Its candidate commit remains the frozen shared baseline
`5ea452c9d86a43ad4bebd507c5872f18249627ce`; evaluator commit is
`e657a6739e3e9bbb268a84d8de47c0682250d380`.

Rows 14 through 63 are the 50 AutoResearch attempts for campaign
`autoresearch-seed42-20260723`. Parent values are experiment database IDs.

| Attempt | Parent | Candidate change | Outcome | Decision |
|---:|---:|---|---|---|
| 1 | 13 via visible action | `WARMDOWN_RATIO` 0.5 to 0.4 | interrupted before GPU | discarded |
| 2 | 13 | `WARMUP_RATIO` 0.0 to 0.05 | `val_bpb=1.2445483608068502` | discarded |
| 3 | 13 | Muon ramp denominator 300 to 150 | `val_bpb=1.1908194994994996` | discarded |
| 4 | 13 | `WINDOW_PATTERN` `SSSL` to `SSSS` | `val_bpb=1.1890557083878242` | retained |
| 5 | 17 | `TOTAL_BATCH_SIZE` `2**19` to `2**18` | `val_bpb=1.1716653252934937` | retained |
| 6 | 18 | `TOTAL_BATCH_SIZE` `2**18` to `2**17` | `val_bpb=1.099018855735464` | retained |
| 7 | 19 | `MATRIX_LR` 0.04 to 0.03 | `val_bpb=1.0998658906719363` | discarded |
| 8 | 19 | `MATRIX_LR` 0.04 to 0.05 | correctness crash, max error 0.135768 | discarded |
| 9 | 19 | `WARMDOWN_RATIO` 0.5 to 0.6 | `val_bpb=1.0986702394247787` | retained |
| 10 | 22 | `WARMDOWN_RATIO` 0.6 to 0.7 | `val_bpb=1.1058997335920697` | discarded |
| 11 | 22 | `WARMDOWN_RATIO` 0.6 to 0.55 | correctness crash, max error 0.122047 | discarded |
| 12 | 22 | `EMBEDDING_LR` 0.6 to 0.5 | `val_bpb=1.0988647520856358` | discarded |
| 13 | 22 | `EMBEDDING_LR` 0.6 to 0.7 | `val_bpb=1.0985479186876161` | retained |
| 14 | 26 | `EMBEDDING_LR` 0.7 to 0.8 | correctness crash, max error 0.143929 | discarded |
| 15 | 26 | `UNEMBEDDING_LR` 0.004 to 0.005 | `val_bpb=1.098233192293993` | retained |
| 16 | 28 | `UNEMBEDDING_LR` 0.005 to 0.006 | `val_bpb=1.0976744953448063` | retained |
| 17 | 29 | `UNEMBEDDING_LR` 0.006 to 0.007 | correctness crash, max error 0.153418 | discarded |
| 18 | 29 | `UNEMBEDDING_LR` 0.006 to 0.0065 | correctness crash, max error 0.131538 | discarded |
| 19 | 29 | `SCALAR_LR` 0.5 to 0.4 | `val_bpb=1.0968251425437556` | retained |
| 20 | 32 | `SCALAR_LR` 0.4 to 0.3 | correctness crash, max error 0.133013 | discarded |
| 21 | 32 | `WEIGHT_DECAY` 0.2 to 0.1 | `val_bpb=1.0967681235011013` | retained |
| 22 | 34 | `WEIGHT_DECAY` 0.1 to 0.0 | correctness crash, max error 0.142902 | discarded |
| 23 | 34 | Adam beta1 0.8 to 0.85 | interrupted by project-volume quota after training | discarded |
| 24 | 34 | Adam beta1 0.8 to 0.75 | correctness crash, max error 0.154791 | discarded |
| 25 | 34 | Adam beta2 0.95 to 0.90 | correctness crash, max error 0.151883 | discarded |
| 26 | 34 | Adam beta2 0.95 to 0.99 | correctness crash, max error 0.142021 | discarded |
| 27 | 34 | `WARMUP_RATIO` 0.0 to 0.01 | `val_bpb=1.1037094687098055` | discarded |
| 28 | 34 | `WARMUP_RATIO` 0.0 to 0.02 | correctness crash, max error 0.1422 | discarded |
| 29 | 34 | `FINAL_LR_FRAC` 0.0 to 0.05 | correctness crash, max error 0.122488 | discarded |
| 30 | 34 | `FINAL_LR_FRAC` 0.0 to 0.10 | correctness crash, max error 0.156537 | discarded |
| 31 | 34 | `WARMDOWN_RATIO` 0.6 to 0.65 | correctness crash, max error 0.135284 | discarded |
| 32 | 34 | `WARMDOWN_RATIO` 0.6 to 0.625 | correctness crash, max error 0.161916 | discarded |
| 33 | 34 | `MATRIX_LR` 0.04 to 0.035 | correctness crash, max error 0.136102 | discarded |
| 34 | 34 | `MATRIX_LR` 0.04 to 0.045 | correctness crash, max error 0.152396 | discarded |
| 35 | 34 | `EMBEDDING_LR` 0.7 to 0.65 | `val_bpb=1.0981045739725503` | discarded |
| 36 | 34 | `EMBEDDING_LR` 0.7 to 0.75 | correctness crash, max error 0.181266 | discarded |
| 37 | 34 | `SCALAR_LR` 0.4 to 0.35 | correctness crash, max error 0.148182 | discarded |
| 38 | 34 | `SCALAR_LR` 0.4 to 0.45 | `val_bpb=1.0980312867473538` | discarded |
| 39 | 34 | `UNEMBEDDING_LR` 0.006 to 0.0055 | `val_bpb=1.0967847138311442` | discarded |
| 40 | 34 | `UNEMBEDDING_LR` 0.006 to 0.00575 | correctness crash, max error 0.154868 | discarded |
| 41 | 34 | `TOTAL_BATCH_SIZE` to `2**16` | assertion crash | discarded |
| 42 | 34 | `TOTAL_BATCH_SIZE` to `3*2**16` | assertion crash | discarded |
| 43 | 34 | `WINDOW_PATTERN` to `SSLS` | correctness crash, token mismatch/error 23.2211 | discarded |
| 44 | 34 | `WINDOW_PATTERN` to `SLSS` | correctness crash, max error 0.1494 | discarded |
| 45 | 34 | logit softcap 15 to 20 | correctness crash, max error 3.94101 | discarded |
| 46 | 34 | logit softcap 15 to 10 | correctness crash, max error 4.99039 | discarded |
| 47 | 34 | Muon `ns_steps` 5 to 4 | correctness crash, max error 0.165815 | discarded |
| 48 | 34 | Muon `ns_steps` 5 to 6 | `val_bpb=1.096754145494427` | retained |
| 49 | 61 | Muon momentum start 0.85 to 0.80 | correctness crash, max error 0.179209 | discarded |
| 50 | 61 | Muon momentum endpoint 0.95 to 0.97 | correctness crash, max error 0.151338 | discarded |

Attempt 48 (database id 61) is the retained quality winner. Relative to the unchanged
seed-42 reference, its `val_bpb` improves by 0.09254418042945822 (7.781410131689786%).
It records median cached decode 161.6290768607186 tok/s, prefill
6.275827996432781 ms, 249.88385009765625 MiB peak allocated inference memory,
352 MiB reserved memory, 3.984375 MiB KV growth, 290,746.5578689759 training tok/s,
88,735,744 training tokens, 22,652.443069458008 MiB peak training memory, and
300.2407066822052 measured training seconds. These secondary measurements did not
affect retention.

Rows with stored total runtime account for 25,601.386746168137 seconds. Attempt 23
adds an event-derived 414.110-second active window from proposal through its final
100% training-progress event; attempt 1 used no GPU. The resulting accounted campaign
time is 26,015.496746168137 seconds (7.226526873935594 GPU-hours), estimated at
$3.1796718245316615 using the frozen $0.44/hour A40 rate. This excludes Pod idle and
storage time, so it is not an invoice estimate.

The campaign state is `completed`, `next_experiment_number=51`, and
`attempt_limit=50`. The canonical local database, exports, logs, samples, completed
candidates, and event streams are synchronized. Checkpoints remain remote.

Rows 64 through 68 are the five completed diagnostic attempts for campaign
`autoresearch-diagnostic5-seed42-20260724`. Parent values are experiment database IDs.

| Attempt | ID | Parent | `FINAL_LR_FRAC` | Training | Inference | Median cached tok/s | Peak allocated MiB | Decision |
|---:|---:|---:|---:|---|---|---:|---:|---|
| 1 | 64 | 61 | 0.05 | `val_bpb=1.095215` (log-rounded) | not started after quota/SQLite failure | — | — | retained |
| 2 | 65 | 64 | 0.075 | `val_bpb=1.0948561434742032` | incompatible, max mean error `0.0341237` | — | — | retained |
| 3 | 66 | 65 | 0.0875 | `val_bpb=1.0952247206829955` | compatible | 163.95254015579656 | 249.88385009765625 | discarded |
| 4 | 67 | 65 | 0.06875 | `val_bpb=1.0953718755313648` | compatible | 163.11501283527514 | 249.88385009765625 | discarded |
| 5 | 68 | 65 | 0.078125 | `val_bpb=1.0949760675404565` | compatible | 161.4235396941305 | 249.88385009765625 | discarded |

Diagnostic attempt 2 (database id 65) is the final retained quality winner. Relative
to starting parent 61, it improves `val_bpb` by `0.0018980020202237213`
(`0.1730562886879342%`). Its stored candidate combines the completed AutoResearch
retained chain, Muon `ns_steps=6`, and `FINAL_LR_FRAC=0.075`. Because its inference
check was incompatible, it has no official cached speed or memory measurement and is
excluded from three-objective analysis.

All five diagnostic attempts completed the measured training phase, totaling
1,501.377537536621 training seconds and 443,809,792 processed tokens. Stored or
event-derived pipeline runtime totals 2,494.9574689865112 seconds
(`0.693043741385142` GPU-hours), estimated at `$0.3049392462094625` using the frozen
`$0.44/hour` A40 rate. Attempt 1 contributes a 477-second artifact-derived active
window because its terminal SQLite write failed. This excludes Pod idle and storage
time and is not an invoice estimate.

The diagnostic campaign state is `completed`, `next_experiment_number=6`, and
`attempt_limit=5`. The canonical local database, exports, logs, samples, candidates,
and event streams are synchronized. Diagnostic checkpoints remain remote under the
off-volume symlink. SQLite integrity and foreign-key checks pass.

## Exact commands

From `/Users/Rami/Documents/dev/autopareto`:

```bash
scripts/run_runner.sh --metadata-preflight
python3 scripts/run_confirmation_ablation.py --validate-only
python3 dashboard/server.py --host 127.0.0.1 --port 8765
```

After the user starts the A40 Pod:

```bash
scripts/push_to_runpod.sh --include-results
scripts/run_remote.sh uv sync --frozen
scripts/run_remote.sh uv run python -m unittest discover -s tests -v
scripts/run_remote.sh uv run python runner.py protection-check
scripts/run_remote.sh uv run python protected/smoke_gpu.py --train train.py
scripts/run_remote.sh uv run python scripts/run_confirmation_ablation.py --validate-only
scripts/run_remote.sh --tmux confirmation-ablation-20260726 \
  uv run python scripts/run_confirmation_ablation.py
scripts/live_from_runpod.sh
```

After all twelve rows are terminal:

```bash
scripts/pull_from_runpod.sh --include-checkpoints
```

Classical sampler validation performs no GPU work and stores no future proposal plan:

```bash
for method in random_search tpe cmaes motpe nsgaii; do
  uv run --no-project --with optuna==4.5.0 --with cmaes==0.12.0 \
    python search/run_search.py \
    --method "$method" --campaign-id "validation-$method" \
    --experiments 25 --training-seed 42 --algorithm-seed 20260725 \
    --validate-only
done
```

## Unresolved problems

- Cached inference CUDA validation on the July 2026 A40 found deterministic
  bfloat16 kernel drift between the canonical FlashAttention full-prefix path and
  the protected SDPA cache path. All 256 generated tokens matched for each of the
  four intact baseline checkpoints; maximum absolute logit error was 0.08686 and
  maximum per-step mean absolute error was 0.01393. The protected absolute
  correctness tolerance is therefore 0.10 (relative tolerance remains 0.03). The
  post-change protected run passed exact token and logit-tolerance checks.
- The diagnostic amendment now uses 16 teacher-forced next-token comparisons, exact
  greedy-token agreement, and a maximum per-step mean absolute logit error of `0.03`.
  Diagnostic attempt 2 matched all greedy tokens but reached `0.0341237` and is
  therefore correctly marked inference-incompatible. Attempts 3 through 5 passed.
- Diagnostic attempt 1's volume-resident checkpoint stopped at exactly 134,217,728
  bytes during quota exhaustion and is not a usable confirmation artifact. Its
  completed training-quality evidence is preserved at the six-decimal precision
  printed by the terminal training log. It must not be rebenchmarked or silently
  treated as an exact full-precision metric.
- The canonical local seed-0 checkpoint `0001.pt` is a truncated PyTorch archive.
  Its transferred copy matched the local SHA-256 and has been quarantined on the
  current Pod. Seeds 1--4 are intact and hash-verified. All five cached baseline
  measurements are already complete and valid; seed 0 needs recovery or retraining
  only if that checkpoint must be reloaded for a future checkpoint-only measurement.

- The RunPod SSH gateway requires a PTY and rejects SCP/non-PTY rsync. Synchronization
  therefore uses a PTY-safe compressed base64 archive; it is slower than native rsync,
  especially for checkpoints. Four concurrent PTY/base64 transfers succeeded and were
  hash-verified, but the gateway limited their aggregate throughput.
- The five existing baseline training-throughput rows divided all processed tokens by a
  timer that excluded compilation/warm-up. They are preserved for auditability. Future
  runs divide post-warm-up tokens by that timer; quality and inference are unaffected.
- The protected inference loader requires candidates to retain the upstream setup marker
  and compatible `GPT`/`GPTConfig` reconstruction contract. Arbitrary edits that remove
  that contract will count as failed experiments.
- The July A40 validation covered the current 256--512 token benchmark range, which does
  not cross the model's 1,024-token short-attention boundary. A separate boundary test
  is still desirable before publishing general sliding-window cache claims, but it is
  outside the fixed headline benchmark.
- Remote locked dependencies, 10 training shards plus the pinned validation shard, and
  the protected 8,192-token tokenizer are prepared on the active Pod.
- Upstream `DEVICE_BATCH_SIZE=128` cannot fit the A40; the two failed feasibility rows
  are deliberately preserved and excluded.
- RunPod's public Pod rate for an A40 was $0.44/hour when checked on 2026-07-20. The
  recorded five-baseline pipeline time therefore estimates to $0.2917 of compute; the
  four new runs alone estimate to $0.2187. Actual billing can be higher because Pod idle,
  dependency/data preparation, and storage time are not represented by experiment rows.

## Dashboard repair completed locally

- Explore now defaults to all completed compatible measurements. Discovery and confirmed
  frontiers are explicit optional filters rather than unexplained hidden defaults.
- Quality is presented lower-is-better over 1.30 to 1.00; cached decode uses 0--200
  tokens/second and inference memory uses 0--300 MiB, with automatic expansion when a
  future result exceeds a bound.
- Training-complete experiments without protected cached inference are preserved as
  quality-only evidence. Experiment 65 remains visible at `1.094856` BPB with unknown
  speed and memory; no deployment metric is invented for it.
- Campaigns use reader-facing names. Playback is isolated by campaign and numbered by
  experiment, while raw telemetry events remain an internal audit detail.
- Lineage defaults to the selected experiment's ancestry and offers a scrollable full
  campaign tree. Live, Evidence, Presentation, desktop, tablet, and phone layouts were
  checked without console errors.
- The local regression suite passes: 28 tests run, with four isolated optimizer tests
  intentionally deferred to the pinned remote environment.

## Pre-AutoPareto study completed — 2026-07-27

The preregistered confirmation and ablation study completed all 12 training runs:

| Recipe | Seeds | Mean validation BPB | Mean paired baseline improvement | Headline cached compatibility |
|---|---|---:|---:|---:|
| Experiment 65 exact | 0, 1, 2 | 1.101704 | 0.114840 | 0/3 |
| Experiment 68 exact | 0, 1, 2 | 1.101425 | 0.115118 | 2/3 |
| Batch-only (`TOTAL_BATCH_SIZE=2**17`) | 0, 1, 2 | 1.103968 | 0.112576 | 3/3 |
| Non-batch-only (Experiment 68 recipe at `2**19`) | 0, 1, 2 | 1.218802 | -0.002258 | 3/3 |

The main causal result is that the smaller total training batch explains nearly all
of the quality gain. The remaining Experiment 68 changes do not improve mean quality
without that batch change. Experiment 65 reproduces as strong quality-only evidence,
but all three cached-inference checks exceed the protected mean-logit-error tolerance.
Experiment 68 reproduces in quality, but one of three seeds is inference-incompatible.
Therefore neither recipe qualifies as a fully confirmed three-objective model.

The expanded deployment matrix measured context lengths 256, 896, 1024, and 1536,
inference batches 1, 2, 4, and 8, and 128 generated tokens:

- intact baseline experiment 4: 16/16 workloads compatible;
- Experiment 68 representative experiment 77: 10/16 compatible;
- batch-only representative experiment 80: 11/16 compatible;
- no workload failed from out-of-memory;
- failures were protected cache-correctness failures, proving that compatibility is
  workload-dependent and cannot be inferred from the headline workload alone.

Historical experiments 61 and 19 were not remeasured because their checkpoints were
unavailable. Their stored historical measurements remain intact; no value was
estimated or reconstructed. The full selection audit is in
`results/deployment_characterization_selection.json`.

Only four important final checkpoints were retained on persistent RunPod storage:
the intact baseline (experiment 4), the median Experiment 65 quality-only run
(experiment 74), the median compatible Experiment 68 run (experiment 77), and the
median compatible batch-only run (experiment 80). Exact paths, byte counts, and
SHA-256 hashes are recorded in `results/preserved_checkpoints/manifest.json`.

The initial checkpoint-mount preflight exposed a dangling temporary checkpoint
symlink before any training began. Three setup-only reservations were archived under
the hidden `setup-checkpoint-mount-20260727` campaign, with the exact audit at
`results/archive/2026-07-27-checkpoint-mount-setup.json`. They are not scientific
attempts and no evidence was deleted.

The dashboard now:

- gives confirmation and ablation campaigns unique reader-facing names;
- shows all six confirmation quality repeats and their compatibility status;
- opens Explore on the representative batch-only seed-2 result instead of the latest
  rejected ablation, and labels each selected result with a plain-language verdict;
- summarizes the ablation as four three-seed means and keeps the six individual seed
  runs behind an explicit disclosure;
- explains why the confirmed frontier is zero instead of calling it pending;
- replaces empty quality-budget cards with one explanation of what evidence is still
  required;
- includes the characterized historical baseline without exposing its archived
  campaign as a duplicate research campaign;
- summarizes deployment compatibility per representative model and keeps the full
  48-workload table available behind a disclosure;
- uses Presentation as a project-level result slide rather than silently replaying the
  latest experiment, with the causal finding, evidence, limitation, and readiness
  state visible together;
- shows failed deployment workloads and distinguishes token mismatches from logit
  tolerance failures;
- preserves Experiment 65 and every other training-complete incompatible result in
  the quality-only section;
- keeps fixed readable axes and permanent experiment selection when switching seeds.

All four dashboard modes were visually rechecked after this final evidence-polish
pass; no remaining overlap or selection-reset defect was observed.

## AutoPareto preflight hardening — 2026-07-27

Before attempt 1, the AutoPareto contract and runner were hardened:

- each campaign immutably registers one exact valid compatible baseline with the same
  training seed;
- campaign frontier and decision calculations contain only that baseline and the
  campaign's own same-seed results;
- proposals must use a current frontier parent, and attempt `N+1` is blocked until
  attempt `N` is terminal with a recorded Pareto decision;
- `campaign-view` exposes only the registered baseline, own history, own actions, and
  own frontier;
- the agent program explicitly permits architecture, optimizer, schedule, batching,
  training-math, and inference-cache research while preserving the protected candidate
  interface;
- the protected launcher captures the real protected `evaluate_bpb` result, requires
  exactly one validation call, and enforces a 300–315 second measured-training window;
- cache compatibility is checked over 32 teacher-forced decode steps at protected
  prefix lengths 64 and 256 before timing, and cache byte reports must equal tensor
  storage reachable from the decode state;
- final AutoPareto representatives are mechanically preregistered as best quality,
  fastest and lowest-memory within a 1% quality budget, and largest remaining
  exclusive hypervolume contribution, with source deduplication before three-seed
  confirmation.

The pre-hardening unused registration is recoverable from
`results/archive/2026-07-27-before-autopareto-hardening-registration.db`. The current
seed-42 campaign registers baseline experiment 13 and program SHA-256
`af42d0417ca6b98360ea6d84399290849d49f9680d38f9581de971ea586b3021`.
It remains ready at attempt 1 with zero experiment rows.

Local SQLite integrity and foreign-key checks pass. The expanded local test suite
passes 36 tests with four pinned-optimizer tests intentionally skipped in the Mac
environment. The previous remote suite passed before this hardening; all 36 current
tests, the strengthened CUDA cache checks, and the protected training contract must be
rerun on the next A40 before attempt 1.

## Exact next action

AutoPareto seed 42 is registered as `autopareto-seed42-20260727` for exactly 25
attempts, training seed 42, and baseline experiment 13. Its state is `ready`, its next
experiment number is 1, and it has zero experiment rows. It must be handed to a fresh
task that uses only `campaign-view`, the current protocol, baseline, own history, and
own Pareto frontier.
Do not expose AutoResearch, confirmation, ablation, or classical-search histories to
that task. Do not start attempt 1 until the user explicitly requests the AutoPareto
campaign.

## Seed-42 A40 preflight infrastructure blocker — 2026-07-28

The authorized 25-attempt seed-42 task stopped during preflight without reserving or
running attempt 1:

- local source was clean at
  `c200ccbe08b88c199e73715849187affd967eabe`;
- the exact source and canonical non-checkpoint results were restored to RunPod Pod
  `befortg5ids4dw`, with matching `runner.py`, program, and SQLite file hashes;
- the remote locked environment audit and all 36 tests passed, including immutable
  campaign registration, campaign isolation, sequential proposal enforcement,
  mechanical Pareto decisions, protected training-contract checks, and CPU cache
  adapter correctness;
- remote metadata, protected-file mutation detection, representative-selector
  validation, and exact campaign registration passed;
- `campaign-view` showed campaign `autopareto-seed42-20260727` ready at experiment 1
  with zero campaign attempts and frontier `[13]`;
- the fresh Pod had no dataset/tokenizer cache, so protected `prepare.py` downloaded
  all 11 pinned shards, trained the 8,192-token tokenizer, built the token-byte lookup,
  and passed its sanity check;
- the fresh Pod also had no checkpoint. The intact local seed-1 baseline checkpoint
  was being transferred solely for strict checkpoint-load and protected multi-prefix
  cached-inference preflight when the RunPod container disappeared. The gateway then
  repeatedly returned `container not found`.

Because the container vanished, strict checkpoint load, the CUDA smoke test, and the
full protected A40 cache-correctness benchmark could not be completed. The transferred
checkpoint must be treated as unverified because its final remote byte count and
SHA-256 could not be read. No `run` command was issued, no scientific attempt was
consumed, and no candidate was created.

The container failure also prevented a final pull. Before it disappeared, the only
authoritative-database operation after byte-identical synchronization was the
idempotent exact registration check; it kept the campaign ready at attempt 1 and zero
rows. The canonical local database remains at that same scientific state. The RunPod
stop API subsequently confirmed Pod `befortg5ids4dw` at desired status `EXITED`;
stopping preserves its `/workspace` volume.

## Maintainer and Cursor continuation contract

This section is for a maintenance/orchestration agent, not for the isolated AutoPareto
research agent.

The durable human objective and publication/open-source standard are recorded in
`README.md`. The contamination-safe research-agent entry point is
`handoffs/autopareto-seed42.md`. Do not create a root `AGENTS.md`; an automatically
loaded maintainer summary would expose historical conclusions to independent research
campaigns.

Before starting a replacement Pod or research task:

1. preserve a timestamped local SQLite backup and run SQLite integrity and foreign-key
   checks;
2. verify the campaign with
   `scripts/run_runner.sh campaign-view --campaign-id
   autopareto-seed42-20260727`;
3. verify the new endpoint and A40, then inspect persistent caches and checkpoint
   hashes before downloading, preparing, transferring, or training anything;
4. treat the interrupted checkpoint upload from Pod `befortg5ids4dw` as invalid unless
   its complete byte count and SHA-256 match the local manifest;
5. finish only the missing protected CUDA/cache preflight before the next scientific
   attempt.

A consolidated pre-restart safety snapshot was created at
`results/archive/2026-07-28-before-seed42-restart.db`; its integrity check passes and
it contains zero seed-42 AutoPareto experiment rows.

To launch either Codex or Cursor as the isolated research agent, provide only:

> Read `handoffs/autopareto-seed42.md` completely and follow it exactly. Continue the
> registered seed-42 campaign from the authoritative `campaign-view`; do not inspect
> any other project history or results.

At a tool-credit or agent transition, never hand off during a running attempt. Finish
the attempt, store its mechanical Pareto decision, pull the authoritative database and
artifacts, verify integrity and the exact next experiment number, and stop before
reserving the next attempt. The replacement agent resumes from SQLite through the same
handoff. This remains valid whether the prior agent completed zero, one, fifteen,
twenty, or all twenty-five attempts.

After the seed-42 campaign is complete, review its evidence before authorizing other
seeds. Documentation consolidation, the larger-model transfer protocol, a formal
novelty review, remaining method replicates, and any AutoPareto--MOTPE hybrid must be
specified outside the frozen seed-42 campaign and committed before their results are
observed.

## Seed-42 checkpoint-preflight clarification — 2026-07-28

A second fresh task completed the database synchronization, locked-environment checks,
A40 detection, persistent dataset/tokenizer preparation, and CUDA smoke, but then
stopped with zero attempts because it interpreted the missing experiment-13 checkpoint
as a hard blocker. That interpretation was too strict.

Experiment 13 remains the immutable scientific baseline by its stored protected
metrics; its checkpoint is not needed to calculate the campaign frontier or train a
new candidate from the frozen baseline source. The local repository contains the
intact same-architecture seed-1 baseline checkpoint at
`results/checkpoints/baseline-a40-seed0/0002.pt`, exactly 159,401,129 bytes with
SHA-256
`d8d03b3ef3df27ea1fe25a5a199b62a9e70ea246f2b443420434f2fb7d1302e9`.
This matches `results/preserved_checkpoints/manifest.json`.

`handoffs/autopareto-seed42.md` now requires that checkpoint solely for strict-load,
CUDA, and protected cached/full inference preflight. It explicitly forbids substituting
it for experiment 13 or using its seed or metrics as campaign evidence. Missing
experiment-13 checkpoint bytes are no longer a blocker. The historical preflight
state above is superseded by the current continuation state below.

## Current Seed-42 AutoPareto continuation — 2026-07-30

The authoritative Seed-42 AutoPareto campaign is ready with the validated prefix
through attempt 4. Attempt 4 is a compatible retained result with `val_bpb`
`1.0992980186424477`, median cached decode `159.0068612604278` tok/s, and peak
inference memory `249.88385009765625` MiB. Its candidate source is preserved and the
checkpoint used for its corrected cached-inference measurement is persistent at
`/workspace/autopareto-diagnostics/attempt4-recreation/0004-recreated.pt` with
SHA-256 `102fee9656cef56be6824c6a7e892702a7ac7e25fc415e3304d2949cdf50fc37`.

The campaign view's next experiment number is 5. Future attempts continue sequentially
from 5 through 25 and may choose attempt 4 as their current quality frontier parent.
For AutoPareto, a saved candidate checkpoint followed by an inference, cache-correctness,
or inference-memory failure now pauses the campaign and blocks the next reservation
until maintainer review; the checkpoint and exact failure remain available for debugging.

## Attempt-5 recovery completed — 2026-07-30

The lost attempt 5 was rerun exactly from attempt 4 on a replacement A40 Pod. The
candidate retained `TOTAL_BATCH_SIZE=2**17`, changed only `KV_HEAD_RATIO` from 1 to
2, and used training seed 42. Its archived candidate SHA-256 is
`20ec34df24572b0319053734cea9653947e1089c3a3f65da3aca2ff23e893507`.

The authoritative attempt 5 is experiment ID 92. It completed with:

- validation BPB `1.1040101100692374`;
- median cached decode `151.36198232135402` tok/s;
- peak allocated inference memory `219.88482666015625` MiB;
- peak reserved inference memory `256.0` MiB;
- incremental KV-cache growth `1.9921875` MiB;
- `39,846,160` parameters.

Every bfloat16 greedy token matched. The bfloat16 mean-logit difference reached
`0.03569479286670685`, so the corrected evaluator invoked strict float32 math-SDPA
adjudication. Float32 passed with maximum mean error
`0.000006122758804849582` and maximum absolute error
`0.000017344951629638672`. Official speed and memory remained measured in bfloat16.
Attempt 5 is therefore inference-compatible and mechanically `retained` as a
quality–memory Pareto trade-off.

The checkpoint is synchronized locally at
`results/checkpoints/autopareto-seed42-20260727/0005.pt`, exactly 134,234,281 bytes
with SHA-256
`99b7160712e5a349fb1adfdf97ba63a11b0e4d378d9766796b07762fb16decce`.

One replacement-Pod tokenizer-link failure occurred before any training. That setup
row was reclassified into the hidden archived campaign
`setup-autopareto-seed42-20260730-tokenizer-cache`; it did not consume scientific
attempt 5. Stale live-event rows belonging to the explicitly discarded earlier
attempt-5 record were removed after exact ID verification. SQLite integrity and
foreign-key checks now pass.

The Seed-42 campaign is `ready`, its next experiment number is 6, and its current
frontier IDs are 84, 85, 86, 87, and 92. No attempt 6 has been proposed or reserved.

## Attempts 6–10 and automatic FP32 correction — 2026-07-31

Seed-42 attempts 6 through 9 completed compatibly and were mechanically retained.
Attempt 10 completed training with `val_bpb` `1.0941956306093856`, but the protected
bfloat16 check produced a greedy-token mismatch at prefix 64 and maximum mean-logit
drift `0.03363482281565666` at prefix 256. The evaluator paused the campaign before
float32 because its earlier fallback covered matching-token numerical drift only.

The protected correctness policy now automatically runs strict float32 math-SDPA
adjudication after every bfloat16 correctness failure, including greedy-token
mismatches. Bfloat16 remains the only precision used for official speed and memory.
The complete A40 test suite passes all 40 tests, including a regression test for this
branch, and the protected-integrity check passes.

Attempt 10 was remeasured once from its saved checkpoint without retraining. Float32
greedy tokens matched at both protected prefixes, with maximum mean error
`0.0000022497160898637958` and maximum absolute error
`0.00001239776611328125`. Its official metrics are:

- validation BPB `1.0941956306093856`;
- median cached decode `152.85274747030297` tok/s;
- peak allocated inference memory `249.88629150390625` MiB;
- peak reserved inference memory `288.0` MiB;
- incremental KV-cache growth `3.984375` MiB.

Attempt 10 is therefore compatible, valid, included, and mechanically `retained`.
The authoritative campaign is `ready` at attempt 11; no attempt 11 has been reserved.
SQLite integrity and foreign-key checks pass. A safety snapshot made before removing
242 orphaned database event duplicates is preserved in `results/archive`; their raw
JSON event files were not deleted.

## Seed-42 AutoPareto completed and preserved — 2026-07-31

The authoritative campaign `autopareto-seed42-20260727` completed exactly 25 of 25
attempts and is terminal at next experiment number 26. All 25 attempts completed
training and are inference-compatible. SQLite integrity and foreign-key checks pass.
A pre-maintenance safety copy is stored at
`results/archive/safety-20260731-before-post-seed42-preparation.db`.

The most important seed-42 trade-offs are:

- attempt 16: `1.1228413979653016` BPB, `242.20686754733217` cached tok/s,
  `113.3213882446289` MiB, `17,301,610` parameters;
- attempt 17: `1.1810906497739186` BPB, `302.44428192347607` cached tok/s,
  `74.50869750976562` MiB, `9,175,806` parameters;
- attempt 21: `1.1003845377646218` BPB, `178.2447095298847` cached tok/s,
  `189.00908660888672` MiB, `31,850,638` parameters;
- attempt 22: the seed-42 quality winner at `1.0939814996982178` BPB,
  `151.31841651770327` cached tok/s, and `249.88629150390625` MiB.

All locally available checkpoints from attempts 5--25 and all candidate sources from
attempts 1--25 are recorded with SHA-256 hashes in
`results/preserved_checkpoints/autopareto-seed42-20260727-manifest.json`. Attempts 16,
17, 21, 22, 24, and 25 are explicitly marked important. The exact AutoResearch
comparison checkpoint is not present locally; its database path and preserved-volume
manifest remain recorded, so the next remote preflight must retrieve and hash it if
the volume still contains it. Otherwise its exact recipe is reproduced during
confirmation rather than estimated.

Seed 42 has a disclosed evaluator amendment:

- attempts 1--4 used evaluator SHA-256
  `62b3730c8059a34aaef8d8b3e3b95c7d488a688add89283890cc99b41012ed6b`;
- attempts 5--9 used
  `5d6c20928c61ec164bdebb975011fb757edc99c0e5fc01316c0948de1e3ae58c`;
- attempts 10--25 used the final automatic float32-adjudication evaluator
  `0b252757d18c79a6a6ca319df6f813e337d41c571b34f1cd538cb138a0153cc0`.

The official bfloat16 speed and memory benchmark did not change. Float32 is only a
correctness fallback and never supplies headline performance values. Selected models
will be remeasured under one final evaluator during confirmation.

The separately preregistered post-search language-quality suite is now
`evaluation/downstream_suite.json`. `downstream_evaluator.py` stores held-out BPB,
LAMBADA, BLiMP, PIQA, ARC-Easy, and fixed illustrative completions in additive SQLite
tables and raw files under `results/downstream_evaluations/`. It is not a search
objective, is hidden from isolated research agents, and must not score research
candidates until all authorized AutoPareto and classical searches finish.

The downstream execution layer is prepared. `scripts/prepare_downstream_selection.py`
mechanically freezes 11 model representatives in `results/downstream_selection.json`;
the current audit finds nine complete local artifact pairs and two missing checkpoints:
the AutoResearch compatible reference and MOTPE quality representative.
`scripts/recover_downstream_representatives.py` first attempts hash-verified recovery
and otherwise performs one explicitly labelled exact-recipe retrain. The official
serial runner, `scripts/run_downstream_suite.py`, verifies every candidate/checkpoint
hash and SQLite integrity, supports a non-recorded smoke test, skips already completed
model evaluations, and refuses to overwrite failures. Each model saves aggregate task
scores plus 18 verbatim fixed-prompt completions, including token IDs, token count,
elapsed generation time, prompt, seed, and decoding settings. No official downstream
evaluation has run yet.

The seed-42 result interpretation is now frozen before seed 0 in
`evaluation/seed42_case_study.json`. It registers the baseline and AutoResearch
references plus AutoPareto quality, balanced, intermediate-efficiency, primary-
efficiency, and extreme-efficiency roles. `change_effect_analysis.py` derives the
parent-to-child quality, speed, memory, and parameter deltas from SQLite. The permanent
dashboard defaults to this curated comparison, retains the complete experiment audit
behind filters, exposes architecture details and the change-effect matrix, and labels
seed-42 findings as promising but unconfirmed. `docs/RESEARCH_LEDGER.md` is the
paper-source research narrative, not an instruction file for isolated search agents.

Post-confirmation quantization is separately frozen in
`evaluation/quantization_suite.json`: BF16 remains the reference, INT8 weight-only is
required, and INT4 is conditional on a verified A40 kernel. These deployment results
remain separate from the fixed full-precision search objectives.

The dashboard uses reader-facing scientific identities rather than raw database
numbers in its main story. Featured frontier points are labelled by role. Historical
quality-only AutoResearch and confirmation records are removed from Explore and kept
in a collapsed Evidence audit disclosure; missing speed or memory is never inferred.
The compatible AutoResearch comparison is explicitly named the compatible
representative, while internal database IDs remain available only as audit metadata.

At this 2026-07-31 checkpoint, local dashboard verification and the full local test
matrix passed. The remaining AutoPareto campaigns were registered as
`autopareto-seed0-20260731` and `autopareto-seed1-20260731`, each ready at attempt 1
of 25 against its same-seed cached baseline. Neither had then reserved an attempt.
This historical state is superseded by the seed-0 completion record below.

## AutoPareto seed 0 completed — 2026-08-04

The authoritative campaign `autopareto-seed0-20260731` is terminal at 25/25 attempts
and next experiment number 26. Attempts 1--16 and 18--25 completed protected training
and cached inference. Attempt 17 was correctly recorded as a consumed infrastructure
failure before training because
`/root/.cache/autoresearch/tokenizer/tokenizer.pkl` disappeared. SQLite
`integrity_check` returns `ok`, foreign-key checks are clean, and every attempt has a
hypothesis, intended change, exact candidate diff, parent, and decision.

The mechanically selected seed-0 discovery representatives are:

- attempt 22 / database ID 134: best quality, `1.093294` BPB, `312.84` cached tok/s,
  `159.88` MiB;
- attempt 12 / database ID 124: lowest memory within one percent of best quality,
  `1.096304` BPB, `258.26` cached tok/s, `132.20` MiB;
- attempt 25 / database ID 137: high-throughput quality model, `1.097434` BPB,
  `408.40` cached tok/s, `141.38` MiB;
- attempt 24 / database ID 136: aggressive efficiency / largest exclusive
  hypervolume contribution, `1.116324` BPB, `351.09` cached tok/s, `105.20` MiB.

Attempt 10 is not an architectural width experiment: attempts 9 and 10 both actually
built depth 4, width 384, three query heads, and three KV heads because the sizing rule
rounds to the 128-dimensional head boundary. Its speed difference is repeated-runtime
evidence, not a causal width effect. Attempt 7 similarly remained width 384 rather than
the intended width 320, making its actual parent-to-child change a clean depth removal.
The frozen, corrected interpretation is in `evaluation/seed0_case_study.json` and the
paper-source narrative is in `docs/RESEARCH_LEDGER.md`.

All 25 candidate sources, logs, database rows, and 24 successful event streams are
local. All 24 successful checkpoints were recovered into
`results/checkpoints/autopareto-seed0-20260731` and their byte sizes and SHA-256 hashes
were recorded in
`results/preserved_checkpoints/autopareto-seed0-20260731-manifest.json`. Attempt 17 has
no checkpoint because its infrastructure failure occurred before training.

The dashboard now combines the frozen seed-42 and seed-0 case studies. Explore
highlights both discovery campaigns; Evidence separates single-variable, compound,
training-only, and duplicate-architecture effects; Presentation reports 50 registered
attempts, 49 compatible results, and the one infrastructure failure. These remain
discovery results, not cross-seed confirmations.

Every future `scripts/run_runner.sh run` executes
`scripts/ensure_persistent_cache.py` before `runner.py` can reserve an attempt. The
preflight requires tokenizer files, the pinned validation shard, at least one training
shard, and a runtime cache resolving to persistent `/workspace` storage. This prevents
a cleared ephemeral `/root` cache from consuming another attempt.

The registered `autopareto-seed1-20260731` campaign remains untouched and ready at
attempt 1/25. Seed-0 checkpoint recovery is complete; the persistent cache resolves to
`/workspace/.cache/autoresearch` and contains the required tokenizer, validation shard,
and ten training shards. Seed 1 may begin only after the full local test matrix passes
and the user explicitly authorizes the campaign.

## Seed-1 readiness verification — 2026-08-04

- All 24 successful seed-0 checkpoints independently match the byte sizes and SHA-256
  hashes recorded in the preserved manifest; attempt 17 correctly has no checkpoint.
- Local SQLite integrity and foreign-key checks pass. Seed 1 remains `ready`, registered
  for training seed 1 and 25 attempts, with zero experiment rows.
- The local suite passes 52 tests, with four optional optimizer-environment tests
  skipped because Optuna is not installed in the Mac environment.
- The pinned RunPod environment passes all 52 tests, including the four optimizer
  tests. Runner metadata preflight, representative selection validation, and downstream
  evaluator validation pass both locally and remotely.
- Source-only synchronization preserved remote results. The protected runtime cache
  now resolves to persistent `/workspace/.cache/autoresearch` and passes its preflight.

No seed-1 attempt has been reserved or run. Starting that campaign remains a separate,
explicit user action.

## AutoPareto seed 1 completed and preserved — 2026-08-05

The historical readiness statement immediately above is superseded. Campaign
`autopareto-seed1-20260731` completed exactly 25/25 attempts on training seed 1. All
25 attempts completed training and protected cached inference, every attempt records
its hypothesis, intended change, exact diff, parent, and decision, and SQLite integrity
and foreign-key checks pass.

The mechanically selected seed-1 discovery representatives are:

- attempt 25 / database ID 162: best quality, `1.095755` BPB, `253.70` cached tok/s,
  `132.20` MiB;
- attempt 21 / database ID 158: fastest within one percent of best quality,
  `1.096740` BPB, `257.14` cached tok/s, `132.20` MiB;
- attempt 10 / database ID 147: lowest memory within one percent of best quality,
  `1.106084` BPB, `249.20` cached tok/s, `113.32` MiB;
- attempt 15 / database ID 152: largest exclusive hypervolume contribution,
  `1.342847` BPB, `534.79` cached tok/s, `49.82` MiB. This is an extreme boundary
  result, not a practical quality recommendation.

All 25 checkpoints are local under
`results/checkpoints/autopareto-seed1-20260731`. Their byte sizes and SHA-256 hashes,
and the hashes of the exact candidate sources, are recorded in
`results/preserved_checkpoints/autopareto-seed1-20260731-manifest.json`. The selector
now deduplicates exact candidate files rather than hashes of diff text. Its pooled,
reproducible output is `results/autopareto_confirmation_selection.json`.

The frozen seed-1 interpretation is `evaluation/seed1_case_study.json`. The dashboard
now presents four AutoPareto trade-offs per seed, uses campaign-specific colors, labels
only the selected point, and separates featured, mechanical, raw-frontier, extreme,
confirmed, and all-result views. All three campaigns remain discovery evidence until
exact recipes are retrained on confirmation seeds 0, 1, and 2.

The next GPU phase is the preregistered fixed-space classical comparison. No classical
attempt has been started by this preparation.

## Classical comparison registered — 2026-08-05

- Random Search, TPE, CMA-ES, MOTPE, and NSGA-II each have one immutable 25-attempt
  campaign for training seeds 42, 0, and 1: 15 campaigns and 375 maximum attempts.
- Every campaign is `ready` with zero attempts; no GPU experiment was started.
- Exact campaign IDs and matched algorithm seeds are frozen in
  `evaluation/classical_campaigns.json`. Every method uses the same algorithm seed for
  a given training seed and the same search-space SHA-256
  `bc20cc05ababd12616c0571b476bfb949d530096aac22d00ad97c2f58137ceee`.
- All five pinned optimizer implementations passed offline proposal validation.
- The quantization plan gate and additive SQLite audit tables validate locally. Actual
  BF16/INT8 work stays blocked until full-precision searches and three-seed confirmation
  finish; INT4 additionally requires a verified A40 kernel.
- The pre-registration database snapshot is
  `results/archive/safety-20260805-before-classical-registration.db` and passes SQLite
  integrity checking.

## TPE seed-42 infrastructure recovery — 2026-08-06

The first launch never reached training because the migrated Pod contained an incomplete
PyTorch installation. The authoritative local database remained clean at 0/25. The stale
optimizer database (three terminal infrastructure failures plus one unnumbered running
trial) is preserved at
`results/archive/2026-08-06-tpe-pretraining-infrastructure-failures-optimizer_state.db`
and is not active optimizer state.

The Pod environment was rebuilt exactly from `uv.lock`. PyTorch 2.9.1+cu128 import,
an A40 CUDA calculation, the persistent tokenizer/data cache, SQLite integrity,
protected-file detection, runner metadata, and the pinned TPE sampler all pass. Classical
execution now runs a mandatory cache/database/registration/PyTorch/A40/CUDA preflight
before creating an Optuna trial or reserving an experiment. Campaign
`tpe-seed42-20260805` is ready at exactly 0/25; no active optimizer database or GPU process
exists.

## TPE seed-42 completion — 2026-08-06

Campaign `tpe-seed42-20260805` completed 25/25 attempts: 23 valid compatible results
and two counted candidate OOM failures (attempts 1 and 20). Attempt 21 is best-quality
(1.097702 BPB), attempt 22 is fastest (216.16 cached tok/s), attempt 15 has the lowest
inference allocation (95.04 MiB), and attempt 23 is the practical balance (1.099956
BPB, 199.12 tok/s, 142.70 MiB).

The authoritative database, active Optuna database, candidate sources, events, logs,
and exports were synchronized locally. Checkpoints 15, 21, 22, and 23 were copied and
verified by SHA-256. Their manifest is
`results/preserved_checkpoints/tpe-seed42-20260805-manifest.json`; the complete
non-causal analysis is `results/tpe-seed42-20260805-analysis.json`. The dashboard now
features these four representatives without presenting speed or memory as TPE
objectives.

CMA-ES has not started. It must begin with a fresh optimizer-state database so no TPE
history can leak into its proposals.

## CMA-ES seed-42 completion — 2026-08-06

Campaign `cmaes-seed42-20260805` completed 25/25 attempts with no failures and 25
protected cached-inference-compatible results. Its exact Optuna state maps one-to-one
to the authoritative SQLite rows and configurations. The four-point frontier is
attempts 15, 19, 22, and 23. Attempt 15 is best-quality (1.108287 BPB); attempt 22 is
fastest and lowest-memory (205.48 tok/s, 119.95 MiB).

All result artifacts and the active CMA-ES optimizer database were synchronized
locally. Frontier checkpoints 15, 19, 22, and 23 were transferred individually through
the PTY-safe transport and verified against remote SHA-256 values. The dashboard,
analysis, manifest, and research ledger now expose the completed campaign without
claiming that speed or memory drove CMA-ES proposals.

## MOTPE seed-42 completion — 2026-08-07

Campaign `motpe-seed42-20260805` completed 25/25 attempts. Attempt 1 OOMed and counts;
the other 24 results completed compatible protected cached inference. The five-point
frontier is attempts 21, 22, 23, 24, and 25. Featured representatives are attempt 22
(best quality, 1.098112 BPB), attempt 25 (fastest, 229.00 tok/s), attempt 23
(lowest memory, 122.45 MiB), and attempt 24 (low-memory quality balance).

The authoritative database, full logs/events/candidates, active Optuna state, and the
watchdog's final database and optimizer snapshots are synchronized locally. Their
SHA-256 values match `results/motpe-seed42-autostop-report.json`. The dashboard and
research ledger now describe MOTPE correctly as a three-objective optimizer.

The MOTPE checkpoints themselves are unavailable. The RunPod checkout retained a
legacy `results/checkpoints -> /tmp/autoresearch-checkpoints` symlink, and stopping the
container cleared those weights. This does not change recorded discovery metrics, but
selected MOTPE configurations must be retrained before downstream or quantization
evaluation. `scripts/ensure_persistent_checkpoint_storage.py`, the launcher gate, and
the runner now repair/reject ephemeral checkpoint storage before any future attempt is
reserved.

## Random Search seed-42 completion — 2026-08-07

Campaign `random-search-seed42-20260805` completed exactly 25/25 attempts. Attempts 1
and 12 OOMed and count; the other 23 results completed protected cached inference. Its
six-point nondominated frontier is attempts 6, 7, 9, 10, 22, and 25. Attempt 10 has
the best quality (`1.103809` BPB), attempt 7 is fastest (`208.89` cached tok/s), and
attempt 25 uses the least inference allocation (`131.07` MiB).

The authoritative database and Optuna database match the remote SHA-256 values
`450567a5de578fa5106bf3ce43873fe335211612bb1a4b8d8ea93a2c2f37ab9f` and
`da15516c318c184f81d12b94c715de33cced48c5a5d81d0cef84cd16998ce4cc`.
All six frontier checkpoints were copied locally and independently hash-recorded in
`results/preserved_checkpoints/random-search-seed42-20260805-manifest.json`. The full
non-causal analysis is `results/random-search-seed42-20260805-analysis.json`; the
dashboard Evidence view loads it automatically.

## NSGA-II seed-42 completion — 2026-08-08

Campaign `nsgaii-seed42-20260805` completed exactly 25/25 attempts with training seed
42, algorithm seed 20260767, and the frozen classical search-space hash. Attempts 1
and 10 OOMed and count; the other 23 attempts completed training and protected cached
inference. SQLite integrity and foreign-key checks pass.

The complete nondominated set is attempts 6, 12, 13, 14, 16, 19, 20, 24, and 25.
Attempt 13 has the best quality (`1.107744` BPB, `166.05` cached tok/s, `249.89` MiB),
attempt 20 is fastest (`214.15` tok/s, `1.162595` BPB, `182.38` MiB), attempt 12 is the
lowest-memory frontier point (`133.57` MiB, `1.143242` BPB, `183.17` tok/s), and
attempt 14 is the strongest quality-speed representative (`1.114167` BPB, `188.96`
tok/s, `270.89` MiB). These are single-seed discovery results, not confirmations.

The original Pod volume was recovered in RunPod's zero-GPU mode after an incomplete
automatic migration exposed an empty replacement volume. The authoritative database,
Optuna state, logs, candidates, events, and four frozen representative checkpoints
were synchronized from the original volume. The auditable summary is
`results/nsgaii-seed42-20260805-analysis.json`; dashboard roles are frozen in
`evaluation/nsgaii_seed42_case_study.json`.

## Discovery freeze and primary confirmation plan — 2026-08-10

All three AutoPareto discovery campaigns and all five matched seed-42 classical
controls are complete. The authoritative database passes SQLite integrity and
foreign-key checks. The complete frozen AutoPareto selector contains 11 unique exact
candidate sources under its preregistered per-campaign rule.

To control GPU cost without silently changing that discovery record, a second,
explicitly disclosed staging rule applies the same four roles globally to the frozen
11-recipe pool. It selects experiments 124 (lowest memory within 1% quality), 134
(best quality), 136 (largest remaining exclusive hypervolume contribution), and 137
(fastest within 1% quality). Their exact candidate paths and SHA-256 hashes are frozen
in `results/autopareto_primary_confirmation_plan.json`.

The next GPU phase is exactly 12 independent retrainings: these four unchanged recipes
on seeds 0, 1, and 2. Each recipe/seed pair uses its own one-attempt confirmation
campaign because campaign training seeds are immutable. No agent proposes changes and
no downstream result may influence this confirmation panel. The local gate is:

```bash
python3 scripts/run_autopareto_primary_confirmation.py --validate-only
```

The Pod remains stopped until the local discovery freeze is committed and the A40
environment, persistent cache, persistent checkpoint directory, and database hashes
pass preflight.

## Primary confirmation completion — 2026-08-12

The frozen four-recipe by three-seed primary AutoPareto confirmation panel completed
all 12/12 runs. All 12 training runs completed, all 12 passed protected cached
inference, and all 12 checkpoints were synchronized locally and matched their remote
SHA-256 values. SQLite integrity is `ok` with zero foreign-key violations.

Confirmed mean quality / speed / allocated inference memory:

- discovery experiment 124: `1.097739` BPB / `244.57` tok/s / `132.20` MiB;
- discovery experiment 134: `1.094745` BPB / `299.38` tok/s / `159.88` MiB;
- discovery experiment 136: `1.125060` BPB / `368.90` tok/s / `105.20` MiB;
- discovery experiment 137: `1.098827` BPB / `360.01` tok/s / `141.38` MiB.

`results/autopareto_primary_confirmation_analysis.json` is the authoritative summary.
The dashboard now recognizes the independent one-attempt confirmation campaigns as
three-seed confirmations of their frozen discovery parents. Primary confirmation is
complete; downstream language evaluation is the next scientific phase, followed by
deployment characterization and quantization on the final selected models.

## Downstream execution gate — 2026-08-14

The first downstream invocation stopped on model 1 of 11 before recording a
scientific result. Held-out causal slices were non-contiguous while frozen candidate
code flattened targets with `view`. The failed infrastructure run remains preserved.
The evaluator now makes the aligned slices contiguous without changing token values,
rejects incomplete or non-finite result bundles, writes each model atomically, and
verifies nine task metrics, 18 fixed generations, raw JSON, and SQLite integrity
before proceeding. Local selection preflight reports all 11 models ready. A one-model
GPU smoke/canary must pass before the complete suite restarts.

## Downstream language evaluation completion — 2026-08-14

The frozen 11-model suite is complete: 11 completed runs, 99 task metrics, 198 fixed
generations, and 11 raw result bundles. SQLite integrity is `ok`, foreign-key
violations are zero, and every run has exactly nine metrics, 18 generations, and a
raw JSON bundle. The selection, suite, and evaluator SHA-256 values remained frozen.

- AutoPareto confirmed quality: held-out BPB `1.157651`, LAMBADA perplexity
  `13.965839`, exact match `0.209004`, BLiMP macro accuracy `0.713881`, PIQA
  length-normalized accuracy `0.569097`, ARC-Easy length-normalized accuracy
  `0.344276`;
- AutoPareto confirmed fast quality: LAMBADA exact match `0.211915`, the highest in
  the frozen panel;
- AutoPareto confirmed low memory: BLiMP macro accuracy `0.721701`, the highest in
  the frozen panel;
- AutoResearch compatible reference: held-out BPB `1.161720`, LAMBADA perplexity
  `15.470450`, exact match `0.196390`, BLiMP `0.704940`, PIQA `0.569641`, and
  ARC-Easy `0.335017`.

TPE and Random Search have slightly lower held-out BPB, while different AutoPareto
representatives lead several language metrics. This is evidence of useful trade-offs,
not universal dominance. Downstream scores were never used during search or model
selection. Two zero-output infrastructure interruptions are archived separately in
`results/downstream_interruptions/`; they are not scientific runs. Next:
deployment-workload characterization, then separately reported quantization.

## Final deployment characterization ready — 2026-08-14

The post-downstream deployment study is frozen and locally validated. It contains
exactly five models: the seed-1 baseline and the confirmed AutoPareto quality,
low-memory, fast-quality, and hypervolume trade-off representatives. Their exact
candidate and checkpoint SHA-256 values are recorded in
`results/final_deployment_selection.json`.

Each model will run 16 A40 workloads: contexts 256, 896, 1024, and 1536 crossed with
inference batches 1, 2, 4, and 8, generating 128 tokens with one warm-up and five
measured repetitions. The runner records prefill latency, cached decode throughput,
allocated and reserved VRAM, KV-cache growth, correctness, and explicit workload
failures. It never overwrites headline discovery or confirmation metrics.

Deployment rows now belong to immutable evaluation sessions. All 233 historical rows
were preserved under `historical-pre-autopareto-a40-20260727`; the frozen legacy final
session `final-deployment-a40-20260814` contains zero measurements and is marked
`superseded_before_execution`. Its fixed warm-up protocol did not adequately control
A40 timing drift.

## Stabilized inference timing gate ready — 2026-08-16

Repeated discovery timing for experiment 137 rose materially across successive
repetitions, so the fixed warm-up count is not strong enough for final paper claims.
Historical search, confirmation, and 233 deployment rows remain unchanged and remain
valid as discovery evidence under their recorded evaluator.

The corrected protocol is `autopareto_inference_stabilized_v2`. It requires two
consecutive five-run warm-up blocks with median throughput drift at most 2% and each
block's coefficient of variation at most 5%, with a hard limit of 30 warm-ups per
pass. It then records three passes of ten repetitions and applies the same stability
test to each measured pass. Headline speed uses synchronized wall-clock time; CUDA
events are diagnostics. All raw repetitions, warm-ups, failures, GPU telemetry,
allocated/reserved VRAM, and KV-cache measurements are append-only in SQLite.

The frozen canary contains only baseline experiment 4 and discovery experiment 137 at
context 256, batch 1, and 256 generated tokens. Candidate/checkpoint and evaluator
dependency hashes are in `results/inference_timing_canary_selection.json`. Session 3
was superseded before execution when dependency hashing was strengthened; it contains
zero measurements. Session 4 is planned and contains zero measurements. SQLite
integrity is `ok`, foreign-key violations are zero, 23 focused tests pass, and the
broader lightweight suite has 45 passing tests plus five expected skips; its one
collection error is solely the Mac environment lacking PyTorch.

Exact next action: synchronize the current source, database, both selected candidates,
and both selected checkpoints to one NVIDIA A40; run
`python3 scripts/run_inference_timing_canary.py --execute`; retrieve the immutable
session and raw JSON; stop the Pod. Do not execute the old full deployment matrix.

## Stabilized inference timing canary completed — 2026-08-16

Immutable deployment session 4 (`inference-stability-canary-a40-20260816-v2`)
completed on one NVIDIA A40. Both frozen checkpoint and candidate hashes matched the
local selection. The baseline recorded 35 adaptive warm-ups plus 30 measurements;
AutoPareto discovery experiment 137 recorded 40 adaptive warm-ups plus 30
measurements. All three measured passes per model passed the preregistered drift,
coefficient-of-variation, cached-correctness, dual-timing, and telemetry checks.

At context 256, batch 1, and 256 generated tokens:

- baseline experiment 4: median `166.8675` tok/s, `249.8799` MiB allocated VRAM,
  `320.0` MiB reserved VRAM, and `3.9844` MiB incremental KV-cache memory;
- AutoPareto experiment 137: median `419.9379` tok/s, `141.3784` MiB allocated VRAM,
  `192.0` MiB reserved VRAM, and `1.4941` MiB incremental KV-cache memory.

Relative to the same-session baseline, experiment 137 is `2.516x` as fast, uses
`43.42%` less allocated VRAM, and uses `62.50%` less incremental KV-cache memory.
SQLite integrity is `ok`, foreign-key violations are zero, and the independent canary
verifier reports no errors. The raw JSON and all 135 append-only warm-up/measurement
rows are synchronized locally. The old fixed-warm-up matrix remains blocked. Next:
freeze the replacement stabilized deployment matrix before running it.

## Stabilized final deployment matrix frozen — 2026-08-17

The successful two-model canary unlocks a new immutable five-model deployment study;
the old fixed-warm-up session remains preserved and blocked. The matrix contains the
same-session baseline and four mechanically selected confirmed AutoPareto roles: best
quality, lowest memory within the one-percent quality budget, fastest within that
budget, and largest exclusive hypervolume contribution.

Each exact checkpoint runs 16 workloads: contexts 256, 896, 1,024, and 1,536 crossed
with inference batches 1, 2, 4, and 8, with 128 generated tokens. Every workload uses
the canary-validated adaptive stabilization protocol, three passes, and ten measured
repetitions per pass. Candidate, checkpoint, suite, evaluator, runner, and stability
policy hashes are frozen in `results/final_deployment_stabilized_selection.json`.
The suite is `evaluation/final_deployment_stabilized_suite.json`; the only authorized
GPU entry point is `python3 scripts/run_stabilized_deployment_suite.py --execute`.

Completed workloads, explicit OOM/unsupported failures, warm-ups, measurements,
wall-clock and CUDA timing, telemetry, allocated/reserved VRAM, and KV-cache values are
append-only. An external interruption that leaves partial stabilized evidence cannot
be resumed under a changed GPU state; it requires a newly versioned session. This
protects the paper from mixing clock and thermal states across Pod lifetimes.

## Stabilized deployment recovery completed — 2026-08-17

Session 5 completed the frozen 5-checkpoint × 16-workload matrix with 62 stable and
18 terminal unstable pairs. It remains unchanged. Session 6 retried exactly those 18
pairs under the same frozen protocol: 4 became stable and 14 remained terminally
unstable (11 throughput-drift outcomes, 3 warm-up-limit outcomes). No retry
correctness failure was recorded. Final evidence is 66/80 stable pairs and 14/80
explicitly unresolved stability pairs; unstable pairs have no final speed or memory
value and must not be plotted as if they did.

The CPU-only audit/export is `scripts/analyze_stabilized_recovery.py`, with output at
`results/final_deployment_stabilized_summary.json`. It verifies both sessions,
applies the append-only replacement rule, and checks SQLite integrity and foreign
keys. The Pod was stopped after retrieval. Next work is documentation/dashboard
presentation and paper tables; any larger-warm-up sensitivity diagnostic is optional,
separate, and must use a new session ID.

## Remaining research phases scoped — 2026-08-18

The downstream language suite is complete: 11 selected models, 99 aggregate task
metrics, and 198 fixed-prompt generations. Quantization, cross-GPU replication, and
larger-model transfer remain unrun. Their bounded selection is recorded in
`evaluation/next_phase_selection.json` and explained in `docs/NEXT_PHASE_PLAN.md`.
The recommended first GPU phase is the preregistered BF16-versus-INT8 quantization
study on the five exact deployment representatives, with INT4 conditional on a
verified A40 kernel. No new Pod should be started until the phase and budget are
approved.

## Phase 1 stabilized inference correction — 2026-08-24

The NON-SCIENTIFIC Phase 1 smoke revealed that the staged Phase 1 executor was still
calling `inference_benchmark.py`, whose `benchmark_config.WARMUP_ITERATIONS = 2`
implements the legacy fixed-two-warmup measurement. It was not invoking the later
protected adaptive deployment evaluator. This did not affect training or any frozen
recipe.

Before Stage B, an inference-only seven-checkpoint rebenchmark was run on one A40 at
the frozen Phase 1 headline workload using
`autopareto_inference_stabilized_v2`: five-run warm-up blocks, two adjacent stable
blocks at <=2% median drift and <=5% CV, a 30-warm-up limit, three passes, and ten
measurements per pass. The headline policy is the median of all 30 successful
measurements; maxima are diagnostic only. Cache correctness passed for every loaded
checkpoint. The first session was unresolved for all seven models. A versioned retry
completed only the MOTPE exact-recipe recovery checkpoint (experiment 300 checkpoint
for source experiment 234) at 268.3438 tok/s; the other six remain unresolved under
the preregistered gate. AutoPareto 299 achieved stable passes near 476.7 and 493.4--
493.6 tok/s across the two sessions but failed a later measured pass, so its ~480 tok/s
smoke regime is reproducible as a diagnostic steady-state observation but is not a
canonical serving metric.

The old fixed-two-warmup values remain preserved in SQLite and the historical
selection/confirmation artifacts. The new raw outputs and comparison table are
`results/phase1/stabilized_inference/` and
`results/phase1/phase1_stabilized_inference_rebenchmark.md`; no historical value was
overwritten. This dated note records the pre-Stage-B state and is superseded by the
completed 21-run equal-token results in `results/phase1/phase1_equal_token_results.json`.

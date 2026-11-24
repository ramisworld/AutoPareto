# Camera-ready E0 and E1: pod runbook

> **Amendment 1 (2026-11-04, post hoc, PI-approved): this is the active plan.**
>
> - **Active plan:** `evaluation/camera_ready_e0_e1_plan_amendment1.json`, sha256 **`89f2901588fceec18ab93d4e74ed652b8f6c6df9329eb8ba1d3a7ead128f60dc`**.
> - **Output root:** `results/camera_ready_2026/amendment1/`.
>
> **History.** The original plan below (sha256 `806a6397…`) ran E0 on 2026-11-04 and **halted**: run 2 made 666 updates against [669, 681]. The original plan file and its results under `results/camera_ready_2026/E0/` are kept unchanged. The diagnosis is in `audit/camera_ready_2026/14_E0_RESULT.md`.
>
> **What changed.** Only one rule changed: run 2's update count is now recorded as a diagnostic, like throughput and memory, instead of halting. The plan's `amendment` block lists every changed field.
>
> **Unchanged:** the data gate, integrity gates, BPB tolerances, run 1's exact tokens and updates, E1 and the analysis plan.
>
> **Rerun.** E0 runs again in full under the new hash. In every command below, replace the plan sha256 with the amendment-1 hash and `results/camera_ready_2026/{E0,E1}` with `results/camera_ready_2026/amendment1/{E0,E1}`.

**Status (original plan):** prepared and validated locally on 2026-11-02. No GPU work has run. Nothing is committed.

| Item | Value |
|---|---|
| Pre-registration | `evaluation/camera_ready_e0_e1_plan.json` |
| **Plan sha256** | **`806a63970258627092d929d4602e54e36a2412e0656a63298e17307c6c829340`** |
| Executor | `scripts/run_camera_ready_e0_e1.py` (validate-only by default) |
| Analysis | `scripts/analyze_camera_ready_e1.py` (validated on synthetic data only) |
| Source plan | `audit/camera_ready_2026/08_EXPERIMENT_PLAN.md`, sections E0 and E1 |
| Output root | `results/camera_ready_2026/{E0,E1}/runs/<run_id>/` |

The plan stores the sha256 of both scripts and of every file they run. Editing any of them, or the plan,
changes the hash. After a change, regenerate the plan, update the hash above, and re-run validate-only.
Commit the plan and both scripts before launch (P3). Every run record and the E0 verdict carry the plan
hash.

## 1. What runs

| Stage | Runs | Contents |
|---|---|---|
| E0 | data gate + 2 | (0) CPU replay of the training token stream. It must match the panel's 1,536-microbatch stream hash `180e0a5b…`, tokenizer and shard hashes. (1) AP-Q (`55a7fd8b…`), 201,326,592 tokens, seed 3. It is compared with `phase1-autopareto_292-seed3`: held-out 1.139616, validation 1.076589, 1,536 updates. (2) AR-ref (`b3a961ee…`), 300 s fixed time, seed 0. It is compared with SNAPDB training_run 23731: validation 1.102901, 675 updates. |
| E1 | 24 | AR-ref and AP-Q × {88,604,672; 157,286,400; 201,326,592 tokens; 300 s} × seeds {6, 7, 8}. The fixed order is in the plan: seed blocks, a cyclic Latin square of budgets, and paired recipes back to back. |

**E0 halt rule.** E0 halts, and E1 is refused, on any of these:
- the data gate fails;
- either E0 run fails its integrity gates twice;
- a BPB difference exceeds 0.005;
- run 1 is not exactly 201,326,592 tokens in 1,536 updates;
- run 2's updates fall outside [669, 681], which is ±1% of 675.

**Integrity gates (every attempt).**
- Exact tokens and updates.
- The milestone step equals the observed final step.
- Recipe, derived-source, checkpoint and evaluator-input hashes.
- The held-out shard hash.
- Checkpoint validation within 0.002 of in-process validation.
- BPB in (0.5, 1.7).
- All 20 implementation hashes unchanged.

**Retry rule.** A failed attempt is re-run once on the same seed, in `<run_id>-attempt2`. If that fails, the run is reported failed. No seeds are added.

### How each cell is executed (no protected code is edited)

**Fixed-token cells.** These use Phase-1 `run_one()` and `evaluate_one()` from `scripts/run_phase1_equal_tokens.py`, unchanged. Only `PHASE1_ROOT` is redirected to `results/camera_ready_2026/<stage>`, one call at a time. A per-run plan view supplies the budget and seed. Validate-only checks two things:
- the protected transform reproduces the panel's executed derived sources byte for byte (`c8a802ba…` for AP-Q seed 3, `aa74738a…` for AR-ref seed 3);
- the transform accepts every E0/E1 budget and seed.

There are no budget or seed blockers:
- All three budgets are multiples of 4 × 131,072, so each run writes 4 milestones.
- The seed-{3,4,5} and 201M assertions exist only in Phase-1 `main()`, `ordered_runs()`, `integrity_gate()` and `require_token_stream_verification()`. E0/E1 do not call those functions.

**300 s cell.** This uses the historical fixed-time path: `runner.py` → `protected/train_launcher.py`.
- `runner.py` itself is not used. It rejects confirmation seeds outside 0–2 and writes to `results/results.db`.
- The executor copies the historical launcher into a per-run isolate, as `runner.py` did. The launcher is commit `c200ccb`, sha256 `aa9dc8b2…`, git blob `8f100f4c…`, read from `study/contracts_2026/historical_reference/protected/train_launcher.py` or from git.
- The isolate also holds the frozen recipe as `train.py` and the hash-checked protected modules it imports.
- The run uses `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` and a 900 s timeout. The seed is forced by the same `_force_seed` wrapper as Phase 1.
- The final checkpoint is scored by the same evaluator as the token cells (`scripts.evaluate_phase1_checkpoints --heldout`), on a one-entry manifest.
- The in-process validation BPB is the quantity behind 23731 and the −0.0067 seeds-0–2 contrast. It is kept as `validation_bpb_in_process`.
- AR-ref's historical runs used the earlier launcher `86cafb0` (`42e828a0…`). It differs only by lacking the post-run contract validation; training is identical.

## 2. Pod environment (from the Phase-1 run records)

The panel ran in these conditions:
- root `/workspace/autopareto`, a dirty checkout at `5bf86b57`;
- Python 3.10.18, torch 2.9.1+cu128, CUDA 12.8, driver 570.211.01, one A40;
- all subprocesses launched with `uv run --locked python`, the Phase-1 `command_prefix()`, because uv was on PATH, using the project venv built from `uv.lock` (sha256 `262a4d55…`);
- data cache `~/.cache/autoresearch` → `/workspace/.cache/autoresearch`;
- held-out shard cache `~/.cache/autopareto/downstream/`.

Recommended root for this run: **`/workspace/autopareto-camera-ready`**, a fresh copy of this worktree.
- Do not overwrite `/workspace/autopareto`. Its untracked phase-1 harness files are the only copies of the code that executed the panel (REPRO M13). Back it up first.
- If you prefer to run inside `/workspace/autopareto`, the executor works there too. Every hashed file must then match, and validate-only will say so.

## 3. Commands

Replace `<POD>` with your SSH target. Do not paste the endpoint into any committed file (P6).

**On the Mac (once).** Provide the panel lockfile and copy the code. The lockfile is not in this
branch:

```bash
cd /Users/Rami/Documents/dev/autopareto-camera-ready
git -C /Users/Rami/Documents/dev/autopareto show autopareto/research-infrastructure:uv.lock > /tmp/uv.lock
shasum -a 256 /tmp/uv.lock   # must be 262a4d55eaa24f47c98ca44cc82ff1660d5daba73df0b1e78246cceac2c429b4
rsync -a --exclude .git --exclude 'results/camera_ready_2026/' ./ <POD>:/workspace/autopareto-camera-ready/
rsync -a /tmp/uv.lock <POD>:/workspace/autopareto-camera-ready/uv.lock
```

**On the pod: prepare** (P1/P5; restore from `/workspace/PRESERVED_20260917T021347Z` first):

```bash
tar czf /workspace/PRESERVED_pod_checkout_$(date -u +%Y%m%dT%H%M%SZ).tgz --exclude='*.pt' --exclude=.venv -C /workspace autopareto
cd /workspace/autopareto-camera-ready
uv sync --frozen                    # Python 3.10.18 + torch 2.9.1+cu128 from uv.lock
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
uv run --locked python scripts/ensure_persistent_cache.py   # binds ~/.cache/autoresearch -> /workspace/.cache/autoresearch
```

**Validate-only** must print `FAIL: 0` and `POD_ONLY: 0`:

```bash
uv run --locked python scripts/run_camera_ready_e0_e1.py \
  --expect-plan-sha256 806a63970258627092d929d4602e54e36a2412e0656a63298e17307c6c829340
```

**E0** (about 45 min; run detached):

```bash
nohup uv run --locked python scripts/run_camera_ready_e0_e1.py --stage E0 --execute \
  --expect-plan-sha256 806a63970258627092d929d4602e54e36a2412e0656a63298e17307c6c829340 \
  > /workspace/camera_ready_E0_console.log 2>&1 &
tail -f /workspace/camera_ready_E0_console.log
python3 -m json.tool results/camera_ready_2026/E0/e0_verdict.json   # status must be PASS
```

**E1, only after E0 PASS** (about 6.4 h). The executor refuses unless `e0_verdict.json` says PASS under the same plan hash:

```bash
nohup uv run --locked python scripts/run_camera_ready_e0_e1.py --stage E1 --execute \
  --expect-plan-sha256 806a63970258627092d929d4602e54e36a2412e0656a63298e17307c6c829340 \
  > /workspace/camera_ready_E1_console.log 2>&1 &
```

If the pod or process dies, re-run the same command with `--resume`.
- Runs that passed are skipped.
- An unfinished directory counts as the failed first attempt; the run continues in `-attempt2`.
- Directories are never reused.

**Analysis** (pod or Mac; matplotlib 3.10.8 is in `uv.lock`):

```bash
uv run --locked python scripts/analyze_camera_ready_e1.py --runs-root results/camera_ready_2026/E1
# on the Mac, after copy-back: /Users/Rami/Documents/dev/autopareto/paper/.venv/bin/python scripts/analyze_camera_ready_e1.py --runs-root results/camera_ready_2026/E1
```

## 4. Expected wall-clock and A40-hours

The estimates come from the Phase-1 records:
- steady throughput: AR-ref 289k tok/s, AP-Q 523k tok/s;
- non-steady training overhead: about 225 s per run (cold per-run compile, startup, in-process validation, checkpoint writes);
- evaluation of 4 milestones plus held-out: about 470 s for AR-ref and 300 s for AP-Q, from the panel's evaluation timestamps;
- a 300 s run takes about 520 s end to end.

| Run | AR-ref | AP-Q |
|---|---|---|
| 88.6M tokens (train + eval) | ~17 min | ~12 min |
| 157.3M tokens | ~21 min | ~14 min |
| 201.3M tokens | ~23 min | ~15 min |
| 300 s fixed time (train + eval) | ~13 min | ~11 min |

| Stage | Wall-clock ≈ A40-hours | Cost at $0.44/h |
|---|---|---|
| E0 (data gate ~15 min + 2 runs) | **0.73 h** | $0.32 |
| E1 (3 seed blocks × 2.15 h) | **6.44 h** | $2.83 |
| **Total** | **≈ 7.2 A40-h** | **≈ $3.15** |

- A retry adds 0.15–0.35 h.
- Re-check the hourly price in the RunPod console.
- 08 and `12_GPU_READINESS_REPORT.md` budget about 4.1 h for E1. That figure assumed about 4 min of overhead per run. The panel records show about 3.7 min of non-steady training plus 4–8 min of evaluation per run.

## 5. Copy back

Everything except checkpoints and compile caches is small (logs, JSON, step timings):

```bash
rsync -a --exclude 'compile_cache/' --exclude '*.pt' \
  <POD>:/workspace/autopareto-camera-ready/results/camera_ready_2026/ \
  /Users/Rami/Documents/dev/autopareto-camera-ready/results/camera_ready_2026/
rsync -a <POD>:/workspace/camera_ready_E0_console.log <POD>:/workspace/camera_ready_E1_console.log \
  /Users/Rami/Documents/dev/autopareto-camera-ready/results/camera_ready_2026/
```

Required per stage:
- `stage_manifest-*.json` (environment, GPU, kernel-cache revision, all checks);
- `stage_events.jsonl`;
- `E0/data_identity.json`;
- `E0/e0_verdict.json`;
- `E1/e1_execution_summary-*.json`;
- `E1/analysis/`.

Required per run:
- `camera_ready_run_record.json`;
- `training_metrics.json` or `fixed_time_training_metrics.json`;
- `checkpoint_evaluations.json`;
- `milestones/manifest.json` or `fixed_time_checkpoint_manifest.json`;
- `step_timing.jsonl`, `timing_summary.json`, `telemetry.json`, `run_metadata.json`;
- `events.jsonl`, `training.log`, `evaluation.log`;
- `phase1_train.py` or `isolate/`.

Checkpoints (`final.pt` and `milestones/step-*.pt`) total about 11 GB. Keep them on `/workspace` and
back them up separately; do not delete them.

## 6. Local validate-only output (2026-11-02, Mac, no GPU)

```
plan: evaluation/camera_ready_e0_e1_plan.json  sha256: 806a63970258627092d929d4602e54e36a2412e0656a63298e17307c6c829340
[      ok] plan_structure_and_sources                         2 E0 + 24 E1 runs; both frozen sources hash-verified; batch 131072
[      ok] implementation_sha256                              20 files match the pre-registered hashes
[      ok] phase1_source_identity                             AP-Q == Phase-1 autopareto_292 source and AR-ref == Phase-1 autoresearch_compatible source (same path, same sha256)
[      ok] phase1_transform_reproduces_panel_derived_sources  {"phase1-autopareto_292-seed3": true, "phase1-autoresearch_compatible-seed3": true}
[      ok] phase1_transform_accepts_every_budget_and_seed     19 fixed-token runs: transform accepted; 4 milestones reachable per budget
[      ok] historical_fixed_time_launcher                     resolved from study/contracts_2026/historical_reference/protected/train_launcher.py
[      ok] fixed_time_protected_constants                     {... TRAINING_BUDGET_SECONDS 300, TRAINING_MAX_OVERRUN_SECONDS 15, TRAINING_TIMEOUT_SECONDS 900, CUDA_ALLOCATOR backend:cudaMallocAsync, prepare.TIME_BUDGET 300 ...}
[      ok] fixed_time_isolate_inputs                          ["prepare.py", "benchmark_config.py", "event_log.py", "protected/inference_adapter.py"]
[      ok] output_root                                        {"existing_run_dirs": {"E0": [], "E1": []}, "exists": false, ...}
[    info] e0_verdict                                         absent: E1 --execute is blocked until E0 PASS
[POD_ONLY] uv_lockfile                                        uv is on PATH ... needs <root>/uv.lock (sha256 262a4d55...; private branch autopareto/research-infrastructure:uv.lock)
[POD_ONLY] python_version                                     3.10.20 via this_interpreter (panel: 3.10.18)
[POD_ONLY] torch_version                                      2.9.1 via this_interpreter (panel: 2.9.1+cu128)
[POD_ONLY] cuda_gpu                                           CUDA available=False gpu=None; an NVIDIA A40 is required
[POD_ONLY] gpu_idle                                           nvidia-smi unavailable
[POD_ONLY] persistent_output_root                             /workspace is absent (not a RunPod pod)
[POD_ONLY] data_cache                                         ~/.cache/autoresearch absent (tokenizer, 10 training shards, validation shard_06542 live on the pod)
[POD_ONLY] heldout_shard_cache                                ~/.cache/autopareto/downstream/915333b4...-shard_06541.parquet absent
summary: {'ok': 9, 'FAIL': 0, 'POD_ONLY': 8, 'info': 1}  -> validated
```

All 8 `POD_ONLY` items are expected off the pod. Locally, `uv_lockfile` is the only one that needs an
action: `uv.lock` must be supplied (section 3). With `--execute`, each `POD_ONLY` check becomes a hard
failure.

Exact subprocess commands for E0, as printed by validate-only. `$ROOT` is the repository root; on the pod it is
`/workspace/autopareto-camera-ready`. The listing for all 26 runs is printed by the same command.

```
# E0 #01 e0-apq-tokens201326592-seed3 (retry dir e0-apq-tokens201326592-seed3-attempt2)
cd $ROOT && env AUTOPARETO_PHASE1_EVENT_LOG=$ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/events.jsonl CUDA_CACHE_PATH=$ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/compile_cache/cuda TORCHINDUCTOR_CACHE_DIR=$ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/compile_cache/torchinductor TRITON_CACHE_DIR=$ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/compile_cache/triton uv run --locked python -m protected.phase1_train_launcher --source $ROOT/results/candidates/autopareto-seed0-20260731/0022-train.py --source-sha256 55a7fd8bf32924f2a35ebc095c5206f3503e671a9d7a9646b94979c8ae49259a --token-budget 201326592 --seed 3 --run-dir $ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3 --transformed $ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/phase1_train.py --checkpoint $ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/final.pt --milestone-dir $ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/milestones --metrics $ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/training_metrics.json --event-log $ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/events.jsonl --model-id autopareto_292 --telemetry-interval-seconds 2.0 --telemetry-mode enabled --model-order-position 1
cd $ROOT && uv run --locked python -m scripts.evaluate_phase1_checkpoints --candidate $ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/phase1_train.py --milestone-manifest $ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/milestones/manifest.json --output $ROOT/results/camera_ready_2026/E0/runs/e0-apq-tokens201326592-seed3/checkpoint_evaluations.json --heldout

# E0 #02 e0-arref-time300s-seed0 (retry dir e0-arref-time300s-seed0-attempt2)
cd $ROOT/results/camera_ready_2026/E0/runs/e0-arref-time300s-seed0/isolate && env CUDA_CACHE_PATH=$ROOT/results/camera_ready_2026/E0/runs/e0-arref-time300s-seed0/compile_cache/cuda PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync TORCHINDUCTOR_CACHE_DIR=$ROOT/results/camera_ready_2026/E0/runs/e0-arref-time300s-seed0/compile_cache/torchinductor TRITON_CACHE_DIR=$ROOT/results/camera_ready_2026/E0/runs/e0-arref-time300s-seed0/compile_cache/triton uv run --locked python protected/train_launcher.py --train train.py --checkpoint $ROOT/results/camera_ready_2026/E0/runs/e0-arref-time300s-seed0/final.pt --metrics $ROOT/results/camera_ready_2026/E0/runs/e0-arref-time300s-seed0/fixed_time_training_metrics.json --seed 0 --event-log $ROOT/results/camera_ready_2026/E0/runs/e0-arref-time300s-seed0/events.jsonl --campaign-id camera-ready-2026-e0-arref-time300s-seed0
cd $ROOT && uv run --locked python -m scripts.evaluate_phase1_checkpoints --candidate $ROOT/results/camera_ready_2026/E0/runs/e0-arref-time300s-seed0/isolate/train.py --milestone-manifest $ROOT/results/camera_ready_2026/E0/runs/e0-arref-time300s-seed0/fixed_time_checkpoint_manifest.json --output $ROOT/results/camera_ready_2026/E0/runs/e0-arref-time300s-seed0/checkpoint_evaluations.json --heldout
```

### Other local checks

- **Analysis.** `scripts/analyze_camera_ready_e1.py --self-test` passed with fabricated records labelled SYNTHETIC, in a temporary directory outside `results/`. It recovers a known crossover (111.5M), the t-interval (df = 2, t = 4.3027), the exact 27-resample bootstrap, the H-E1a/H-E1b and outcome-table verdicts, the n = 2 path for a missing run, and the synthetic-input guard. It also passed with and without matplotlib.
- **Executor.** The gates, E0 verdict, retry and resume logic were tested offline (12 unittest cases, kept outside the repo):
  - the real panel artifacts of `phase1-autopareto_292-seed3` pass the new token gate unchanged;
  - wrong seed, wrong budget, a replaced checkpoint, and a checkpoint/in-process mismatch are all caught;
  - each E0 halt condition triggers.

## 7. Open items and risks (decide before committing the plan)

1. **`uv.lock` is not in this branch.** It is needed on the pod; see section 3.
2. **Pre-registration choices to confirm.** Each changes the plan hash if altered:
   - held-out BPB is E1's primary metric (validation is secondary; 08 names neither);
   - E0 run 2's update window is [669, 681] rather than literal equality;
   - the data gate runs first in E0.
3. **Two files differ from the hashes in the Phase-1 preflight (M13):** `protected/phase1_train_launcher.py` and `scripts/evaluate_phase1_checkpoints.py`. The versions that ran the panel are not in git. The transform and runtime match, and the executed derived sources reproduce exactly. E0 run 1 is the empirical check.
4. **The flash-attn3 kernel is unpinned.** `kernels.get_kernel("kernels-community/flash-attn3")` fetches without a revision. The executor records the cached revision but does not gate on it. E0 catches numeric drift.
5. **Steady-state metric normalization.** `steady_state_equivalent_seconds` is normalized to 201M tokens for every budget, so E1 reports `steady_state_training_seconds` (the run's own).
6. **No training-subprocess timeout on the token cells.** The Phase-1 `run_one` has none, so watch the console log. The fixed-time cell keeps the historical 900 s timeout and kills the whole process group.

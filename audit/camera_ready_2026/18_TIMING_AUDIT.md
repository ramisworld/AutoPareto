# 18 — Timing audit: decode-measurement stability and the fixed-time training contract (2026-11-05)

No protocol, threshold, timer or campaign state was changed. All measurements below are non-study
runs on the second camera-ready pod (`GPU-9f05dd40…`), using the registered baseline checkpoints.

## Summary
- **Training timer.** It works as designed and is identical to the Codex campaign's. Training is
  GPU-bound and stable to 0.2% per update, and all 300 s of the window are training. One baseline
  disclosure follows in §4.
- **Decode measurement.** The code is byte-identical to the Codex campaign's. Cached decode at batch 1 is
  **CPU-bound**: the GPU is busy 25% of the time, with 618 kernel launches per generated token. The measured
  throughput is therefore the speed of one CPU thread. On this host, single-thread CPU speed varies by
  about 2–3% from second to second, the same size as the registered 2% stability limit. Instability is a
  property of the host's CPU, not of the GPU or the code.

## 1. Code identity with the Codex campaign
Compared against the final Codex manifest (`…_20260917_posttraining_failure_receipt_fix.json`, A40 `GPU-35264a0f…`):
- **Byte-identical:** `inference_benchmark.py`, `benchmark_policy.py` (all thresholds), `inference_model_host.py`,
  `inference_ipc.py`, `protected/inference_adapter.py`, the active inference config, `benchmark_config.py`,
  `candidate_worker.py`, `candidate_sandbox.py`, `contracts.py`, `remote_runner.py`, `protected/phase1_runtime.py`,
  `batching.py`, `study_runtime.py` and the baseline `train.py` (`b2fb8788…`).
- **Changed (Claude-backend amendments), failure handling only:**
  - `train_launcher.py`: a worker-written failure receipt is no longer overwritten by the parent's later classification.
  - `candidate_process.py`: a candidate whose training body returns before the protected stop is now a
    candidate failure. In the Codex campaign the body was re-run, which produced untrained checkpoints at
    BPB 3.2159.
  - Neither changes the timer or the measurement of a normal run.

## 2. Decode benchmark: method as implemented
- One evaluator process holds both models on the GPU. Each model gets a discarded 20-s cached-decode
  warm-up, run one after the other.
- **Bracket:** 8 rounds of baseline-before, candidate, baseline-after windows, each at least 8 s long.
  The baseline is interpolated at the candidate window's midpoint. The search scalar is the median of
  the 8 candidate/baseline ratios.
- **Stability rule:** the ratios' IQR/median must be ≤ 0.02 and their CV ≤ 0.05.
- **Repetition:** a 1024-token prefill, then 127 single-token cached decode steps (bf16 autocast, greedy
  argmax kept on the GPU). `torch.cuda.synchronize()` brackets prefill and the whole decode loop, timed
  with `perf_counter`.
- **Verified sound:**
  - Synchronization fully encloses the timed GPU work.
  - Warm-up is discarded.
  - There are no host↔device transfers in the timed loop: prompts are already on the GPU, and there is no `.item()`.
  - The model and KV cache stay on the GPU throughout.
  - Peak memory is measured separately after the baseline is released.
- **Not controlled:**
  - The GPU clock is not locked. This turned out not to matter (§3).
  - CPU affinity is not set, and the PyTorch thread count is 48.
  - The adapter is eager Python: per layer it runs norm, rotary, `torch.cat` KV growth, `arange` and attention,
    with no CUDA graphs and no compile. That is why one CPU thread sets the speed.
- **Why the ratio does not cancel the noise:** interpolation removes slow drift common to neighbouring
  windows. CPU jitter at the 1–10 s scale differs between the three sequential windows, so it passes
  straight into the ratios.

## 3. Measured root cause
Script: `scripts/pod/decode_timing_diagnostics.py`. Report on the Mac:
`runs/nonstudy_decode_timing_diagnostics_20261105/` (`SHA256SUMS`).

| Test | Result |
|---|---|
| A. Profile of one repetition | GPU kernel time 0.48 s of 1.89 s wall (**25% busy**); 79,139 launches (618 per generated token) |
| B. 20 official windows, unpinned (as the protocol runs) | window tok/s CV 2.3%, IQR/median 2.3%, range 9.7%; **50%** of rolling 8-window sets fail the 2% rule |
| — during B | SM clock **1740 MHz in every sample** (maximum boost); cgroup CPU throttling **0 ms**; process CPU/wall = 1.00 (one thread fully busy, never waiting on the GPU); about 25 involuntary context switches per window; host CPU pressure about 100 ms/s |
| C. CPU-only control (pure Python, no GPU), 60 × 1 s | rate CV 2.5%, IQR/median 2.3%, range 18.8%; its first 8 chunks fail the 2% rule (0.0227) |
| D. 12 windows pinned to one core (diagnostic only) | CV 0.7%, IQR/median 1.2%; **0%** of rolling 8-window sets fail |
| Host | Xeon Gold 6342, 2 sockets × 24 cores × 2 threads, cpufreq governor `schedutil` (idle cores at 800 MHz, turbo to 3.5 GHz), container quota 7.65 CPUs, load 9–13 |

**Interpretation.** The GPU and the cgroup quota are ruled out. Throughput follows the speed of one CPU
thread, which varies on a shared host with frequency scaling: the scheduler moves the thread between
cores whose clocks differ, and co-tenants share the turbo budget. The pure-CPU control shows the same
variation with no GPU at all, and pinning to one core removes most of it.

**Cross-host proxy.** The absolute baseline decode speed tracks single-thread CPU speed:
- Codex comparator host: median 150.3 tok/s (145–158 over 42 attempts); 1 of 42 unstable.
- Old Opus pod: 143.3 tok/s; attempt 1 stable.
- This host: 136–146 tok/s; 2 of 3 bracketed runs unstable.

**Caveats.**
- The diagnostic's in-process clock sampler costs throughput through the Python GIL: absolute speed
  was 111 tok/s against about 145 in protocol runs. B and D both carry this cost.
- D ran after B, so pinning is confounded with time. Its effect should be replicated before any use.
- Earlier hosts' CPU models were never recorded.

## 4. Fixed-time training contract (Contract T)
- **Timeline (`train_launcher._read_events` and `candidate_process.CandidateObserver`):**
  1. The worker builds the model, optimizer and dataloader.
  2. **11 untimed warm-up updates** run (compilation; about 56 s, the first about 38 s), then `WARMUP_STOP`.
  3. In the same process, the candidate state is rebuilt; its initial-state hash must match the warm-up's.
  4. `boundary_ready` fires just before the first scientific forward pass: **the 300-s clock starts here**,
     on the parent's `perf_counter_ns`.
  5. After every optimizer update the worker synchronizes CUDA and blocks on the parent.
  6. The parent sends `STOP` at the first update completing at or after 300.0 s, otherwise `CONTINUE_TIMED <elapsed>`.
  7. Validation, checkpointing and inference all run **after** the stop.
  8. Per-step data loading is **inside** the window, as in the original loop.
- **Measured.**
  - The first timed update takes 1.77 s, against 1.84 s steady, so no compilation leaks into the window.
  - Steady update CV is 0.002. All 300 s are training.
  - Overshoot is at most one update (1.6 s).

| Run | Updates | Tokens | Protected s | s/update | BPB |
|---|---|---|---|---|---|
| This pod, baseline T | 164 | 85,983,232 (= 164 × 524,288) | 301.62 | 1.841 | 1.19913 |
| Codex comparator A40, 2026-09-16 validation | 164 | 85,983,232 | 301.58 | — | 1.19910 |
| Old Opus pod, baseline T | 162 | 84,934,656 | 300.63 | 1.859 | 1.20110 |

- **Disclosure: the baseline's schedule is cut at 93%.** The registered baseline `train.py` drives its LR
  schedule with its own clock, which skips its first 11 steps (`step > 10`). At the protected stop it
  therefore reports `progress` 0.932 (old pod 0.929), so the last part of its warmdown never runs.
  - This is identical in both agent arms and in every baseline registration, so it is not a cross-arm bias.
  - Agents are told to use `contract_progress(...)`, which reports the protected elapsed time.
  - It is a property of the frozen Contract T and should be stated in the paper.
- **Hardware sensitivity.** Fixed-time update counts move by about 1–1.5% between A40s: 162 vs 164 here,
  and E0 v1's halt at −1.33%. This affects training, not decode stability.

## 5. Options (none applied)
1. **Unchanged protocol on a quieter host.** Screen the host for 60 s first (CPU-only control and
   governor; no GPU spend), then run the predeclared validation. This keeps comparability with the Codex
   arm, whose host was quiet.
2. **Disclosed operational amendment: pin the inference evaluator to one CPU core.** Windows, thresholds
   and the ratio definition stay unchanged. This needs a change to the frozen launch path, validation that
   baseline-vs-baseline ratios stay at about 1.0, and disclosure that Opus inference ran under a stricter
   OS environment than Codex.
3. **Not recommended.**
   - Resume here unchanged: about two thirds of attempts would be denied Pareto eligibility, against
     2% for Codex.
   - Retry unstable measurements (selection on noise).
   - Loosen the 2% threshold.

## 6. Host screen for the next pod (PI decision 2026-11-05: try one quieter host, protocol unchanged)
- `scripts/pod/host_cpu_screen.py` takes 60 s and uses no GPU or torch. It records the CPU model,
  governor, quota, load and CPU pressure, then times fixed pure-Python work in 1-s chunks.
- **Screen rule, set before use:** pass if the CPU-only rate's IQR/median is ≤ 0.02 over the 60 chunks
  **and** at most half of the rolling 8-chunk sets exceed 0.02. Otherwise, report before any GPU time.
  The screen is advisory; the validation acceptance rule (doc 17 §8) is unchanged.
- **Reference result on this pod** (00:55 UTC, load 13.7): IQR/median 0.172, 75% of rolling sets
  fail, so `report_before_gpu`. The screen flags this host.
- If the next host also fails the validation, replacement stops and the options in §5 go back to the PI.

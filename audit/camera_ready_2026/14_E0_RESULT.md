# E0 result (2026-11-04): v1 HALT on run 2's update window; amendment-1 rerun PASS

Plan `evaluation/camera_ready_e0_e1_plan.json`, sha256 `806a63970258627092d929d4602e54e36a2412e0656a63298e17307c6c829340`, unchanged.
Pod: fresh RunPod A40 `GPU-4702993a-fd40-d985-292e-6398bd236183`, driver 595.91.07.
Launched 04:43:58 UTC with `HF_HUB_OFFLINE=1`, after validate-only (FAIL 0, POD_ONLY 0). The stage finished at 05:26:10 UTC.
Evidence: `results/camera_ready_2026/E0/` (copied back; 33 files hash-identical to the pod; checkpoints stay on the pod).

## Verdict: `HALT`
The single halt reason is `e0-arref-time300s-seed0: updates 666 outside [669, 681]`. The executor refuses E1.

| Gate | Result |
|---|---|
| Data identity | PASS: stream `180e0a5b…` for seeds 3 and 6 over 1,536 microbatches; 13/13 files |
| Run 1 AP-Q, 201,326,592 tokens, seed 3 | All integrity gates pass. Exactly 201,326,592 tokens in 1,536 updates. Held-out 1.1398974 against 1.1396158 (Δ +0.00028). Validation 1.0770575 against 1.0765888 (Δ +0.00047). |
| Run 2 AR-ref, 300 s, seed 0 | All integrity gates pass. In-process validation 1.1041533 against 1.1029011 (Δ +0.00125, tolerance 0.005). **666 updates against 675 historical; window [669, 681]: HALT.** |

## Diagnosis (read-only; the rule says to diagnose drift first)
- **Data, software and kernel are identical to the panel's.**
  - Data: the data gate passed.
  - Python 3.10.18, torch 2.9.1+cu128 from `uv.lock` `262a4d55…`.
  - flash-attn3 resolved offline to `e29f138f`, as recorded in the stage manifest.
- **Throughput is the only difference.**
  - AR-ref steady throughput: 285,982 tok/s, against 289,842 historical (−1.33%). 675 × (1 − 0.0133) ≈ 666, so the deficit accounts for the 9 missing updates.
  - AP-Q steady throughput: 517,572 tok/s, against the panel's 522,708 (−0.98%).
  - Peak training memory is identical to 6 significant figures.
- **The slowdown is uniform, not stalls.**
  - AP-Q step time, steps 21–1536: median 252.8 ms against the panel's 250.6 ms (+0.9%); p10 +1.0%; p90 +0.9%.
  - CPU steal time is 0. The slowdown is in this GPU/driver combination, not CPU contention or I/O.
- **Historical spread.** AR-ref at 300 s gave 675, 675 and 675 updates (seeds 0–2, July pod) and 678 (seed 42, discovery), at 289.8–291.1k tok/s. The September panel pod ran AR-ref at about 289k tok/s. 666 lies outside the spread of both earlier A40 hardware epochs.
- **Remaining differences.** The physical A40 differs. The driver is 595.91.07 (panel: 570.211.01). The power limit is 300 W at the default; no clock or thermal throttling is active, though SW power capping has accumulated, as is usual for A40s under load. None of these can be changed from inside the pod.

## Observations
- The plan's own rule records throughput and memory "as diagnostics only". Run 2's update window is the one halting criterion that measures throughput rather than quality.
- The quality reproduction passes with wide margin: all three BPB deltas are ≤ 0.0013, against a 0.005 tolerance.
- **Not decided.** Whether to amend the rule or replace the hardware is the PI's decision. Any amendment would need a new plan file and hash, with this HALT verdict kept as evidence.

## Side finding
`/workspace` (MooseFS over FUSE) ignores `chmod`: directories stay 0777 and files 0666.
- Candidates cannot reach `/workspace`. It is outside the deny-by-default Landlock allowlist in `study/contracts_2026/candidate_sandbox.py`.
- The `/workspace/mirror` copy of `/var/lib/autopareto` (made 05:29 UTC; its 2 checkpoints are hash-identical) is a backup, not a trust boundary.

## Amendment 1 (PI decision, 2026-11-04)
The PI chose to keep this A40 and amend only E0's throughput acceptance rule, on two conditions: the failed result is preserved, and the amendment is documented openly.
- **New plan:** `evaluation/camera_ready_e0_e1_plan_amendment1.json`, sha256 `89f2901588fceec18ab93d4e74ed652b8f6c6df9329eb8ba1d3a7ead128f60dc`. The original plan file is byte-identical, `806a6397…`.
- **What changed:** run 2's update count is recorded as a diagnostic instead of halting. Everything else is unchanged: BPB, token-integrity and reproducibility checks, E1 and the analysis.
- **Rerun:** E0 runs again in full under the new hash, into `results/camera_ready_2026/amendment1/E0/`. The verdict above is kept and reported alongside it.
- **Offline check before launch, on the real records above:**
  - the original plan reproduces this HALT;
  - the amended plan passes them;
  - every BPB, token-count and failed-twice condition still halts under the amended plan.

## Amendment 1 rerun (2026-11-04): `PASS`
- **Plan and timing.** Plan sha256 `89f29015…`. Validate-only first: 18 ok, FAIL 0, POD_ONLY 0. Launched at 07:29:22 UTC on the same pod. The stage finished at 08:06:28 UTC.
- **Verdict.** `PASS`, with no halt reasons.
- **Evidence.** `results/camera_ready_2026/amendment1/` holds 33 stage files and 3 console/launch logs, all hash-identical to the pod. The checkpoints stay on the pod.
- **Earlier verdict.** The original HALT verdict above is unchanged (`bd953480…`).

| Gate | Result |
|---|---|
| Data identity | PASS |
| Run 1 AP-Q, 201,326,592 tokens, seed 3 | Exactly 201,326,592 tokens in 1,536 updates. Held-out 1.1391452 against 1.1396158 (Δ −0.00047). Validation 1.0766111 against 1.0765888 (Δ +0.00002). |
| Run 2 AR-ref, 300 s, seed 0 | In-process validation 1.1042673 against 1.1029011 (Δ +0.00137, tolerance 0.005) |

Diagnostics (not halting):
- **AP-Q throughput.** 521,827 tok/s, against 522,708 (−0.17%). Peak memory is identical.
- **AR-ref throughput.** 286,971 tok/s, against 289,842 (−0.99%). Peak memory is identical to 6 significant figures.
- **AR-ref update count.** 668 updates against 675 (−1.04%).
  - 668 is still outside the v1 window [669, 681], so **the original plan would have halted this rerun as well**.
  - The pass therefore depends on amendment 1, and is reported as such.
- **Reproducibility across the two runs on this pod.** Every BPB moved by ≤ 0.0009 between the halted run and the rerun.

Operational note: my completion watcher self-matched its `pgrep` pattern and never exited. The pod sat idle from 08:06 to 09:38 UTC, until the watcher was killed by hand. This had no effect on the evidence.

## Effect of the GPU conditions on comparisons with historical Codex results
- **Fixed-token (N) contract.** BPB at a fixed token budget does not depend on throughput. E0 run 1 reproduces the panel to within 0.0005 BPB. Hardware speed cannot change N-arm quality comparisons.
- **Fixed-time (T) contract and speed objectives.** These carry the throughput offset.
  - **Env A** (E0/E1) is 0.2–1.3% slower here.
  - **Env B** (Pair-A) is −0.42% here: 282,520 against 283,707 tok/s. At G2, T ran 162 updates against 163 (T BPB +0.0012; N BPB −0.0002).
  - The Codex G4 best T candidate improves on the baseline by about 0.098 BPB, roughly 80× the hardware effect on the T baseline.
- **Historical hardware already varied.** The Codex G4 T arm itself ran on two A40s (`GPU-98579d60…` for attempts 1–9, `GPU-35264a0f…` for attempts 10–22). A cross-pod hardware offset of this size was already part of the historical comparison.
- **Fair claims.** The confirmatory comparisons are within-pod and paired: the Opus winner against the re-registered baseline on seeds 6–8, E1, and E1X. A cross-campaign comparison, Opus against Codex search trajectories, is descriptive only. A same-pod comparison needs the Codex winner rerun here.
  - The frozen confirmation pre-registration (`adf8ce71…`) already lists this as optional and approval-gated: the Codex G4 T-arm best-quality candidate (attempt 14, BPB 1.1016) on the same 12-run design.

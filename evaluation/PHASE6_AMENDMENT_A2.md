# Phase 6 amendment A2: one extra attempt of P1 MOTPE seed 8

Approved by the author on 2026-11-16 at about 12:40 UTC. The author chose "Amendment A2 rerun
(Recommended)":

> Run MOTPE seed 8 once more (attempt 3), with the same code and gates, on local disk under the A1
> storage rules. Keep both failed attempts on record. Run its downstream and P0b inference with the
> frozen functions, then resume the remaining chain stages unchanged.

- **Written before** attempt 3 ran.
- **Parent plan:** `evaluation/phase6_unified_plan.json`, sha256 dc1e605e..., left unchanged.
- **A2 plan:** `evaluation/phase6_amendment_a2_plan.json`. It records:
  - the parent and A1 plan hashes;
  - the hashes of this file and of the A2 code;
  - the hashes of both failed attempt records.

## 1. Why

- **What failed.** Both P1 attempts of MOTPE seed 8 stopped before their first training step, at
  12:17 and 12:18 UTC (`PHASE6_INCIDENTS.md`, incident 3).
- **Cause.** Triton's compile workers could not create directories in the run directory on the
  network volume (`OSError: [Errno 116] Stale file handle`).
- **Why it is not the recipe.** MOTPE seeds 6 and 7 passed with the same code and environment.
- **What the frozen plan allows.** One retry per run. So without A2, MOTPE would have only 2 of
  the 3 matched seeds.

## 2. What A2 does

| Item | Under A2 |
|---|---|
| Training | Attempt 3 of `p6-motpe-tokens201326592-seed8` through the parent runner's `execute_train_attempt`, which calls E1's token executor and token gates unchanged. Same code, source hash, seed, token budget and launcher. The attempt is marked as retried, as attempt 2 was. |
| Storage | A1's rules (`PHASE6_AMENDMENT_A1.md`, section 1): Env A on local disk, with a freeze identical to the volume environment's; data on local disk, verified against the E1 manifest; the attempt runs in `/root/phase6_local_a2`. It is then copied to `results/camera_ready_2026/phase6/runs/p6-motpe-tokens201326592-seed8-attempt3`, and every file's SHA-256 is compared. |
| Run record | Same fields as the other P1 records. It adds an `amendment` block (A2, the A2 plan hash, A1 storage), `executed_in` and the copy manifest hash. |
| Timing | A1's timing-anomaly measure is computed and reported. It is not a gate, because P1 matched-token runs have no timing gate. |
| Frozen downstream | The parent runner's `execute_downstream`, for this one run only. Its results go into the same database and directory as the other P1 checkpoints. |
| P0b inference | The parent runner's `execute_inference`, for the label `motpe-seed8-native` only. It runs in Env B, as all P0 runs do. |
| lm-eval (P2) | `run_phase6_p2.py` takes this checkpoint when the frozen lookup finds none, but only if its record passed and names the current A2 plan hash. This changes `run_phase6_p2.py`, so the A1 plan was regenerated before any P2 run. Nothing else in A1 changed. |

- **What is redirected.** For downstream and inference, the parent functions are called unchanged.
  Only the lookup that finds this one run's record or checkpoint is pointed at attempt 3, and only
  for the duration of that call.
- **What is kept.** Both failed attempts stay on record, and nothing is deleted.
- **If attempt 3 fails.** It is reported and not repeated.

## 3. Order of execution

1. **The r2 chain stops after P1 training.** The stage exits 1 because this run is reported failed.
   The resume script `camera_ready_phase6_chain_r3.sh` continues only if this run is the only one
   reported failed.
2. **The remaining frozen stages run under the parent plan, as before.**
   - P1 downstream: it skips this run, since it has no passing attempt 1 or 2.
   - P0b: the same command, given every P0b label except `motpe-seed8-native` with the runner's
     existing `--labels` option.
   - Controls and anchors.
   - Nothing about storage has changed at this point.
3. **A1 storage is switched on** exactly as A1 describes: the data is copied and verified, and the
   symlink is re-pointed.
4. **A2 runs:** training, then downstream, then inference.
5. **The A1 pilot runs**, and work stops for review before P2.

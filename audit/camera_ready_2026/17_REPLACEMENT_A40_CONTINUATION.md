# 17 — Replacement A40 (2026-11-04): pod rebuilt; continuation prepared, not applied

Status: **campaign paused at attempt 2 (global position 4); the live campaign state is untouched.**
The PI approved validation and resume. Validation passed its BPB halt rule, but the pod failed the
decode-stability check, so the cache fix, the continuation and the resume were **not** applied (§8).

## 1. What changed
- The PI stopped the first camera-ready pod while the campaign was paused (doc 16), then moved to a
  new pod. RunPod wipes the container disk on stop; the `/workspace` volume carried over.
- New pod: NVIDIA A40 **`GPU-9f05dd40-e39c-d188-0e4d-3835f1b9e0c2`**, driver 595.91.07 (unchanged),
  kernel 6.8.0-147, same image and volume. The proxy and TCP endpoints are not recorded here.
- **Survived** (`/workspace`): the asset bundle, the Env A venv and persistent data cache, the E0
  evidence, and the 05:29 UTC mirror of `/var/lib/autopareto` (the G2 baseline state).
- **Lost with the old container disk:** the campaign's pod-side state
  (`/var/lib/autopareto/pair_a_claude_opus55_t_seed3`). That covers attempt 1's durable receipt, logs and
  checkpoint, and attempt 2's payload and FAILED receipt. The Mac keeps attempt 1's result, both candidate sources and the
  FAILED receipt (inside `controller_state.json`). Confirmation retrains from source, so the lost
  attempt-1 checkpoint is provenance only.

## 2. Pod rebuild (no GPU work)
Run from the runbook (`scripts/pod/fresh_a40_bootstrap.sh`) with one deviation: state already on
`/workspace` is verified, not rebuilt. Log: `/workspace/pod_setup_logs/migration_20261104T225657Z.out`.

| Check | Result |
|---|---|
| Base image, driver, disks | `check` stage passed |
| Asset bundle | 87 files verified; the persistent Env A cache matches it byte for byte (13 files) |
| Env A | interpreter restored (uv 0.9.0, Python 3.10.18); `uv sync --frozen --inexact` audited 74 packages; all 10 pins match |
| Env B | rebuilt; all pins match (torch 2.8.0+cu128, triton 3.4.0, kernels 0.12.1, huggingface-hub 1.30.0) |
| Trusted tree | 122 files, identical to the Mac worktree (combined hash `6692803d…`) |
| Registered cache and candidate inputs | rebuilt from the bundle, 45 entries verified, root-owned, read-only, no ACLs |
| Paired-baseline checkpoints | restored from the mirror; T `f7ca17c6…` and N `72d05e97…` verified |
| Linux sandbox tests | 176 run, OK (4 skipped) |
| Tokenizer | retrained natively; both registered hashes reproduce |
| E0/E1 validate-only (amendment-1 plan `89f29015…`) | ok 18, FAIL 0, POD_ONLY 0 |

The bootstrap's own validate-only step still passes the v1 plan hash and is refused by design. Its
log on the pod was overwritten; the Nov 4 copy is preserved on the Mac in
`runs/fresh_a40_setup_20261104/`.

## 3. Precedent: the Codex arm's cross-A40 continuation (2026-09-16)
At global position 20, the Codex arm moved from `GPU-98579d60…` to `GPU-35264a0f…`:
- It reused the predecessor calibration: N\* 85,458,944 was kept, not re-measured.
- It validated the new A40 with paired T and N baseline smoke runs.
- `tools/prepare_campaign_continuation.py` copied the completed prefix into a new state root, wrote
  a continuation manifest, and re-armed the abandoned, unconsumed next invocation with its exact
  candidate. The abandoned invocation was recorded under `migration/`.

The same path fits this move: position 4 holds a durable execution record for candidate `ab63dbad…`
that never ran.

## 4. Single-arm fix to the continuation module
`continuation.py` predates single-arm studies. It required every prefix position to be completed,
every later position to be pending, and a state directory for both campaigns. The fix skips
`not_in_study` positions and the inactive campaign only when the state records `active_contracts`;
two-arm states are checked as before.
- Tests: 2 new (single-arm prepare; inactive positions rejected without the declaration). All 15
  continuation and requeue tests pass. The new single-arm test fails on the unpatched module.
- `continuation.py` is in the immutable deployment, so the continuation manifest records its new hash
  (`a3046502…` → `efdee709…`). It is the only deployment change. It is control-plane code that runs on
  the Mac; it has been deployed to the pod's trusted tree so the 122-file gate matches.

## 5. Dry run (scratch copy; the live state is unchanged, sha `c821de3a…` before and after)
- Position 4 is `running` / `prepared` with the **same candidate** `ab63dbad…`, a new invocation
  `pair-a-004-4b7202a1…` and the abandoned `pair-a-004-ccb492e3…` recorded in `migration/`, with its
  FAILED receipt.
- Counts: 1 completed, 1 running, 23 pending, 25 not in study.
- The controller's `_verify_final_manifest` and `_initialize_state` accept it.
- Scientist configuration, contracts and inference provenance are unchanged from the frozen manifest.
- Read-only launch gates over the controller's proxy transport: manifest, transport and state config,
  122 trusted files, GPU `GPU-9f05dd40…`, registered cache and both checkpoints all pass. Only the
  fresh-state-root check fails, as expected for a continuation.

## 6. Cache-permission fix, tested on a scratch copy
- An online kernel load into a copy of the registered cache reproduced the incident exactly:
  `trees/e29f138f….json` is created 0600, along with `.agent_harnesses.json` 0644, `__pycache__` 0755/0644
  and `refs/main` still `e29f138f`.
- After `chmod 0444`, a second online load leaves the file unchanged and the sandbox's cache rule
  finds no violation. The copy was then deleted.

## 7. Proposed steps (need approval)
1. **New-A40 validation (~20 min GPU).** Run `scripts/pod/g2_reregister_pair_a_baseline.sh` unchanged,
   with `G2_STATE=/var/lib/autopareto/pair_a_claude_opus55_a40_validation_20261104`. It applies the same
   halt rule: stop if the T smoke BPB differs from 1.19989 by more than 0.005.
2. **Cache fix.** Run one online kernel load into the real registered cache, as the trusted stage
   would, then `chmod 0444` the tree file. Re-run the cache rule and the 15 absolute-file hashes.
3. **Continuation.** Build it with the reused calibration record and the validation results, a new
   local state root and a new remote state root. Freeze the manifest read-only, then rerun the
   read-only gates.
   - **Differs from 2026-09-16:** keep the existing paired checkpoints `f7ca17c6…`/`72d05e97…`. Every
     attempt re-measures the paired baseline next to the candidate on the current GPU (bracketed
     inference), so all 25 attempts can share one reference. The smoke runs serve as hardware
     validation only.
4. **Resume** under the supervisor, with a new supervisor directory.
5. **Mirror** `/var/lib/autopareto` to `/workspace` every 15 minutes during the campaign.

**Disclosure for the paper:** the Claude arm spans two A40s. Attempt 1 ran on `GPU-4702993a…`;
attempts 2–25 run on `GPU-9f05dd40…`.

## 8. Validation result (2026-11-04 23:35 – 2026-11-05 00:09 UTC): BPB passes, decode timing too noisy
The committed script `g2_reregister_pair_a_baseline.sh` (sha `15975b14…`) ran unchanged, with state in
`/var/lib/autopareto/pair_a_claude_opus55_a40_validation_20261104`. Evidence is mirrored to `/workspace`, and a
copy is on the Mac in `runs/replacement_a40_validation_20261104/` (`SHA256SUMS`; matches the pod).

| Run | BPB | Tokens / updates | Inference |
|---|---|---|---|
| T smoke (300 s) | **1.19913** (\|Δ\| 0.00075 vs 1.19989: **halt rule passes**) | 85,983,232 / 164 in 301.6 s | unstable: drift 0.0202 > 0.02 |
| N smoke (N\*) | 1.19988 | 85,458,944 / 163 | unstable: drift 0.0506 > 0.02 |
| Old pod G2 T, for reference | 1.20110 | 84,934,656 / 162 | stable |

Training throughput and quality on this A40 match the registered baseline. The decode measurement does
not. Because a campaign attempt whose bracketed measurement is unstable cannot enter the Pareto
archive (`campaign.candidate_disposition`), I repeated the campaign's exact bracketed measurement
three times. Each run was inference only: the T validation checkpoint as candidate, bracketed by the
paired checkpoint `f7ca17c6…`.

| Bracketed repeat | Normalized drift (limit 0.02) | Status | Candidate tok/s |
|---|---|---|---|
| 1 | 0.0148 | stable | 134.1 |
| 2 | 0.0309 | **unstable** | 145.8 |
| 3 | 0.0418 | **unstable** | 148.7 |
| Codex comparator A40 (`GPU-35264a0f…`), 42 attempts | median 0.0063, p90 0.0152, max 0.0175 | 1 of 42 unstable | |
| Claude attempt 1 (old pod) | 0.0134 | stable | |

- The noise comes from the baseline segments: their absolute range drift is 13–18%.
- The host is shared and busy. The container's CPU quota is 7.65 of 96 CPUs, host load is 9–13,
  and CPU pressure is "some" about 9–10% (`/proc/pressure/cpu`). The GPU itself showed no thermal
  or power slowdown during decode.

**Decision.** Do not resume on this pod. At about 2 in 3 unstable, most remaining Opus attempts would
be excluded from the Pareto archive, while the Codex comparator lost only 1 in 42. That would bias
the cross-agent comparison.

**Not done:** the registered-cache fix, the continuation, the resume and the mirror loop. The live
campaign state and the frozen manifest are unchanged. The pod's trusted tree carries the patched
`continuation.py`, which only the continuation manifest uses.

**Proposed acceptance rule for the next pod, set before any result:** both smokes must pass the BPB
halt rule, **and** all three bracketed inference repeats must be stable (normalized drift ≤ 0.02).
On a host like the comparator's, all three pass with probability of about 0.93 ((41/42)³).

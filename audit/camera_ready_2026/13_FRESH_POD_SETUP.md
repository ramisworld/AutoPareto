# 13 — Fresh RunPod A40: assets, configuration and setup (2026-11-02)

The previous pod and its `/workspace` snapshot no longer exist. Everything below assumes a new,
empty A40 pod. This supersedes step R ("restore from `/workspace/PRESERVED_…`") in
`12_GPU_READINESS_REPORT.md`, and the "back up `/workspace/autopareto`" step in
`evaluation/CAMERA_READY_E0_E1.md`. Both are runbook deviations only: no plan or protocol hash
changes.

Full asset audit: `supporting/ASSET_AUDIT_20261102.md`.

## 1. What is recoverable

| Asset | Status | Evidence |
|---|---|---|
| 12 data shards (10 train, validation 06542, held-out 06541) | Downloadable, verified | HF serves the recorded sha256 and size at revision `915333b4…` (= current `main`). Preserved locally by the asset step (§7). |
| Tokenizer (`tokenizer.pkl`, `token_bytes.pt`) | Regenerable | Not stored anywhere. Two fresh pods in September retrained it to identical hashes, and local tests show the training is deterministic. Local reproduction: §7. |
| Token-stream hash `180e0a5b…` | Regenerable | Record present; replayable on CPU (§7). |
| flash-attn3 kernel `e29f138f` (torch 2.8 and 2.9 builds) | Downloadable, **at risk** | The repo's latest commit is a removal notice; copies are preserved locally (§7). |
| `uv.lock` `262a4d55…` (Env A) | Present | In git on `autopareto/research-infrastructure`. |
| Env B package pins | Present | Recovered from the September provisioning logs (no lockfile). |
| 122 Pair-A trusted files | Present | All in the worktree; the 6 amended files differ by design. |
| Baseline registration record `8ce36b3c…` | Present | `runs/pair-a-official-20260914-new-a40/provenance/` |
| 21 Phase-1 panel `final.pt` | Present | `.git/lfs`, hash-verified |
| E1 recipe sources | Present | `b3a961ee…`, `55a7fd8b…` |
| Downstream datasets (4) | Downloadable | Pinned revisions still served; `datasets==5.0.1` is not in `uv.lock` (disclosed). |
| **Paired baseline checkpoints** T `0ccea0c4…`, N `23be533b…` | **Lost** | Lived only on the pod. Must be re-registered on the new A40 (§5, needs approval). |
| `calibration.json` `211d0b74…` | Lost (values survive) | N\* = 85,458,944 is a frozen constant; the controller never reads the file. |
| Pod-side Codex state (receipts, logs, candidate checkpoints, `RESUME.md`) | Lost | Codex sources and histories survive locally. Provenance-only loss. |
| Codex Docker image `877dae34…` | Lost | Not needed by the Claude backend. |
| Phase-1 harness versions that ran the panel (audit M13) | Lost earlier | E0 run 1 is the pre-registered check. |

**Disclosure:** the final Pair-A manifest's `runtime_cache_closure` is a stale 2026-09-07 record
(torch 2.9.1, torch29 kernel build). The September pods actually ran torch 2.8.0 with the torch28
build. The controller does not gate kernel identity; the setup pins the torch28 build for Env B.

## 2. RunPod configuration (please create exactly this)

| Setting | Value |
|---|---|
| Cloud | Secure Cloud, **On-Demand** (not Spot/interruptible: a 25-attempt campaign must not be preempted) |
| GPU | 1 × **NVIDIA A40** (48 GB) |
| Container image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (Docker Hub digest `sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`) — Python 3.12, torch 2.8.0+cu128, Ubuntu 24.04 |
| Container disk | **≥ 60 GB** (Env B, `/opt` and `/var` trees; wiped if the pod stops) |
| Volume disk (`/workspace`) | **≥ 150 GB** (Env A, assets, E0/E1/E1X run directories and checkpoints, state mirror) |
| Ports | Enable **SSH over exposed TCP** (port 22). No HTTP ports are needed. |
| SSH key | The account's public key must be `~/.ssh/id_ed25519.pub`, fingerprint `SHA256:xIDUysG5ZSB/wN5vJ8qpndnolknW5wQZPdnGui2kFLs` |
| Environment variables | None. Do not set `PYTORCH_CUDA_ALLOC_CONF`. |
| Lifetime | Keep running from setup to the end of the campaign. Stopping wipes the container disk. |

**After creating it, send me:**
1. the proxy SSH target, which looks like `<pod-id>-<suffix>@ssh.runpod.io`;
2. the exposed-TCP line, which looks like `ssh root@<IP> -p <PORT>`.

Neither goes into a committed file.

## 3. Mac → pod transfer (over the exposed TCP endpoint)
```bash
P="-p <PORT> -i ~/.ssh/id_ed25519"; H=root@<IP>
cd /Users/Rami/Documents/dev/autopareto-camera-ready
rsync -a -e "ssh $P" --exclude .git --exclude .venv --exclude 'results/camera_ready_2026/' ./ $H:/workspace/autopareto-camera-ready/
git -C /Users/Rami/Documents/dev/autopareto show autopareto/research-infrastructure:uv.lock | ssh $P $H 'cat > /workspace/autopareto-camera-ready/uv.lock'
rsync -a -e "ssh $P" /Users/Rami/Documents/dev/autopareto_assets_20261102/ $H:/workspace/assets/
ssh $P $H 'install -d /workspace/g1_check' && rsync -a -e "ssh $P" /Users/Rami/Documents/dev/autopareto/.git/lfs/objects/9d/be/9dbea197499a5be43814e1eeb6c649f2cb0591189f8d7de1af6d80cc9e65f8f7 $H:/workspace/g1_check/final.pt
```
The G1 checkpoint is the LFS object for `results/phase1/runs/phase1-autopareto_292-seed3/final.pt` on
`autopareto/research-infrastructure` (sha256 `9dbea197…`, 92,286,569 B). It is not checked out in the
main tree; the September snapshot copy has the same hash.

## 4. Pod setup: no training, no GPU use beyond queries
```bash
cd /workspace/autopareto-camera-ready && bash scripts/pod/fresh_a40_bootstrap.sh all
```
Stages, each runnable alone:
* `check`: A40, CUDA ≥ 12.8 driver, x86_64, `setpriv`, base image Python and torch, free disk.
* `assets`: `sha256sum -c` on the asset folder, then places the Phase-1 caches.
* `env-a`: uv 0.9.0, Python 3.10.18, `uv sync --frozen` from `uv.lock`, pins checked.
* `env-b`: `/opt/autopareto_runtime` with the recorded pins and torch 2.8.0.
* `pair-a`: the candidate user (UID/GID 2001), root-only state, the 122 trusted files, the registered
  cache and read-only candidate inputs.
* `verify`: runs the Linux-only sandbox tests that are skipped on the Mac, plus E0/E1 validate-only
  (it must report FAIL 0 and POD_ONLY 0), and writes a readiness receipt with GPU UUID, kernel,
  versions and the amended-file hashes.

**Environment verified** means the receipt shows:
* all pins matched;
* the sandbox tests passed (the Codex Docker test is expected to error: there is no Docker on the pod);
* validate-only gave FAIL 0 and POD_ONLY 0.

The script has not been run on Linux yet; the receipt is its test.

## 5. GPU steps, each needing your approval, in order
1. **G1 — evaluator check (~5 min).** Re-score the LFS checkpoint `phase1-autopareto_292-seed3/final.pt`
   with the held-out evaluator in Env A. Expected 1.139616; any difference above 1e-4 halts before E0.
2. **G2 — Pair-A baseline re-registration (disclosed amendment, ~25 min).** On the new A40, run the
   registered baseline (`b2fb8788…`, seed 3) through `run_t_smoke.py` (300 s) and `run_n_smoke.py`
   (N\* = 85,458,944), with `AUTOPARETO_TRUSTED_STATE_ROOT=/var/lib/autopareto/pair_a_claude_opus55_baseline`.
   * The new checkpoints replace the lost T/N paired checkpoints, via
     `derive_claude_campaign_manifest.py --a40-paired-baseline-t/-n /path:sha256`.
   * `run` hash-verifies them before any position (commit 1bfd868).
   * **Halt rule (proposed):** if the T smoke validation BPB differs from the registered 1.19989 by
     more than 0.005 (E0's tolerance), stop and report.
   * The campaign keeps the registered baseline value for its Pareto decisions, as the Codex arm did
     across its two A40s. This is a third hardware epoch, to be disclosed.
3. **E0** (pre-registered; halts on its own rule).
4. Then derive and freeze the campaign manifest (new GPU UUID, SSH host, checkpoints) and launch the
   campaign as in `12_GPU_READINESS_REPORT.md` §6. Codex selection is unchanged.

Between stages, `bash scripts/pod/fresh_a40_bootstrap.sh mirror` copies durable pod state to
`/workspace`. Each stage's results are pulled back to the Mac and added to the external backup
(`BACKUP_RECOMMENDATION.md`).

### Operational settings for E0/E1/E1X (no plan-hash change)
* Prefix every executor command with `HF_HUB_OFFLINE=1`. `get_kernel` is unpinned, so online it would
  follow the kernel repo's `main`, which carries a removal notice. Offline, it resolves the preserved
  `e29f138f` build; the panel runs resolved the same revision online. The Pair-A candidate sandbox
  already sets this variable.
* Keep `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_ALLOC_CONF` unset; the fixed-time cell sets the
  historical allocator itself.

## 6. Cost (A40 Secure on-demand, about $0.44/h; confirm in the console)
| Stage | A40-h |
|---|---|
| Setup and verify | ~1.0 (GPU idle but billed) |
| G1 + G2 | ~0.5 |
| E0 | 0.73 |
| Claude campaign | ~7–8 |
| Opus confirmation | ~2.7 |
| E1 | ~6.4 |
| E1 extension | ~4.7 |
| **Total** | **≈ 23–24 A40-h ≈ $10–11**, plus volume storage of about $0.07/GB/month (≈ $10/month for 150 GB) |

## 7. Local pre-verification (no GPU)
`/Users/Rami/Documents/dev/autopareto_assets_20261102/` is read-only, 2.6 GB, with 87 files checked
by `SHA256SUMS`. It contains:
* the 12 climbmix shards from revision `915333b4`, each matching the manifest and HF hashes;
* the flash-attn3 torch28 and torch29 cu128 builds at `e29f138f`, laid out as an HF hub cache that
  resolves offline;
* the tokenizer;
* `install_assets.sh`, with `env-a` and `pair-a` modes.

Results:
* `tokenizer.pkl` reproduces byte for byte from the pinned `prepare.py`.
* `token_bytes.pt` equals the registered `23517e05…` after the macOS serialization ID is replaced with
  the Linux one. Every other byte, including the tensor values, matches. The bootstrap's `verify` stage
  retrains it natively on the pod and requires both registered hashes.
* The executor's own `data_identity_gate()` passes on this Mac's CPU: token stream `180e0a5b…` for
  seeds 3 and 6 over 1,536 microbatches, and all 13 file hashes match.

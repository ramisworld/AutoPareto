# AutoPareto fresh-A40 asset recoverability audit (2026-11-02)

Read-only audit. Nothing in the main checkout, the worktree or the backup clone was changed. No GPU, SSH, model call, install or large download was made.

**How each claim was established**
- **V (verified):** I computed the sha256 myself, read the header in an HF HEAD response, or read the HF API metadata on 2026-11-02.
- **R (recorded):** I read it in a local record: a manifest, a JSON file, or the transcript of an earlier agent session that captured pod output. I did not recompute it.
- **I (inferred):** my own reasoning.

**Local search.**
- Roots: `~/Documents` (main checkout with `.git/lfs`, worktree, backup clone), `~/.cache` and `~/Downloads`. I skipped `.venv` and `node_modules`.
- Pass 1 hashed every file whose size matched a recorded size, plus every file matching an asset name pattern (`*.pt`, `*.parquet`, `tokenizer.pkl`, `calibration*.json`, `uv.lock`, `*.abi3.so`, and so on): 617 files out of 360,296 walked.
- Pass 2 hashed every file under 50 MB in both checkouts, `~/.cache` and `~/Downloads`: 179,279 files.
- I also listed asset filenames in every git ref, including the Codex turn-diff tree.
- Scratch outputs are in this folder: `local_search_result.json`, `hf_climbmix_heads.txt`, `fa3_heads.txt`, `tokenizer_retrain_evidence.txt`, `provisioning_commands.txt`, `preservation_transcript_tool_outputs.txt`, and `determinism/`.

## (a) Asset table

Consumers: **E0** (environment smoke), **E1** (crossover), **CC** (Claude Opus 5.5 single-arm T campaign on the Pair-A harness), **OC** (Opus winner confirmation), **E1X** (E1 extension with the Opus winner).

### Data and tokenizer

| Asset | Recorded hash / version | Local copy? | How to recover | Status | Consequence |
|---|---|---|---|---|---|
| `data/shard_00000…00009.parquet` (10 training shards) | `data_manifest.json` (sha `6a6d9c60…`, aggregate `12118270…`): `054ddbd9…`/92,291,918; `d4cfe1de…`/92,124,317; `5e80d759…`/91,803,619; `b349c85d…`/91,793,972; `87dae96f…`/91,428,652; `3d9aebf1…`/91,845,922; `6a3f2afd…`/91,455,616; `731eea33…`/91,202,032; `e7837913…`/91,155,507; `f22410fe…`/92,072,356 | None (V) | Download from `karpathy/climbmix-400b-shuffle`. HF `x-linked-etag` (sha256) and `x-linked-size` equal the recorded values for all 10, at `main` and at `915333b4…` (V). `main` currently resolves to `915333b4…` (`x-repo-commit`, V). The repo's last commit is `915333b4` (2026-03-03, V). | **downloadable-verified** | All five consumers need them. E0's data gate and the Pair-A controller's launch gate both check these hashes. `prepare.py` downloads from unpinned `main`, so download from `resolve/915333b4…` and check sha256 before first use. |
| `data/shard_06542.parquet` (validation) | `769fe59d…` / 91,699,537 | None (V) | Same check, both revisions (V) | **downloadable-verified** | All five consumers. |
| `shard_06541.parquet` (held-out) | `51da1a9b…` (E0/E1 plan; panel downstream `results.json` R). Size 91,897,402 (V header). | None (V) | `downstream_evaluator.download_heldout_shard()` fetches it from `resolve/915333b4…`. Header matches at that revision and at `main` (V). | **downloadable-verified** | E0, E1, OC, E1X (held-out BPB) |
| `tokenizer/tokenizer.pkl` | `888c9fa8…` / 94,590 | None anywhere (V). No public host: `prepare.py` trains it locally. | Regenerate with `prepare.py` `train_tokenizer()` (rustbpe, vocab 8,192, the first 1e9 chars of the sorted training shards). **Precedent (R):** on two fresh pods `prepare.py` downloaded the 11 shards from `main`, trained the tokenizer, and `sha256sum` showed exactly `888c9fa8…` / `23517e05…`. Runs: 2026-09-14T09:25Z, pod <pod-id>, trained in 40.5 s; 2026-09-15T23:02Z, pod riedurg43ydin5, after `rm -rf /workspace/recovery_home`, trained in 41.9 s. Source: `~/.codex/sessions/2026/09/{14,16}/rollout-*01a064f6*.jsonl`, excerpt in `tokenizer_retrain_evidence.txt`. **Local test (V):** rustbpe 0.1.0 produced byte-identical pickles across runs and across RAYON thread counts 1, 3 and default (synthetic corpus). | **regenerable** (hash check required) | All five consumers need it. Verify the sha before E0 or CC. If it differed, E0's data gate and the controller launch gate would both refuse, and a new data registration would break comparability (see b). |
| `tokenizer/token_bytes.pt` | `23517e05…` / 34,373 | None (V) | Written by `train_tokenizer()`, reproduced in the same two runs (R). **Local test (V):** with torch 2.9.1, `torch.save` is byte-deterministic: zip timestamps are fixed at 1980 and the serialization_id is deterministic. An 8,192-element int32 tensor saved as `token_bytes.pt` is exactly 34,373 B, so the container format matches. The archive name comes from the filename, so it must be saved as `token_bytes.pt`. | **regenerable** | Same as `tokenizer.pkl` |
| Token-stream hash (1,536 microbatches, 201,326,592 tokens) | `180e0a5b…`; first and last microbatch `f0fcc459…` / `3aaf3cb4…` | Recorded in `results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/token_stream_verification.json` (file sha `a0c9ec42…`, V; matches the prefix the plan cites). Also in git `e74771d`. | Replay `scripts/verify_phase1_token_stream.py` on CPU from the 13 verified files. Loader determinism is analytic (no RNG). | **regenerable** | E0 data gate (hard halt). The record exists locally, so the tokenizer manifest's "missing" status is stale. |
| `data_manifest.json` | `6a6d9c60…` | Main checkout and worktree, same hash (V) | – | **present** | Controller launch gate; E0 gate source |

### Pair-A (Claude campaign) pod-side state

| Asset | Recorded hash / version | Local copy? | How to recover | Status | Consequence |
|---|---|---|---|---|---|
| `calibration.json` (`/var/lib/autopareto/pair_a_state/calibration/`) | `211d0b74…`; N\*=85,458,944; runs 163/162/163 updates | None (V) | Cannot be recovered byte for byte. It had already gone before the 09-17 snapshot: on the continuation pod, `/var/lib/autopareto/pair_a_state` did not exist (R, transcript 2d2481de 02:17Z). Its values survive in every manifest and in `runs/pair-a-continuation-preparation-20260916/provenance/reused_calibration_20260916.json` (`86edb029…`, V present). | **lost** (values preserved) | CC: the controller never reads this file (no calibration gate in `campaign_controller.py`, V grep); N\* is a frozen constant. OC's N arm uses N\* as a constant. A fresh 3-run calibration under `continuation.py` (accept within ±1 quantum = 524,288 tokens) is optional evidence that N\* still equals 300 s of exposure on the new A40. |
| Paired baseline checkpoint **T** (`…/pair_a_official_20260916_continuation/baseline_checkpoints/t_result.checkpoint.pt`) | `0ccea0c4…`, 159,405,955 B (R) | None (V; every `.pt` under the roots was hashed) | Not recoverable | **lost** | CC and OC need it: the bracketed decode benchmark measures every candidate against it. The derived Claude manifest inherits this hash unchanged, so `preflight` fails. `run` checks only the path (I, static read of `_verify_transport_config`). A new registration is needed. |
| Paired baseline checkpoint **N** | `23be533b…`, 159,405,955 B (R) | None (V) | Not recoverable | **lost** | OC's N arm; the CC manifest also carries it, and the CLI requires it. A new registration is needed. |
| Earlier paired baselines (old A40 GPU-98579d60): T `1a601884…` (calibration run2), N smoke `d7170557…` | R | None (V) | – | **lost** | None. These were already superseded on 2026-09-16. |
| Baseline registration record `runs/pair-a-official-20260914-new-a40/provenance/baseline_registration_new_a40.json` | file `8ce36b3c…`; `starting_source_sha256` `b2fb8788…`; validation BPB 1.1998858; tokens 85,458,944 | Present in three `runs/` directories and the backup clone (V). The other `runs/` variant (`cab0d112…`, 0909) and `pair-a-seed3-baseline-registration.json` (`24ea5ed9…`) are different records. | – | **present** | The CC controller loads it and it passed the launch baseline gate (12_GPU_READINESS). |
| Registration source artifacts: calibration run1 `result.json` `5cabad22…`, run1 checkpoint `f0f6a40f…`, baseline-vs-baseline control `d9f0a93a…` | R | None (V) | – | **lost** | Provenance only. The record is self-contained. Re-running the control on the fresh A40 is advisable. |
| 0916 replacement-A40 validation receipts: N receipt `3d96f300…`, T `result.json` `a0536713…` | R | None (V) | – | **lost** | Their values are preserved in the continuation manifest. |
| Registered cache (`/var/cache/autopareto/pair_a_registered_cache`: 13 data/tokenizer files + `huggingface/hub/…flash-attn3`) and its full-snapshot inventory `3e05d2b8…` | R | None (V) | Rebuild: the 11 parquet files are downloadable-verified, the 2 tokenizer files are regenerable, flash-attn3 as below. Then root ownership, dirs 0555, files 0444, no ACLs. | **regenerable** (inventory file lost) | The CC launch gate checks the 13 files by sha256 (`_registered_cache_file_hashes`, V code). It does **not** check the kernel files. |
| Trusted repository `/opt/autopareto_trusted` (`immutable_deployment`, 122 files, plus evaluator `f05d47a1…`, active inference config `eb59d397…`, prompt tokens `78bba12b…`, schedule `ccd498d8…`) | Final 0917 manifest (file sha `b0918357…`) | **All 122 present (V).** 116 match in the worktree. The 6 Claude-amended files (`bridge`, `campaign_controller`, `candidate_process`, `isolation`, `scientist_config`, `train_launcher`) differ in the worktree by design and match in the main checkout (the executed versions). The four extra files match in both. | Deploy from the Mac. The 2026-09-15 restore installed a tar closure from the Mac the same way (R). | **present** | CC (deploy the worktree versions plus the derived manifest), OC. A Codex resume could use the main-checkout versions. |
| Pair-A study venv `/opt/autopareto_runtime` | Manifest `runtime`: Python 3.12, torch 2.8.0+cu128, triton 3.4.0, kernels 0.12.1, huggingface_hub 1.30.0, pyarrow 23.0.1, tiktoken 0.12.0, CUDA 12.8 | None. **No lockfile ever existed** (R, audit D.2). | Rebuild from the recorded recipe (R, Codex 09-14/09-16 sessions). See (c). | **regenerable, unverified** (version pins only; no wheel hashes) | CC and OC. Rebuild Env B exactly as in (c) to stay comparable with the Codex G4 arm. |
| flash-attn3 kernel, **torch28** build (what the G4 pods actually ran) | `build/torch28-cxx11-cu128-x86_64-linux/_flash_attn3_5630d32.abi3.so`, sha `2391775c…`, 804,184,512 B. Seen as cached blobs and `cpython-312` pyc on the 09-14 and 09-16 pods (R). | None (V) | Present at `e29f138f` and at `main` (= `e29f138f`), with the same sha and size (V header). | **downloadable-verified** (at risk) | CC and OC under Env B. The controller does not gate kernel identity. |
| flash-attn3 kernel, **torch29** build (manifest `runtime_cache_closure`; panel and E0/E1 env) | rev `e29f138f…`; `_flash_attn3_cuda_2d990fe.abi3.so` `385c5960…` / 804,184,680 B, plus 10 small files with recorded blob ids | None (V) | All 11 files are served at `e29f138f`: the `.so` by sha256, the small files by git-blob id and size, both equal to the recorded blob names (V). The `.so` is identical at `9542c462` and `fe6ed5bb` (V), so the panel (2026-08-26, before `e29f138f` existed) almost certainly used the same binary (I). | **downloadable-verified** (at risk) | E0, E1, E1X. The code fetches it **unpinned** (`get_kernel()` with no revision). Pre-seed the HF cache at `e29f138f`. **Risk (V):** `e29f138f` is the commit "Add model repository removal notice", which says model-type kernel repos are removed "starting from September 13, 2026". The repo is still served today. The new kernel-type repo `kernels/kernels-community/flash-attn3` has **neither** the torch28 nor the torch29 cxx11-cu128 build (V). If the model repo disappears, both registered binaries are lost. |
| Codex Scientist image `autopareto-scientist-codex:0.153.0` | `sha256:877dae34…` (base `node:22-bookworm-slim@sha256:83f487e0…`, `@openai/codex@0.153.0`) | Not in `docker image ls` (V) | A rebuild from `scientist_runtime.Dockerfile` gives a new digest (I) | **lost** | Not relevant to the Claude backend. Resuming Codex would need a manifest amendment. |
| Pod-side Codex campaign state: `/var/lib/autopareto/pair_a_official_20260916_continuation` (2.6 GB of bridge results, invocations, durable receipts), 44 positions of candidate attempts and checkpoints, payloads, `RESUME.md`, `STATE_SNAPSHOT/` (`file_manifest.sha256`, `python_env.txt`, ownership map, bash history) | R (transcript 2d2481de) | Not on the Mac (V). Mac-side `runs/…-r2` holds position records, `history.jsonl`, 23 T-arm sources, the controller state, and all 21 inherited sources with matching sha (V). | – | **lost** (pod side) | Codex candidate checkpoints cannot be re-evaluated. OC's optional Codex arm retrains from source, so it is not blocked. |
| Pod provisioning: user `autopareto-candidate`, directories, modes | uid=2001, gid=2001 (R) | No provisioning script in any repo (V grep). The recipe is in Codex session logs (R, `provisioning_commands.txt`). | Re-run the recipe (see c.3) | **regenerable** | CC and OC. `candidate_sandbox._candidate_identity()` and `_verify_permission_boundary()` fail closed without it. |

### Phase-1 / E0 / E1

| Asset | Recorded hash / version | Local copy? | How to recover | Status | Consequence |
|---|---|---|---|---|---|
| 21 panel `final.pt` (LFS) | 21 oids in `research-infrastructure` | **All 21 in `.git/lfs/objects`, sha256 = oid (V)**: 92,286,569 (AP292 ×3), 52,700,789 (AP294), 79,701,877 (AP299), 159,405,169 (AR-compatible, baseline), 80,232,461 (MOTPE), 125,845,901 (TPE). The snapshot directory holds 10 complete copies and 1 truncated one (R, inventory). | – | **present** | Panel numbers can be re-derived. On the fresh pod, re-scoring `phase1-autopareto_292-seed3/final.pt` with `evaluate_phase1_checkpoints --heldout` (recorded held-out 1.139616) is a cheap no-training check of the evaluator and kernel (suggestion). |
| `uv.lock` (panel env) | `262a4d55…` | Git `autopareto/research-infrastructure:uv.lock`, commit `e74771d` (V). Not on disk. pyproject `147f82fd…` (V), `.python-version` "3.10" `7a41a413…` (V). | `git show` as in the runbook | **present** | E0, E1, E1X. `uv run --locked` needs it. |
| E1 recipes | AR-ref `b3a961ee…` (`…diagnostic5-seed42-20260724/0005-train.py`); AP-Q `55a7fd8b…` (`autopareto-seed0-20260731/0022-train.py`) | Both checkouts (V) | – | **present** | E0, E1, E1X |
| E0/E1 plan and implementation | plan `806a6397…`; 20 implementation files | All 20 match in the worktree (V) | – | **present** | E0, E1 |
| OC/E1X pre-registrations and adapter | `CLAUDE_CAMPAIGN_CONFIRMATION_PREREG.md` `adf8ce71…`; `E1_OPUS_EXTENSION_PREREG.md` `a1f6365f…`; `scripts/pair_a_to_phase1_source.py` `b1f7737b…`; baseline `train.py` `b2fb8788…`; historical launcher `aa9dc8b2…`; `run_t_smoke` `e15cbd00…`; `run_n_smoke` `19dadd0f…` | Worktree (V) | – | **present** | OC, E1X |
| Panel-executed Phase-1 harness | `phase1_train_launcher.py` `6b39dbcc…`, `evaluate_phase1_checkpoints.py` `2c2d0e8d…` (preflight) | None (V). Current versions are `1808f371…` and `e2de6aae…`. | – | **lost** (already known, M13) | E0 run 1 is the pre-registered empirical check. No new registration is needed. |
| E0 reference values | SNAPDB `training_run` 23731 (val 1.102901); panel `phase1-autopareto_292-seed3` | `results/results.db` returns 23731 → 1.102901075 (V). Panel run records are in the snapshot directory (R). | – | **present** | E0 |
| Downstream suite datasets | lambada `900124bf…`, blimp `877fba08…`, piqa `142f6d73…`, ai2_arc `210d026f…` | Not in the local HF cache (V) | Each revision exists and equals current `main`, public, not gated (V API). The project recorded no per-file content hash. | **downloadable-verified** (by commit id only) | OC, E1X |
| `datasets` package | 5.0.1 (suite, panel downstream `environment_json`) | – | **Not in `uv.lock`** (V). It was installed ad hoc on the panel pod (I). | **downloadable-unverified** (version pin only) | OC, E1X downstream. Disclose it as an add-on to the lock. |

## (b) What is LOST, and what a new registration needs

1. **Paired baseline checkpoints T `0ccea0c4…` and N `23be533b…`.** These are the only lost items that block an experiment, and they block **CC** and **OC**.
   - **What a new registration needs.** Follow the 2026-09-16 `replacement_a40_validation` precedent:
     1. On the fresh A40, train the registered baseline (`b2fb8788…`, seed 3) with the protected harness, using `tools/run_t_smoke.py` (300 s) and `tools/run_n_smoke.py` (N\* = 85,458,944).
     2. Record the new checkpoint sha256s, validation BPB, decode median and CV, correctness, peak memory, and the GPU UUID.
     3. Run a baseline-vs-baseline bracketed control.
     4. Freeze them in a disclosed amendment of `a40.paired_baseline_checkpoints` and `a40.gpu_uuid`. `derive_claude_campaign_manifest.py` currently inherits these checkpoints unchanged, so it needs an explicit, recorded replacement.
     5. Verify the hashes with `preflight`. `run` checks only the paths.
   - **Comparability with Codex.** BPB comparisons keep the same registered baseline (BPB 1.19989), data, tokenizer (if its hash verifies), code and schedule.
     - The decode objective is normalized against a *contemporaneous* baseline in every window. The design is unchanged, but the baseline bytes differ.
     - Absolute tok/s and memory are hardware- and driver-specific.
     - The Codex G4 T arm already spans two physical A40s (positions 1–19 on GPU-98579d60, 20–44 on GPU-35264a0f). The fresh pod is a **third hardware epoch** and must be disclosed.
     - So Claude-vs-Codex differences are confounded by hardware and timing. The confound can be reduced but not removed, for example by re-running the Codex G4 best-quality source on the fresh pod, as in OC's optional arm.
   - **OC.** Its BPB contrast retrains both recipes at seeds 6–8, so it is unaffected. Its inference ratios use the new paired baseline.
2. **`calibration.json` `211d0b74…`.** It had already been absent since 2026-09-16, and the continuation proceeded on recorded values. N\* is a frozen constant, so nothing needs re-registering. Optionally, run the `continuation.py` replacement-A40 3-run calibration check: accept if |Δ| ≤ 524,288 tokens, otherwise disclose that N\* is no longer exactly 300 s of exposure on the new A40.
3. **Registration source artifacts, 0916 smoke receipts, the cache inventory, and the pod-side Codex state** (receipts, invocation logs, candidate checkpoints, `RESUME.md`, `STATE_SNAPSHOT`). None is consumed by E0, E1, CC, OC or E1X. These are provenance losses to disclose. Resuming Codex positions 45–50 would need a state-migration amendment.
4. **Codex Docker image digest `877dae34…`.** Lost. It is irrelevant to the Claude backend; a Codex resume would need an amendment.
5. **Panel-executed harness versions (M13).** Lost before this incident. E0 is the pre-registered check, and plan hash `806a6397…` is unaffected.
6. **Runbook steps that cannot be executed (not lost assets).**
   - 12_GPU_READINESS step R ("restore from `/workspace/PRESERVED_20260917T021347Z`, verify `0ccea0c4…`").
   - CAMERA_READY_E0_E1 §2/§3 ("back up `/workspace/autopareto`").
   - That volume never contained `/workspace/autopareto` (R: `PATH_ABSENT /workspace/autopareto` on 2026-09-15; the 09-17 listing shows only Pair-A backups and `recovery_home/.cache/autoresearch`).
   - These are runbook deviations, not plan-hash changes.

**At risk but recoverable today.**
- Both flash-attn3 binaries (`385c5960…` torch29 build, `2391775c…` torch28 build) exist only in the HF model-type repo that carries a removal notice. Preserve them off-pod right after the first download (~0.8 GB each). Losing them would make the kernel a lost item for E0/E1 (torch29) and CC/OC (torch28).
- No other blocker exists for **E0/E1/E1X**: every input is present, downloadable-verified, or regenerable with a recorded hash, and the comparison with the paper's panel stays valid if the E0 data gate passes.

## (c) Exact versions and pins

**c.1 Env A: Phase-1 panel, E0/E1/E1X** (`uv sync --frozen` from `uv.lock` `262a4d55…`; lock `revision = 3`)
- Python 3.10.18; panel recorded "main, Jun 4 2025, GCC 13.3.0" (R). `.python-version` pins only "3.10"; a patch mismatch is reported as INFO, not FAIL.
- uv 0.9.0 (R, `a40_environment.json`).
- From `uv.lock` (V):
  - torch `2.9.1+cu128` (download.pytorch.org/whl/cu128), triton 3.5.1, kernels 0.12.1;
  - tiktoken 0.12.0, rustbpe 0.1.0, pyarrow 23.0.1, numpy 2.2.6 (also 2.4.2 for other markers);
  - requests 2.32.5, huggingface-hub **1.5.0**, matplotlib 3.10.8;
  - nvidia-cuda-runtime-cu12 12.8.90, cudnn 9.10.2.21, cublas 12.8.4.1, nccl 2.27.5.
- pyproject `147f82fd…` pins `torch==2.9.1`, `cmaes==0.12.0` and `optuna==4.5.0`.
- `datasets==5.0.1` (downstream only; not locked).
- CUDA 12.8. Panel driver 570.211.01, Ubuntu with glibc 2.39, kernel 6.8.0-52 (R; not controllable, record only).
- flash-attn3 `kernels-community/flash-attn3@e29f138f…`, build `torch29-cxx11-cu128-x86_64-linux`, `.so` `385c5960…`.
- Data: the 13 files in `data_manifest.json`, held-out `51da1a9b…` at `915333b4…`, stream `180e0a5b…`.

**c.2 Env B: Pair-A harness, CC/OC** (must match the Codex G4 arm)
- Base image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. R for 09-01 and 09-03; I for 09-14 and 09-16, which show the same system Python 3.12.3 and torch `2.8.0+cu128` at `/usr/local/lib/python3.12/dist-packages`.
- `python3 -m venv --system-site-packages /opt/autopareto_runtime`.
- Then `pip install`:
  - pyarrow==23.0.1, rustbpe==0.1.0, tiktoken==0.12.0;
  - huggingface_hub==1.30.0, kernels==0.12.1;
  - cmaes==0.12.0, matplotlib==3.10.8, numpy==2.2.6;
  - optuna==4.5.0, pandas==2.3.3, requests==2.32.5;
  - safetensors (unpinned);
  - then **triton==3.4.0**, which replaced 3.5.1 on 09-16.
- `pip check` passed; `import triton,torch` printed `3.4.0 2.8.0+cu128 True` (R).
- flash-attn3 `@e29f138f`, build **`torch28-cxx11-cu128-x86_64-linux`**, `.so` `2391775c…` (R: cached on the G4 pods).
- **Discrepancy to disclose (I).** The final manifest's `registered_artifacts.runtime_cache_closure` is the stale 2026-09-07 record: torch 2.9.1+cu128 and the torch29 build `385c5960…`. Its `runtime` block, and the 09-14/09-16 pod outputs, say torch 2.8.0+cu128 and triton 3.4.0 with the torch28 build. Kernel identity is not gated by the controller.
- Manifest constants: `candidate_uid` 2001, CUDA 12.8, flash-attn3 revision `e29f138f…`.
- Wheels and the RunPod image tag were not checked: those hosts are outside the permitted HF-only checks. Env B has no lockfile and no wheel hashes.

**c.3 Pod-side state Env B must recreate** (from `candidate_sandbox.py`, V code, and the recorded 09-14/09-16 commands, R)
- **Candidate identity:**
  - `groupadd --system --gid 2001 autopareto-candidate`;
  - `useradd --system --uid 2001 --gid 2001 --home-dir /var/lib/autopareto/candidate-home --create-home --shell /usr/sbin/nologin autopareto-candidate` (09-16; 09-14 used `/nonexistent`);
  - non-root; `setpriv` present.
- **Trusted, root-only paths:**
  - `/opt/autopareto_trusted`: root, no group/other bits (`chmod -R u+rwX,go-rwx`).
  - `/var/lib/autopareto/<trusted_state_root>` (CC: `pair_a_claude_opus55_t_seed3`): root 0700.
  - `/var/lib/autopareto/pair_a_trusted_backup`: root 0700, if present.
- **Registered cache** `/var/cache/autopareto/pair_a_registered_cache/{data,tokenizer,huggingface/hub/models--kernels-community--flash-attn3}`: root-owned, dirs 0555, files 0444, no POSIX ACLs, no symlinks escaping the cache.
- **Study venv** `/opt/autopareto_runtime`: root-owned, not group/other-writable.
- **Candidate paths:**
  - `/opt/autopareto-candidate-attempts` (root, 0711/0755);
  - `/opt/autopareto_candidate_contexts` (0755);
  - `/opt/autopareto_candidate_inputs/per_attempts` (0711);
  - `/opt/autopareto_candidate_runtime`;
  - `/var/lib/autopareto/candidate_attempts`;
  - `/opt/autopareto-candidate-tmp` (2001:2001 0700, 09-14 recipe).
- **Storage.** All of these sit on the **ephemeral container rootfs**. Mirror them to `/workspace` and off-pod before any stop, using `rsync --chmod=Du+w` for the read-only trees.
- **Paired baseline checkpoints.** New ones at `…/<state>/baseline_checkpoints/{t,n}_result.checkpoint.pt`, registered by sha256 (see b.1).
- **Mac side (not pod).** Claude CLI 2.1.214 (frozen in the derived manifest). The controller is run with `/opt/homebrew/bin/python3`.

## Notes
- Every recorded data hash was frozen on the Phase-1 pod cache on 2026-08-31. The panel token-stream record lists identical tokenizer hashes and identical shard sizes; the panel recorded no shard sha (R).
- The HF checks ran on 2026-11-02 around 07:30 UTC. `climbmix-400b-shuffle` `main` = `915333b4` (static since 2026-03-03). `flash-attn3` `main` = `e29f138f` (2026-09-01).
- I wrote nothing outside this scratch folder.

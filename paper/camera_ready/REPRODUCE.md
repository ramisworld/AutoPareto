# Running and reproducing AutoPareto

> **Public release (2026-11-24).** Start with the top-level [README](../../README.md). Three blockers named below are now resolved:
> - **Lockfile:** `uv.lock` is at the repository root.
> - **Asset bundle:** `assets/build_assets.sh` rebuilds the "private asset bundle" from pinned public revisions.
> - **Base manifest and baseline record:** the Opus campaign's base manifest and baseline record are in `study/contracts_2026/provenance/`.
>
> Pod IDs and SSH hosts are redacted throughout; see [docs/PUBLIC_RELEASE_NOTES.md](../../docs/PUBLIC_RELEASE_NOTES.md).

> **Status (2026-11-14).** Written by reading the source; **no command below was executed** for this guide, and each block says so (`# not executed; derived from source`). Argument names are copied from the argparse definitions. Hosts, ports, SSH keys, pod IDs and API keys are deliberately omitted (`<HOST>`, `<KEY>`, `<pod-id>`). The paper's Section 2.6 and Appendix C summarise this file.

## What is supported today

| Level | Question it answers | Supported? | What blocks an outside researcher |
|---|---|---|---|
| 1. Artifact reproduction (no GPU) | Do the paper's numbers, tables and figures follow from the archived records? | **Yes for the author; partly for others.** `paper/build_camera_ready.sh` recomputes everything and aborts on any mismatch above 1e-9. The pre-registered E1 and Phase 5 analyses rerun from committed records. | The full build reads private inputs (database snapshot, `runs/` histories, a local-only branch); they ship only with the pending release. |
| 2. Recipe reproduction (one A40) | Does a frozen recipe give the same quality under the recorded contract? | **Yes, with setup.** Hashed plans, frozen sources and executors exist for E1 and the Opus retraining. | Training-shard revision and the flash-attention kernel are not pinned in code; `uv.lock` is on a local-only branch; the pod asset bundle is private; checkpoints are published only as hashes. |
| 3. Agent-campaign reproduction | Does a comparable autonomous search behave the same way? | **Comparable, not identical.** The Pair-A controller runs Codex CLI or Claude Code with the recorded prompts, schemas and permission checks; an offline dry run exists. | Provider model state cannot be frozen; the Codex container image was not preserved; pod setup is partly manual. The historical campaigns (AP-Q, AR-ref) were run by hand and have no entry point. |

**Checkouts.**
- **Path prefixes.** Unprefixed paths are relative to the camera-ready worktree (branch `autopareto/camera-ready-opus`). `EVIDENCE:` marks the evidence tree: the main checkout, branch `autopareto/professor-ready-2026`.
- **Remote branches.** `origin` holds `master`, `main`, `autopareto/professor-ready-2026`, `agenthub` and `exp/H100/mar8`.
- **Local-only branches.** `autopareto/camera-ready-opus` and `autopareto/research-infrastructure`. The latter holds `uv.lock`, the historical `runner.py` and the LFS panel checkpoints.
- **README and licence.** Both repositories carry the same Apache-2.0 `LICENSE` and the same top-level `README.md`, which is the August 2026 public-release README. It covers the 201.3M-token panel and a legacy figure script, but not E1, Phase 5, the camera-ready build or the agent controller.

## Level 1: artifact reproduction (no GPU)

Rebuilding the paper regenerates every number, table and figure from archived records.

```sh
# not executed; derived from source (paper/build_camera_ready.sh, paper/camera_ready/BUILD.md)
python3.10 -m venv /tmp/ap-paper && /tmp/ap-paper/bin/pip install -r paper/requirements.txt
#   pins: matplotlib==3.10.9 numpy==2.2.6 pypdf==6.17.0 beautifulsoup4==4.15.0 ; draft built with CPython 3.10.20
AUTOPARETO_PAPER_PYTHON=/tmp/ap-paper/bin/python AUTOPARETO_EVIDENCE_ROOT=/path/to/main/checkout \
  sh paper/build_camera_ready.sh      # tectonic 0.17.0 on PATH -> paper/camera_ready/autopareto_camera_ready_draft.pdf
```

- **Defaults.** `AUTOPARETO_PAPER_PYTHON` defaults to `paper/.venv/bin/python`, which does not exist in this worktree. `AUTOPARETO_EVIDENCE_ROOT` defaults to the author's absolute path, set in `analysis/cr_common.py`.
- **Steps.** `analysis/cr_evidence.py`, `lineage_labels.py`, `cr_tables.py`, `cr_figures.py`, `cr_v2.py` and `generate_figures.py`, then `tectonic main.tex`, then `cr_page_check.py --write`.
- **Self-check.** `cr_evidence.py` aborts if any recomputed quantity differs from `audit/camera_ready_2026/supporting/{statistical_audit,claims_recomputation}.json` by more than 1e-9.
- **Private inputs** (in no public branch):
  - the SQLite snapshot `EVIDENCE:results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db` (`*.db` is git-ignored);
  - `EVIDENCE:results/logs/` and `EVIDENCE:results/candidates/` (the latter partly untracked);
  - the agent-arm histories `EVIDENCE:runs/pair-a-claude-opus55-t-seed3-new-a40-continuation-20261105c/pair-a-fixed-time-seed3/history.jsonl` and `EVIDENCE:runs/pair-a-official-20260914-new-a40-continuation-20260916-r2/pair-a-fixed-time-seed3/history.jsonl` (`runs/` is untracked);
  - the selection manifests, read through `git show autopareto/research-infrastructure:results/...`.
- **Committed inputs.** The E1, Phase 2 and Phase 5 records are under `results/camera_ready_2026/`.

The committed analyses can be recomputed on a CPU from the committed run records; they need only matplotlib. Write each output to a new directory:

```sh
# not executed; derived from source
python3 scripts/analyze_camera_ready_e1.py --runs-root results/camera_ready_2026/amendment1/E1 --output-dir /tmp/e1   # or --self-test; --no-figure
python3 scripts/analyze_e1_exposure_decomposition.py --runs-root results/camera_ready_2026/amendment1/E1 --out-dir /tmp/e1x
python3 scripts/analyze_phase5.py --out /tmp/p5 [--no-figures] [--skip-5b]   # reads phase5/A and phase5/B (B/runs_json.tar.xz)
```

`scripts/analyze_phase2_verification.py` cannot be rerun from the repository. Its required `--pod-copy`, `--winner-dir`, `--august-bundles` and `--campaign-results` are local copies of pod directories.

## Level 2: recipe reproduction (one NVIDIA A40)

**Environment ("Env A").** The panel, E1 and Phase 5 runs used:
- Python 3.10.18 and uv 0.9.0, installed with `uv sync --frozen` against `uv.lock` (sha256 `262a4d55…`);
- torch 2.9.1+cu128, triton 3.5.1, kernels 0.12.1, numpy 2.2.6, pyarrow 23.0.1, tiktoken 0.12.0 and rustbpe 0.1.0;
- `datasets==5.0.1`, outside the lock.

Sources: `environment_expectations` in `evaluation/camera_ready_e0_e1_plan_amendment1.json`, `study/contracts_2026/provenance/a40_environment.json`, and `pyproject.toml` (torch comes from the cu128 index). `uv.lock` exists only on the local branch: `git show autopareto/research-infrastructure:uv.lock > uv.lock`.

**Pod setup** is semi-manual. `audit/camera_ready_2026/13_FRESH_POD_SETUP.md` specifies the RunPod image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, at least 60 GB of container disk, at least 150 GB of `/workspace`, and no `PYTORCH_CUDA_ALLOC_CONF`.

```sh
# not executed; derived from source (scripts/pod/fresh_a40_bootstrap.sh; as root, from /workspace/autopareto-camera-ready)
bash scripts/pod/fresh_a40_bootstrap.sh check|assets|env-a|env-b|pair-a|verify|mirror|all
```

The `assets` and `pair-a` stages run `install_assets.sh` from a **private asset bundle** that is not in the repository. The bundle holds the shards, the tokenizer and an HF kernel cache. Without it, build the cache yourself:

```sh
# not executed; derived from source (prepare.py:375-378)
python prepare.py --num-shards 10 --download-workers 8   # -> ~/.cache/autoresearch/{data,tokenizer}
```

- **Shards.** `prepare.py` downloads 10 training shards and the validation shard `shard_06542.parquet` from `huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main`.
  - The code does not pin a revision. The recorded revision is `915333b4f8b8684f39aeaafea600fea6f43fb703`.
  - `evaluation/downstream_suite.json` records that revision and also pins the held-out shard `shard_06541.parquet`.
- **Tokenizer.** It is retrained deterministically. The hashes must match `tokenizer.pkl` `888c9fa8…` and `token_bytes.pt` `23517e05…` (`study/contracts_2026/provenance/tokenizer_manifest.json`, `data_manifest.json`).
- **Data gate.** E0 checks 13 file hashes and the token-stream hash `180e0a5b…` before any GPU run.
- **Flash-attention kernel.** Each recipe calls `kernels.get_kernel("kernels-community/flash-attn3")` at import, without a pinned revision.
  - The recorded build is `e29f138f`, and upstream now carries a removal notice.
  - Run with `HF_HUB_OFFLINE=1` against a cache that already holds that build.

**Frozen recipes.** Each source is sha256-gated by its plan. Model IDs are in `evaluation/phase1_equal_token_preflight.json`.

| Name (old ID) | Source | sha256 |
|---|---|---|
| AR-ref (exp 68) | `results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py` | `b3a961ee…` |
| AP-Q (exp 134, AP292) | `results/candidates/autopareto-seed0-20260731/0022-train.py` | `55a7fd8b…` |
| AP-F (exp 137) / AP-S (exp 136) | `…/autopareto-seed0-20260731/0025-train.py` / `0024-train.py` | `36e3ad91…` / `6d0d8d1d…` |
| TPE (183) / MOTPE (234) / Baseline | `results/candidates/{tpe-seed42-20260805/0021,motpe-seed42-20260805/0022,baseline-a40-seed0/0002}-train.py` | in plan |
| Opus A23/A24/A27 (Phase 5A) | `evaluation/phase5a_sources/opus_attempt{23,24,27}.phase1.train.py` (from `scripts/opus_contract_t_to_phase1_source.py`) | in plan |

**E0/E1 executor.** It compares AP-Q with AR-ref at 88.6M, 157.3M and 201.3M tokens and at 300 s, on seeds 6–8. The plan is `evaluation/camera_ready_e0_e1_plan_amendment1.json` (sha256 `89f29015…`). The executor only validates unless `--execute` is passed.

```sh
# not executed; derived from source (scripts/run_camera_ready_e0_e1.py:1336-1344)
HF_HUB_OFFLINE=1 uv run --locked python scripts/run_camera_ready_e0_e1.py      # validate only [--json]
HF_HUB_OFFLINE=1 uv run --locked python scripts/run_camera_ready_e0_e1.py --stage E0 --execute [--expect-plan-sha256 S]
HF_HUB_OFFLINE=1 uv run --locked python scripts/run_camera_ready_e0_e1.py --stage E1 --execute [--resume] [--fixed-time-launcher P]
```

- **Token cells.** They use `run_one` and `evaluate_one` from `scripts/run_phase1_equal_tokens.py`. Those call `-m protected.phase1_train_launcher --source --source-sha256 --token-budget --seed --run-dir …`, then `-m scripts.evaluate_phase1_checkpoints --candidate --milestone-manifest --output [--heldout] [--suite]`.
- **300 s cell.** It uses `study/contracts_2026/historical_reference/protected/train_launcher.py` (sha256 `aa9dc8b2…`). The executor sets `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` for it.
- **Output root.** It is fixed inside the hashed plan at `results/camera_ready_2026/amendment1`.
  - The executor refuses to run if an E0 verdict or run directories already exist there, and this worktree has both committed. Rerun in a clone with that directory moved aside.
  - E1 needs an E0 `PASS` under the same plan hash.
  - Off RunPod, the `/workspace` checks report `POD_ONLY`.
  - Doc 13 estimates E1 at about 6.4 A40-hours.

**Other executors.** Each validates by default and runs only with `--execute`.
- **201.3M-token panel** (seven recipes, seeds 3–5). Command: `scripts/run_phase1_equal_tokens.py --stage {smoke,B,B-inference,C,C-inference} --execute`, with plan `evaluation/phase1_equal_token_preflight.json`.
  - It writes into `results/phase1`.
  - It needs `results/phase1/token_stream_verification.json`, which exists only in the private pod snapshot.
  - Treat it as historical and use the E1 executor for new runs.
- **Phase 5A** (Opus recipes at 201.3M tokens, seeds 6–8, plus an AR-ref drift control). Command: `scripts/run_phase5a_opus_fixed_tokens.py [--stage train|downstream] --execute [--expect-plan-sha256 S]`. Plan: `evaluation/phase5a_opus_fixed_tokens_plan.json`, pre-registered in `evaluation/PHASE5_PREREG.md`.
- **Phase 5B** (inference ablations).
  - `scripts/phase5b_inference_ablation.py make --sources --prepare --out` builds the variant contexts.
  - `scripts/pod/phase5b_run_ablations.sh` measures them. It needs the Pair-A checkpoints, which are not distributed, and pod paths under `/opt` and `/var/cache`.

**Private dependencies at this level:** `uv.lock` (on the local-only branch); the asset bundle; the unpinned shard download and kernel; checkpoints, which are published only as hashes; and RunPod `/workspace` storage.

## Level 3: agent campaign reproduction (Pair-A, Codex CLI or Claude Code)

**The historical discovery campaigns that produced AP-Q and AR-ref have no entrypoint.** `audit/camera_ready_2026/25_HISTORICAL_AGENT_PROVENANCE.md` records how they ran:
- **Operator.** They were run by hand through the Codex desktop app with gpt-5.6-sol at high effort; seed-42 A1–A5 used Cursor's agent.
- **Runner.** The agent drove the protected `runner.py` on the pod over SSH, following `programs/program_autopareto.md`.
- **Where the runner lives.** It exists only on the local branch (`git show autopareto/research-infrastructure:runner.py`). A snapshot is in `study/contracts_2026/historical_reference/runner.py`.
- **Runner arguments.** Its `register-campaign` and `run` subcommands take `--campaign-id`, `--seed`, `--parent-experiment-id`, `--coding-model` and `--reasoning-level`.

**Pair-A controller.** The Pair-A campaigns (contract T: 300 s, seed 3) do have a controller. It runs on the operator's machine using only the Python standard library, runs the agent CLI locally and drives the A40 over SSH.

```sh
# not executed; derived from source (study/contracts_2026/campaign_controller.py:1585-1654); from the repo root
PYTHONPATH=$PWD python3 -B -m study.contracts_2026.campaign_controller run --execute \
  --contracts T --agent-backend claude|codex --state-root S --baseline-record B \
  --manifest M --manifest-sha256 SHA --scientist-bin BIN --model ID --reasoning EFFORT \
  --scientist-timeout-seconds 600 --a40-host <HOST> --a40-identity <KEY> [--a40-port P] \
  --a40-repository /opt/autopareto_trusted --a40-python /opt/autopareto_runtime/bin/python \
  --a40-study-venv /opt/autopareto_runtime --a40-cache /var/cache/autopareto/pair_a_registered_cache \
  --a40-state-root /var/lib/autopareto/<run> --a40-paired-baseline-t-checkpoint T.pt --a40-paired-baseline-n-checkpoint N.pt
# siblings: `dry-run --position N --model --reasoning ...` (no agent, no A40); `preflight --state-root ...` (no launch)
```

- **Contracts.** `--contracts` takes `T,N` (default) or `T`; contract N uses N\*=85,458,944. The Claude arm ran `T`. The Codex arm ran `T,N`, and the paper uses only its T campaign.
- **Remote environment ("Env B").** The A40 side runs `/opt/autopareto_runtime`: torch 2.8.0+cu128 and triton 3.4.0, with no lockfile. The pins are in `fresh_a40_bootstrap.sh env-b`.
- **Codex arm, recorded launch.** `EVIDENCE:runs/pair-a-official-20260914-new-a40-continuation-20260916-r2/controller_logs/launch_controller.sh`, for its 2026-09-14 state root: gpt-5.6-sol, `--reasoning high`, codex-cli 0.153.0, 600 s, default contracts.
- **Claude arm, recorded launch.** `audit/camera_ready_2026/15_CAMPAIGN_READINESS.md` §3: claude-opus-5-5, `--reasoning xhigh`, Claude Code 2.1.318, 600 s, later raised to 900 s by a recorded amendment (doc 20).

**Agent command lines** (`_scientist_command`, `campaign_controller.py:389-486`). Both backends get the same prompt and phase schema.
- **Claude** (host): `claude -p <prompt> --model M --effort E --output-format stream-json --verbose --json-schema <schema> --no-session-persistence --safe-mode --setting-sources local --strict-mcp-config --disable-slash-commands --tools Read,Edit,Write,Glob,Grep --permission-mode acceptEdits`.
  - `study/contracts_2026/isolation.py` requires these flags and forbids others, such as `--dangerously-skip-permissions`, `--add-dir` and `--resume`.
  - Effort must be one of `low`, `medium`, `high`, `xhigh`, `max`.
- **Codex**: `codex exec --ephemeral --ignore-user-config --skip-git-repo-check -c approval_policy="never" --json --output-schema <schema> -C <context> -s <sandbox> -m M -c model_reasoning_effort=E <prompt>`.
  - The controller has no execution-backend flag, so Codex always runs in the Docker image `autopareto-scientist-codex:0.153.0`, with `danger-full-access` inside the container.
  - The image comes from `study/contracts_2026/scientist_runtime.Dockerfile`. No build command is recorded, and the original image is lost.
  - Codex authentication is mounted at `/run/codex/auth.json`.

**Program, context and parent rule.**
- **Programs.** `study/contracts_2026/programs/program_fixed_time.md` (contract T) and `program_fixed_tokens.md` (contract N).
- **Agent context.** Each call sees `train.py`, `prepare.py`, `study_runtime.py`, `program.md` and `campaign_view.json`. Only `train.py` is writable (`isolation.py`, `context_allowlist.json`).
- **Two calls per attempt.**
  - `select_parent`, with `scientist_parent_selection_schema.json`.
  - `edit_child`, with `scientist_child_edit_schema.json`.
- **Parent check.** `_validate_parent_selection` requires the parent to appear exactly once in `source_registry.jsonl`, belong to the same campaign, be `parent_eligible`, and match its recorded sha256.
- **Eligibility.** `campaign.candidate_disposition` requires a finite validation BPB and a passed cached-decode correctness check.
- **Schedule.** It is hash-frozen in `study/contracts_2026/provenance/pair_a_interleaved_same_index_schedule_20260902.json`.

**State.**
- `S/controller_state.json` and `S/provenance/*.json` (the frozen manifests).
- `S/pair-a-fixed-time-seed3/`, which holds:
  - `history.jsonl`, `campaign_view.json` and `source_registry.jsonl`;
  - `sources/`, the lineage;
  - `position-NNN-*.json`, `pending/` and `scientist_diagnostics/`.
- A position is never retried automatically.

**Wrappers.** The Claude arm used these:

```sh
# not executed; derived from source
python3 -B scripts/campaign_supervisor.py run --supervisor-dir D --state-root S [--notify] [--usage-pause-utilization F] [--label L] -- <controller argv>
python3 -B scripts/campaign_supervisor.py status|watch --supervisor-dir D --state-root S
python3 -B scripts/run_extension_attempt.py prepare --state-root S | run|campaign|arm <controller run args>
python3 -B scripts/auto_requeue_scientist_timeouts.py --state-root S --supervisor-dir D [--max-timeouts-per-position 3] [--max-total-requeues 12] -- <supervisor cmd>
PYTHONPATH=$PWD python3 -B scripts/check_campaign_launch_gates.py --manifest M --manifest-sha256 S --state-root R --baseline-record B ...  # Claude only
```

**Setup before `run`.** These steps are partly manual; audit docs 13, 15 and 20 record them.
1. **Pod.** Run `fresh_a40_bootstrap.sh pair-a`. It creates the candidate account (UID 2001), copies the 122 files listed in `scripts/pod/trusted_files.txt` to `/opt/autopareto_trusted`, and installs the read-only registered cache.
2. **Paired baselines.** Re-register them on the new A40 with `study/contracts_2026/tools/run_t_smoke.py` (no arguments) and `run_n_smoke.py --n-star 85458944`. Both are configured through environment variables such as `AUTOPARETO_TRUSTED_STATE_ROOT`.
3. **Claude manifest.** Freeze it with `study/contracts_2026/tools/derive_claude_campaign_manifest.py --base --output --state-root --trusted-state-root --scientist-bin --model --reasoning [--a40-ssh-host] [--a40-gpu-uuid] [--a40-paired-baseline-t PATH:SHA] [--a40-paired-baseline-n PATH:SHA]`.
   - `--base` is the last Codex manifest, `EVIDENCE:runs/pair-a-official-20260914-new-a40/provenance/final_pre_study_manifest_pair_a_20260914_new_a40.json`.
   - No tool freezes a new Codex manifest; those were frozen in September 2026.
   - The Codex arm ran from the main checkout's untracked `study/`, which differs from this worktree in six amended files.

**Offline rehearsal** (no model call, GPU or network):
- `python3 -m study.contracts_2026.tools.dry_run_claude_campaign --state-root <new empty dir>` runs against the fake CLI in `tools/fake_claude_cli/` and a mocked A40.
- `python -m unittest discover -s study/contracts_2026/tests -t .` runs the unit tests.
- `study/contracts_2026/tools/claude_canary.py --output F --execute` makes one real model call.

**Private dependencies at this level:**
- **Provider state:** model IDs `gpt-5.6-sol` and `claude-opus-5-5`, CLI auto-updates and seat limits.
- **Credentials:** Codex and Claude subscription authentication.
- **Infrastructure:** the lost Codex image; the RunPod SSH proxy; pod-local `/var/lib` state; paired-baseline checkpoints that exist only on the pod.
- **Logs:** the historical Codex and Cursor session logs, which stay private because they contain credentials.


## The 13-recipe test (Phase 6), added 2026-11-18

- **Plan and pre-registration:** `evaluation/phase6_unified_plan.json` (SHA-256 `dc1e605e`), `evaluation/PHASE6_PREREG.md`;
  amendments `evaluation/PHASE6_AMENDMENT_A1..A4.md`; incidents `evaluation/PHASE6_INCIDENTS.md`.
- **Analysis (CPU, from the archived run records in `results/camera_ready_2026/phase6/`):**
  `python3 analysis/cr_p6.py` writes `results/camera_ready_2026/phase6/analysis/unified_13.json`;
  `python3 analysis/cr_p6_paper.py` writes Figures 1 and 3, the 13-recipe tables and `cr_macros_p6.tex`,
  including the post hoc `stability_check` of the timing-flagged inference runs. Both run inside
  `paper/build_camera_ready.sh`.
- **Re-running the GPU work (one A40; each executor only validates without `--execute`):**
  - `python scripts/run_phase6.py --stage train --execute` (201.3M-token runs and drift controls);
  - `python scripts/run_phase6.py --stage downstream --execute` (frozen four-task evaluator);
  - `python scripts/run_phase6.py --stage inference --block {P0a,P0b,control,anchor} --execute`
    (runs `scripts/phase6_unified_inference.py` once per recipe and seed);
  - `python scripts/run_phase6_p2.py --stage {pilot,fixed,lmeval} --execute` (300 s runs and lm-eval, amendment A1);
  - amendments A2 to A4: `scripts/run_phase6_a2.py`, `run_phase6_a3.py`, `run_phase6_a4.py`.
  - Every executor accepts `--expect-plan-sha256` to refuse a changed plan.
- Flags were checked against each script's argument parser on 2026-11-18; no command was executed for this note.

# AutoPareto

An AI research agent edits a small Transformer's `train.py`, trains each version for 300 s on one NVIDIA A40, and is
scored on three objectives at once: quality (bits per byte), decode speed and memory. We ran it with GPT-5.6 (Codex CLI)
and Claude Opus 5.5 (Claude Code). Opus shrank the model and rewrote the inference engine: about 32× faster decoding and
4.6× faster generation.

**Paper:** [OpenReview](https://openreview.net/forum?id=U9ICjFvtN8) · **Poster:** [PDF](poster/out_v7/autopareto_poster_v7_A1.pdf) ·
NeurIPS 2026 AutoMLR workshop

## Replicate it in five steps

### 1. Clone

```bash
git clone https://github.com/ramisworld/autopareto.git
cd autopareto
```

### 2. Check the paper's numbers (no GPU, one minute)

```bash
pip install uv
uv run --no-project --python 3.10 python analysis/cr_p6.py
git diff --stat
```

This recomputes the paper's 13-recipe test from the run records in `results/` and prints one line per recipe, for
example `Opus A27  bpb=1.1580±0.0020 fixed=1.159 decode=4583 x32.01 …`. `git diff` prints nothing: the recomputed numbers
are identical to the published ones. Newer Pythons can differ in the last digit of a few standard deviations.

### 3. Rent one A40 and set it up

Rent **one NVIDIA A40 (48 GB)**. On RunPod use the image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, a 60 GB
container disk and a 150 GB volume on `/workspace`. Then, on the GPU machine as root:

```bash
git clone https://github.com/ramisworld/autopareto.git /workspace/autopareto-camera-ready   # the setup expects this path
cd /workspace/autopareto-camera-ready
bash scripts/pod/fresh_a40_bootstrap.sh check     # one A40, CUDA 12.8 or newer, enough disk
bash scripts/pod/fresh_a40_bootstrap.sh env-b     # the runtime the agent's training runs use
AP_PYTHON=/opt/autopareto_runtime/bin/python bash assets/build_assets.sh /workspace/assets   # data and kernel, 2.6 GB
bash scripts/pod/fresh_a40_bootstrap.sh assets    # installs them
bash scripts/pod/fresh_a40_bootstrap.sh env-a     # the pinned environment from uv.lock
```

The data and the flash-attention kernel are downloaded from pinned Hugging Face revisions, and every file is checked
against its recorded SHA-256. Any mismatch stops the setup.

### 4. Retrain the recipes

```bash
mv results/camera_ready_2026/phase6 results/camera_ready_2026/phase6.published   # keep the published runs to compare
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/run_phase6.py                       # checks only; changes nothing
.venv/bin/python scripts/run_phase6.py --stage train --execute --run-ids p6-arref-tokens201326592-seed6-drift
python3 scripts/compare_with_published.py p6-arref-tokens201326592-seed6-drift
```

That run retrains the AutoResearch reference recipe (AR-ref, seed 6) for 201.3M tokens, which takes about 20 minutes.
The study's pre-registered rule accepts a rerun whose held-out BPB is within 0.005 of the original run's 1.0947; our own
rerun gave 1.0968.

To retrain all 27 runs (about 8 A40-hours):

```bash
.venv/bin/python scripts/run_phase6.py --stage train --execute
```

The inference, 300 s and downstream stages are described in [evaluation/PHASE6_PREREG.md](evaluation/PHASE6_PREREG.md).

### 5. Run your own agent campaign (GPT or Opus)

Rehearse the whole loop first. This needs no GPU and makes no model calls; it takes a few seconds:

```bash
python3 -m study.contracts_2026.tools.dry_run_claude_campaign --state-root /tmp/autopareto-dry-run
```

A real campaign needs three things: the A40 from step 3, your laptop with Claude Code or Codex CLI logged in, and SSH
access from the laptop to the A40.

**On the A40:**

```bash
bash scripts/pod/fresh_a40_bootstrap.sh pair-a     # sandbox user, read-only data, trusted copy of the harness
G2_STATE=/var/lib/autopareto/my_baseline bash scripts/pod/g2_reregister_pair_a_baseline.sh   # registers the starting model
nvidia-smi -L                                      # note the GPU UUID
```

The second command prints the paths and SHA-256 hashes of two baseline checkpoints. You need them below.

**On your laptop, for Claude Opus 5.5:**

<details>
<summary>Freeze the campaign settings for your A40, then launch 25 attempts</summary>

```bash
git clone https://github.com/ramisworld/autopareto.git && cd autopareto
B=/var/lib/autopareto/my_baseline/nonstudy_tests
PYTHONPATH=$PWD python3 study/contracts_2026/tools/derive_claude_campaign_manifest.py \
  --base study/contracts_2026/provenance/public_base_manifest_opus_campaign.json \
  --output runs/opus/manifest.json --state-root runs/opus --trusted-state-root /var/lib/autopareto/my_baseline \
  --scientist-bin "$(which claude)" --model claude-opus-5-5 --reasoning xhigh --scientist-timeout-seconds 900 \
  --a40-ssh-host root@<A40 address> --a40-gpu-uuid <GPU UUID> \
  --a40-paired-baseline-t $B/t_smoke_artifacts/result.checkpoint.pt:<its sha256> \
  --a40-paired-baseline-n $B/n_smoke_artifacts/result.checkpoint.pt:<its sha256>

PYTHONPATH=$PWD python3 -B -m study.contracts_2026.campaign_controller run --execute \
  --contracts T --agent-backend claude --model claude-opus-5-5 --reasoning xhigh --scientist-timeout-seconds 900 \
  --scientist-bin "$(which claude)" --state-root runs/opus \
  --manifest runs/opus/manifest.json --manifest-sha256 <sha256 printed by the previous command> \
  --baseline-record study/contracts_2026/provenance/baseline_registration_new_a40.json \
  --a40-host root@<A40 address> --a40-port <SSH port> --a40-identity ~/.ssh/id_ed25519 \
  --a40-repository /opt/autopareto_trusted --a40-python /opt/autopareto_runtime/bin/python \
  --a40-study-venv /opt/autopareto_runtime --a40-cache /var/cache/autopareto/pair_a_registered_cache \
  --a40-state-root /var/lib/autopareto/my_opus_campaign \
  --a40-paired-baseline-t-checkpoint $B/t_smoke_artifacts/result.checkpoint.pt \
  --a40-paired-baseline-n-checkpoint $B/n_smoke_artifacts/result.checkpoint.pt
```

- **Results:** each attempt's code, hypothesis and scores are written to `runs/opus/`.
- **Resume:** rerun the same command.
- **Our settings:** the paper's Opus campaign used Claude Code 2.1.318 at effort `xhigh`. Its timeout was 600 s at
  first and was raised to 900 s from attempt 9.

</details>

**For GPT-5.6:**
- **Command:** use the same controller with `--agent-backend codex --model gpt-5.6-sol --reasoning high`.
- **Codex image:** Codex runs inside a Docker image. Build it with
  `docker build -f study/contracts_2026/scientist_runtime.Dockerfile -t autopareto-scientist-codex:0.153.0 study/contracts_2026`.
- **Settings file:** the step that freezes campaign settings for a new A40 is automated only for Claude so far
  (`derive_claude_campaign_manifest.py`).
- **Our settings:** the paper's GPT campaign used codex-cli 0.153.0 at `high` with a 600 s timeout.

Agent campaigns are comparable, not identical: provider models change over time.

## What is where

| Folder | Contents |
|---|---|
| `study/contracts_2026/` | the agent loop: controller, sandbox, prompts (`programs/`), schemas, tests |
| `scripts/` | experiment runners; each only validates unless given `--execute` |
| `results/camera_ready_2026/` | every run record behind the paper |
| `analysis/` | scripts that recompute the paper's numbers, tables and figures |
| `paper/`, `poster/` | the paper sources and the poster |

- **Every runner, and what is and is not reproducible:** [paper/camera_ready/REPRODUCE.md](paper/camera_ready/REPRODUCE.md).
- **What was removed or redacted for this public release:** [docs/PUBLIC_RELEASE_NOTES.md](docs/PUBLIC_RELEASE_NOTES.md).
- **The August 2026 release README:** [docs/README_AUGUST_2026_RELEASE.md](docs/README_AUGUST_2026_RELEASE.md).
- **License:** Apache-2.0.

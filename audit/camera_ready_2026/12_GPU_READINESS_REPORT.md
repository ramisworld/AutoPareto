# 12 — GPU-phase readiness report (2026-11-02)

Branch `autopareto/camera-ready-opus`, worktree `/Users/Rami/Documents/dev/autopareto-camera-ready`.
No model call, RunPod action, GPU run, upload or destructive git action has been made.

## 1. Evidence preservation

| Item | State |
|---|---|
| Full APFS clone (repo, `.git`, LFS, results, runs, paper) | `/Users/Rami/Documents/dev/autopareto_evidence_backup_20261102/autopareto_apfs_clone`, read-only; 15,094 files in original and clone; 1,596 evidence files (11.71 GB) hash-identical; 21 LFS objects verified |
| Git state | `state/git_head.txt` (6bcb69e on `autopareto/professor-ready-2026`), refs, porcelain status, stash, worktrees, uncommitted-diff patch |
| Branch bundle | `state/autopareto_branches.bundle` (26.5 MB), `git bundle verify` passed, test restore of `research-infrastructure` (930 files at e74771d) |
| Limitation | Same physical disk only (43 GiB free, no external volume). Needs an off-machine destination. |

## 2. Code changes (commits on the branch)

| Commit | Change |
|---|---|
| 7551f83 | Snapshot of previously untracked controller, analysis, audit and paper sources |
| bb7f088 | Import of the uncommitted `protected/inference_adapter.py` (sha 80f4d8c9…); matches the hash executed in the last Codex continuation |
| 73651dc | Harness fix + Claude backend + tests + dry-run tools |
| 8ec53fb | `tools/derive_claude_campaign_manifest.py` |
| 0865556 | `tools/claude_canary.py` |

Line counts against 7551f83 (`git diff --numstat`): bridge.py +79/−5, campaign_controller.py +69/−7,
candidate_process.py +23, isolation.py +64, scientist_config.py +46/−2. New: two test files
(318 lines), dry-run tool, offline fake CLI, manifest derivation, canary.

### 2.1 Harness fix (the 3.2159 BPB bug)
* `candidate_process.py`: a candidate training body that returns before the protected stop raises
  `CandidateExitedBeforeStop` (classified `candidate` → `scientific_failure`) instead of re-entering
  the loop and checkpointing an untrained model. Before `torch.save`, the candidate's own `step`
  must agree with the observed optimizer updates (±1).
* `tests/test_harness_regressions.py` runs the real worker in-process on CPU: 5/5 pass on the fixed
  worker; 3/5 fail on the original worker (the tests detect the bug).
* Not added, to keep the search policy identical to the Codex campaign: a quality-sanity bound for
  parent eligibility, noise margins in Pareto acceptance, and harness-enforced LR progress
  (audit 08, E8 items 3–5).

### 2.2 Claude Code backend
* Selected only by `--agent-backend claude`; default stays `codex`. `ScientistConfig.as_dict()` is
  byte-identical for Codex, so every frozen Codex manifest still verifies.
* Invocation: `claude -p <same prompt> --model claude-opus-5-5 --effort xhigh --output-format
  stream-json --verbose --json-schema <same registered schema> --no-session-persistence --safe-mode
  --setting-sources local --strict-mcp-config --disable-slash-commands --tools
  Read,Edit,Write,Glob,Grep --permission-mode acceptEdits`, cwd = the staged context.
* Refused flags: `--dangerously-skip-permissions`, `--allow-dangerously-skip-permissions`,
  `--add-dir`, `--fallback-model`, resume/continue/fork, MCP/plugin/agent/settings, system-prompt
  overrides, `--bare`, `--allowedTools`. Model and effort must equal the frozen config.
* Environment: `ANTHROPIC_*`, `CLAUDE_*`, `AWS_*`, `GOOGLE_*`, `VERTEX_*`, OpenAI and Codex variables are
  removed; auto-updater disabled. Subscription login is used via HOME/Keychain; any
  `apiKeySource` other than `none` is rejected (no API-key billing).
* Stream checks per call: exactly one init event whose `model` equals the configured model (no silent
  substitution), `permissionMode` acceptEdits, no tools beyond the five, no MCP servers, success
  result, the configured model present in `modelUsage`, exactly one schema-valid proposal.
* Single-arm mode: `run --contracts T` runs the 25 fixed-time positions of the frozen schedule; the 25
  fixed-token positions are recorded `not_in_study`. Resume checks the stored active set.
* Latent controller bug fixed: committing attempt 25 tried to build the view for attempt 26 and
  left the position uncommitted.

### 2.3 Equivalence with the Codex workflow

| Condition | Codex G4 T arm | Claude backend | Evidence |
|---|---|---|---|
| Prompt text | controller prompt | identical bytes | `test_claude_command_reuses_the_codex_prompt_and_schema` |
| Output contract | 2 registered JSON schemas | same files via `--json-schema` | same test; `_validate_claude_command` |
| Two calls per attempt (select parent, edit child) | yes | yes; child edit must reproduce the controller-held plan | dry run |
| Fresh agent per call, no shared history | `codex exec --ephemeral` | `--no-session-persistence`, new process | command validation |
| Visible context | `train.py`, `campaign_view.json`, `history.jsonl`, `results/` | same staged directory, same file modes; post-call context hash checks unchanged | bridge (unchanged checks) |
| Code execution by the agent | shell utilities in `node:22-bookworm-slim` container; no Python/torch, so training code could not be run | no shell; file tools only; cannot run code | Dockerfile, tool list |
| Reading outside the context | impossible (container mounts) | expected to be refused by Claude Code permissions; **unverified until the canary** | `tools/claude_canary.py` |
| Contract, schedule, baseline, data, evaluator, Pareto rule | frozen manifest | inherited unchanged; only 6 file hashes + Scientist block + state roots + remote namespace change | `derive_claude_campaign_manifest.py` passes the controller's launch gate |
| Model/effort | gpt-5.6-sol / high | claude-opus-5-5 / xhigh | stream checks |

Isolation is weaker than Codex's container (host process, network, Keychain). Containment rests on
the CLI permission system, file modes and hash checks. The canary tests the key property before
any campaign call.

### 2.4 Local verification
* Unit tests: `python -m unittest discover -s study/contracts_2026/tests -t .` → 172 run, 2 errors,
  2 skipped. Both errors are environmental and pre-existing: the Codex Docker image is absent on this
  Mac, and one adapter fixture refers to a file version not preserved locally.
* Offline 25-attempt dry run (`tools/dry_run_claude_campaign.py`, fake CLI + mock A40): interruption
  at attempt 4 resumed; attempt 3 failed and was never used as a parent; 25 unique invocations
  (no attempt trained twice); all records `claude-opus-5-5`/`xhigh`; 25 positions `not_in_study`.
* Derived manifest (draft in scratchpad): changes exactly the 5 amended hashes and the state roots,
  and passes `_verify_final_manifest` with the real `/opt/homebrew/bin/claude` config.
* Real registered baseline (`baseline_registration_new_a40.json`) loads into a single-arm state
  and passes the launch baseline gate: `baseline-calibration-new-a40-20260914-run1`, validation BPB
  1.19989, the same baseline the Codex T arm started from.
* Claude CLI: 2.1.214, Anthropic-signed (team Q6L2SF6YDW); `claude auth status`: logged in via
  claude.ai, org "<redacted>", subscription `team`, first-party provider.

### 2.5 Codex remains functional, with one environmental gap
The Codex code path is unchanged. The image `autopareto-scientist-codex:0.153.0` is not present on
this Mac. Rebuilding from `study/contracts_2026/scientist_runtime.Dockerfile` will give a different
digest from the frozen `sha256:877dae34…`, so resuming the Codex campaign would need a manifest
amendment.

## 3. Run plan (single A40, strictly sequential: no concurrent jobs on the GPU)

| Step | GPU | Wall-clock | Gate before next step |
|---|---|---|---|
| R. Restore pod from `/workspace/PRESERVED_20260917T021347Z`, record GPU UUID, verify checkpoint and cache hashes | – | ~1 h | hashes match |
| D. Deploy the 6 amended files to `/opt/autopareto_trusted`, verify sha256 | – | 10 min | hashes = derived manifest |
| C. Claude canary (one call, Mac only) | – | 5 min | parser accepts; outside read refused |
| E0. Environment reproduction (2 runs) plus data-identity replay | 0.73 h | 0.73 h | \|ΔBPB\| ≤ 0.005, tokens equal, fixed-time updates in [669, 681]; else halt |
| E8. Self-terminating candidate smoke on the pod | <0.1 h | 10 min | classified scientific_failure, no checkpoint |
| E1. Crossover, 24 runs (refused unless `E0/e0_verdict.json` is PASS) | ~6.4 h | ~6.4 h | integrity gates per run |
| AC. Claude campaign, 25 fixed-time attempts | ~4.7 h busy | ~7–8 h | stops itself on any infrastructure fault |

Per-attempt figures come from the Codex T arm (22 attempts): 670 s mean A40 time
(300 s training, ~57 s validation, ~253 s bracketed inference benchmark) plus ~4 min of Scientist
time. E1 is 6.4 h rather than audit 08's 4.1 h: Phase-1 records show ~3.7 min of compile/startup and
4–8 min of milestone and held-out evaluation per run. E1 is ordered before the campaign because it
answers the main reviewer concern.

E0/E1 preparation (`evaluation/camera_ready_e0_e1_plan.json`, sha256 `806a6397…c829340`;
`scripts/run_camera_ready_e0_e1.py`; `scripts/analyze_camera_ready_e1.py`;
runbook `evaluation/CAMERA_READY_E0_E1.md`) reuses the Phase-1 launcher and evaluator unchanged and
writes only under `results/camera_ready_2026/`. Validate-only: 9 ok, 0 FAIL, 8 POD_ONLY (data cache,
held-out cache, CUDA/A40, `/workspace`, `uv.lock`). Exp 134 is byte-identical to the Phase-1 AP-Q
source (`55a7fd8b…`); the current transform reproduces the panel's derived sources byte for byte.

## 4. Cost

| Item | Estimate |
|---|---|
| A40 pod, ~16 h total at the audit's $0.44/h (confirm in the console) | ≈ $7 |
| Claude Opus 5.5 / xhigh on the Team seat: 50 calls; ~1M input tokens per attempt (~0.87M cached) and ~14k output, scaled from the Codex campaign | ~25M input (mostly cache reads), ~0.35M output; billed to the seat, not per token |
| Risk | A seat usage limit mid-campaign halts the controller at that position (`reconciliation_required`, before remote execution). No result is lost; the existing `amend_control_plane_manifest.py` requeues it. |

## 5. Validity findings that affect the paper (details: `11_VALIDITY_VERIFICATION.md`)

* Confirmed: one BPB implementation across all eras (summed nats ÷ (ln 2 × summed decoded bytes),
  BOS/special tokens masked); all 21 Phase-1 runs consistent and their 21 `final.pt` files
  hash-verified; −0.006680 / +0.045073 / +0.042034 reproduce; exp 68/234 substitutions are
  byte-identical sources and feed no paper number.
* The 3.2159 runs: 8 unique runs (19 signature files) show the re-executed-body signature; the
  bytecode of a 2026-09-05 `candidate_process.py` has the same loop.
* New: held-out BPB is 0.050–0.072 above validation with a recipe-dependent gap, so the two are
  not interchangeable; seed-42 rollback deleted 579 event rows (not 242); G3 is not
  schedule-confounded (only G4, 19 attempts); `prepare.py` downloads training shards from the
  unpinned `main` branch (the manuscript now says so; Pair-A pins all 13 shards by SHA-256);
  67 discovery rows record a wrong `git_commit`; the 11→4→3 cut to 3 has no recorded rule.
* 99 flags in `supporting/historical_validity_flags.json`; no historical artifact was altered.

## 6. Exact commands (in order; nothing below has been run)

`<IP>`/`<PORT>` = the pod's "SSH over exposed TCP" endpoint (needed for tar/rsync; the
`ssh.runpod.io` proxy does not carry file transfers). `<PROXY_HOST>` = the proxy target the
controller uses. Neither goes into a committed file.

**R. Restore and verify (pod)**
```bash
ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<IP>
cat /workspace/RESUME.md                      # follow it; rsync read-only trees with --chmod=Du+w
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
sha256sum /var/lib/autopareto/pair_a_official_20260916_continuation/baseline_checkpoints/t_result.checkpoint.pt  # 0ccea0c4…
ls -d /workspace/autopareto ~/.cache/autoresearch   # Phase-1 environment for E0/E1
```

**D. Deploy the six amended files (Mac → pod)**
```bash
cd /Users/Rami/Documents/dev/autopareto-camera-ready
tar -cf - study/contracts_2026/{bridge,campaign_controller,candidate_process,isolation,scientist_config,train_launcher}.py | ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<IP> 'tar -xf - -C /opt/autopareto_trusted && cd /opt/autopareto_trusted && sha256sum study/contracts_2026/{bridge,campaign_controller,candidate_process,isolation,scientist_config,train_launcher}.py'
```
Expected: the `after` hashes in the derived manifest's `derivation.changed_immutable_paths`
(at d4a200d: bdc4efbb… bridge, 0f324296… campaign_controller, 41fc8a96… candidate_process,
5b38ed29… isolation, 57166b0d… scientist_config, 0bf6da10… train_launcher).

**M. Derive and freeze the Claude manifest (Mac)**
```bash
cd /Users/Rami/Documents/dev/autopareto-camera-ready && PYTHONPATH=$PWD /opt/homebrew/bin/python3 -B -m study.contracts_2026.tools.derive_claude_campaign_manifest --base /Users/Rami/Documents/dev/autopareto/runs/pair-a-official-20260914-new-a40-continuation-20260916-r2/provenance/final_pair_a_continuation_manifest_20260917_posttraining_failure_receipt_fix.json --output /Users/Rami/Documents/dev/autopareto/runs/pair-a-claude-opus55-t-seed3/provenance/claude_t_manifest.json --state-root /Users/Rami/Documents/dev/autopareto/runs/pair-a-claude-opus55-t-seed3 --trusted-state-root /var/lib/autopareto/pair_a_claude_opus55_t_seed3 --scientist-bin /opt/homebrew/bin/claude --model claude-opus-5-5 --reasoning xhigh --a40-ssh-host <PROXY_HOST> --a40-gpu-uuid <GPU_UUID>
```

**C. Claude canary (Mac; one model call on the Team seat)**
```bash
cd /Users/Rami/Documents/dev/autopareto-camera-ready && PYTHONPATH=$PWD /opt/homebrew/bin/python3 -B -m study.contracts_2026.tools.claude_canary --scientist-bin /opt/homebrew/bin/claude --output /Users/Rami/Documents/dev/autopareto/runs/claude_canary_stream.jsonl --execute
```
Pass = exit 0: parser accepted the real stream, outside read refused, correct parent id returned.

**E0 / E1 (pod)** — full setup in `evaluation/CAMERA_READY_E0_E1.md` §3 (rsync of the worktree to
`/workspace/autopareto-camera-ready`, `uv.lock` from `autopareto/research-infrastructure`, `uv sync --frozen`):
```bash
cd /workspace/autopareto-camera-ready && unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
uv run --locked python scripts/run_camera_ready_e0_e1.py --expect-plan-sha256 806a63970258627092d929d4602e54e36a2412e0656a63298e17307c6c829340
uv run --locked python scripts/run_camera_ready_e0_e1.py --stage E0 --execute --expect-plan-sha256 806a63970258627092d929d4602e54e36a2412e0656a63298e17307c6c829340
uv run --locked python scripts/run_camera_ready_e0_e1.py --stage E1 --execute --expect-plan-sha256 806a63970258627092d929d4602e54e36a2412e0656a63298e17307c6c829340
```

**AC. Claude campaign (Mac; resumable by re-running the identical command)**
```bash
cd /Users/Rami/Documents/dev/autopareto-camera-ready && PYTHONPATH=$PWD nohup caffeinate -i /opt/homebrew/bin/python3 -B -m study.contracts_2026.campaign_controller run --execute --contracts T --agent-backend claude --state-root /Users/Rami/Documents/dev/autopareto/runs/pair-a-claude-opus55-t-seed3 --baseline-record /Users/Rami/Documents/dev/autopareto/runs/pair-a-official-20260914-new-a40/provenance/baseline_registration_new_a40.json --manifest /Users/Rami/Documents/dev/autopareto/runs/pair-a-claude-opus55-t-seed3/provenance/claude_t_manifest.json --manifest-sha256 <SHA_FROM_STEP_M> --scientist-bin /opt/homebrew/bin/claude --model claude-opus-5-5 --reasoning xhigh --scientist-timeout-seconds 600 --a40-host <PROXY_HOST> --a40-identity /Users/Rami/.ssh/id_ed25519 --a40-repository /opt/autopareto_trusted --a40-python /opt/autopareto_runtime/bin/python --a40-study-venv /opt/autopareto_runtime --a40-cache /var/cache/autopareto/pair_a_registered_cache --a40-state-root /var/lib/autopareto/pair_a_claude_opus55_t_seed3 --a40-paired-baseline-t-checkpoint /var/lib/autopareto/pair_a_official_20260916_continuation/baseline_checkpoints/t_result.checkpoint.pt --a40-paired-baseline-n-checkpoint /var/lib/autopareto/pair_a_official_20260916_continuation/baseline_checkpoints/n_result.checkpoint.pt >> /Users/Rami/Documents/dev/autopareto/runs/pair-a-claude-opus55-t-seed3.controller.log 2>&1 &
```
Codex stays selectable with `--agent-backend codex` (the default) and its usual flags.

## 7. Open items

* Team-seat authorization for this research use is not established by `claude auth status`.
* The real stream format and outside-context refusal are unverified until the canary.
* Snapshot layout and whether `/workspace/autopareto` and `~/.cache/autoresearch` survived are
  unknown until the pod starts. If the data cache must be re-downloaded from `main`, E0's
  shard-hash and token-stream gate decides whether it is the same data.
* E0/E1 pre-registration choices to confirm (each changes the plan hash): held-out BPB as the
  primary E1 metric; E0 fixed-time update tolerance [669, 681]; data-identity replay in E0.
* Phase-1 launcher/evaluator files differ from the Phase-1 preflight hashes (audit M13); E0 run 1
  is the real test. `kernels-community/flash-attn3` revision is recorded but not gated.
* Backup is same-disk only.

## 8. Independent pre-launch review (2026-11-02) and dispositions

| Finding | Severity | Disposition |
|---|---|---|
| H1 Launcher overwrote the worker's `candidate` failure receipt with `unknown`, so any training-time candidate failure (including the new early-return failure) halted the campaign | High | Fixed in `train_launcher.py` (worker receipt is authoritative). End-to-end launcher test fails with the fix disabled. No executed Codex position took this path (both arms: zero training-time failures). |
| H2 Step/update check rejected legitimately trained candidates (micro-step counters, counters carried across warm-up) | High | Check removed; redundant once no checkpoint can be saved without the protected stop. Test now asserts such a candidate completes. |
| H3 Another model could produce part of a run and still be accepted | High | `CLAUDE_CODE_NO_MODEL_FALLBACK=1`; every assistant message must come from the configured model; no other model may have output tokens; CLI version must equal the frozen one; structured output required. |
| M4 Claude payloads reused the Codex arm's remote paths (`pair-a-position-NNN-…`) on the restored pod | Medium (evidence overwrite) | Remote namespace frozen in the derived manifest (`claude-opus55`); dry run asserts it. |
| M5 Single-arm mode not enforced by the manifest | Medium | Manifest's frozen arms must equal `--contracts`. |
| M3 No OS sandbox for the Claude Scientist | Medium | Canary extended: absolute-path Read, Glob, Grep, a listing of the real `runs/` records, a write outside the context, and a real edit with a stray-file check. To be disclosed in the paper. |
| L1/L2/L5 Free-text fallback, `splitlines`, missing forbidden flags | Low | Fixed. |
| M1 Commit not idempotent across a crash inside the commit window | Medium, pre-existing | Not changed (shared with Codex; fails closed). |
| M2 Parent eligibility read from the registry, not the current frontier | Medium, pre-existing | Not changed, so the search policy stays identical to the Codex arm. Disclosed. |
| L4 Agent process survives SIGINT/SIGTERM of the controller | Low | Operational: do not interrupt during an agent call; if interrupted, kill the agent before resuming. |
| L6 `preflight` does not understand single-arm states | Low | Not used for this run. |
| E0/E1 M1/M2: no per-run timeout; interrupted runs can leave training alive | Medium | Operational, to keep plan sha `806a6397…` unchanged: expected durations are known and runs are monitored; before any `--resume`, check `nvidia-smi --query-gpu=memory.used` and `pgrep -f launcher`. |

Tests after fixes: 174 run, 1 error (`test_timeout_terminates_a_real_disposable_docker_container`: the
Codex Docker image is not present on this Mac; Codex-only). The stage-shared value-embedding adapter
test was a stale fixture (missing rotary buffers), now fixed; the protected adapter is unchanged.

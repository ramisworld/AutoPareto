# 15 — Opus campaign readiness (2026-11-04): all launch checks pass; not launched

Campaign: `pair-a-claude-opus55-t-seed3`. It runs 25 fixed-time (Contract T) attempts, seed 3,
Claude Opus 5.5 at xhigh, on the fresh A40. Nothing below has started a campaign attempt.

## 1. The monitoring fix (E0's missed completion)
- **Cause.** E0's watcher waited with `while pgrep -f "run_camera_ready_e0_e1.py --stage E0"`. Its
  own `bash -c` command line contained that pattern, so the loop never ended. E0 finished at
  08:06 UTC, and the pod sat idle until 09:38 UTC.
- **Fix.** `scripts/campaign_supervisor.py` (commits 97b3662, e150692) wraps the controller and
  never searches process names.

| Need | How |
|---|---|
| Reliable exit detection | The supervisor is the controller's parent and gets its exit from `waitpid`. An exclusive `flock` on `<supervisor-dir>/lock` is inherited by the controller, so the lock is free only when both are gone. `status` and `watch` test the lock, not process names. |
| Persistent logs | `controller.log` is appended, never truncated. `events.jsonl` is appended and fsync'd: launch, every position transition, every Scientist call with seat usage, alerts, and the exit with its classification and log tail. `status.json` is rewritten every 15 s. |
| Progress tracking | One event per attempt (BPB, Pareto decision, inference status). `status` gives the counts and the current position; `watch` prints one line per attempt and exits on the exit record. |
| Failure alerts | macOS notifications on start, exit and every alert: reconciliation required, stall (> 45 min without activity), seat usage ≥ 80%, abnormal exit. `watch` raises `ALERT LOST` if both processes vanish without an exit record, and an alert if the supervisor dies while its controller runs on. |
| Safe resume | The supervisor never edits state, retries or restarts. Resuming is re-running the identical command. A second supervisor or controller is refused while the lock is held, or while any process from an earlier controller group, or any recorded Scientist call, is still alive. On exit it terminates a Scientist call left behind by an interrupted controller (audit 12 §8, finding L4). |
| No-overage margin | Once any seat window reaches 80%, it stops the controller at the next safe point, while the current attempt trains on the A40. That stop is resumable with no loss and no retraining. The controller's own halt at 95% is checked only after a call ends, and the CLI reports usage once, mid-call. |

**Tests.** `scripts/tests/test_campaign_supervisor.py` has 8 tests, all passing in 60 s. They drive the
real controller loop offline: the fake CLI, plus a mock A40 that keeps durable receipts on disk.
They cover:
- a full run with a concurrent `watch`;
- a refused second launch;
- SIGTERM during a remote wait, then resume;
- SIGKILL of the controller during a remote wait, then resume;
- a stop during a Scientist call (the position pauses, the agent is killed, and it is never re-run);
- a killed supervisor whose controller keeps the lock, then reported as lost;
- a seat-usage pause at attempt 3's remote launch, then resume;
- refusal when a leftover Scientist process is alive.

Every resume ends with 25 unique attempts, each trained exactly once, and a history of attempts
1–25. Removing the agent cleanup, or the lock inheritance, makes a test fail.

**How a resume behaves.** It never loses or repeats an attempt, by stop point:

| Stop point | State | Re-running the identical command |
|---|---|---|
| During the remote run (`launch_requested`) | `running` | Waits for the durable receipt, then commits. The remote launch is idempotent by invocation ID, so there is no retraining. |
| During a Scientist call, or before remote launch | `reconciliation_required` | Refuses (`campaign is paused`). The position is requeued only with `amend_control_plane_manifest.py`, after review. |
| Seat-usage pause | `running`, launched | As the first row, after the usage window resets. |

## 2. Launch checks

| Check | Result |
|---|---|
| G4 comparator decision | Frozen in commit 853368d at 09:57 UTC, before any Opus state existed: `evaluation/CLAUDE_CAMPAIGN_CONFIRMATION_G4_COMPARATOR.md` (sha256 `f5a59d52…`). Attempt 14, source `b612b6e8…` (byte copy in `evaluation/confirmation_sources/`). Same 12-run design, 18 runs in all. The pre-registration `adf8ce71…` is unchanged. |
| Frozen protocols | E0/E1 v1 `806a6397…` and amendment 1 `89f29015…` are unchanged. The confirmation pre-registration `adf8ce71…` and E1X `a1f6365f…` are unchanged. Baseline source `b2fb8788…`. |
| E0 evidence | Both sets are committed read-only and hash-identical to the pod (992b69d; `results/camera_ready_2026/README.md`). The original is HALT (`bd953480…`); amendment 1 is PASS (`517750d6…`). |
| Campaign manifest | `runs/pair-a-claude-opus55-t-seed3/provenance/claude_t_manifest.json`, sha256 **`c6e3ca0647a90371edc08ad68a190c3c13e21d23efd165f2e4cc9bc96c959fa6`**, read-only, untracked because it holds the SSH proxy host. It is derived from the last executed Codex manifest. Exactly the 6 expected harness files changed hash. The only replaced fields are: state roots, SSH host, GPU UUID, and the T/N paired-baseline checkpoints. Contracts, schedule, data, tokenizer, inference protocol, baseline registration and N\* are inherited unchanged. Status `FROZEN_PRE_STUDY_CLAUDE_T_ONLY_FRESH_0_OF_25`; active contracts `["T"]`; remote namespace `claude-opus55`. |
| Controller launch gates, read-only (`scripts/check_campaign_launch_gates.py`) | Passed over the controller's own proxy transport. The manifest verifies against its sha. Transport and state-root config match. All **122 trusted files** on the pod match. GPU **`GPU-4702993a-fd40-d985-292e-6398bd236183`**. **15 absolute files** match: the registered cache plus both paired-baseline checkpoints. The registered baseline `baseline-calibration-new-a40-20260914-run1`, BPB **1.19989**, passes the protected-metrics gate. The local state root holds only the manifest. |
| Baseline checkpoints (G2 attempt 2) | T `f7ca17c6…` (BPB 1.20110, \|Δ\| 0.0012 against the registered value); N `72d05e97…` (BPB 1.20004). Both are on the pod under `/var/lib/autopareto/pair_a_claude_opus55_baseline_attempt2/` and in the `/workspace` mirror. |
| Model identity | The CLI auto-updated to **2.1.318** today; the old canary covered 2.1.214. All 13 flags used are present. A new one-call canary on 2.1.318 **passed**: only `claude-opus-5-5`, `apiKeySource: none`, `acceptEdits`, five file tools plus StructuredOutput, no MCP; every outside read, list and write was refused. The manifest freezes 2.1.318, and every call must report that version. |
| Seat usage (canary, about 09:59 UTC) | 5-hour window 3% (resets 14:30 UTC); 7-day window 25% (resets 2026-11-06 04:00 UTC); overage not in use. |
| GPU environment | Idle: 0 MiB, no compute apps, no ECC errors. Driver 595.91.07. The remote state root `/var/lib/autopareto/pair_a_claude_opus55_t_seed3` does not exist yet. No candidate attempts are left over. 46 GB free on the container disk. |
| Supervisor wiring | It ran the real controller's no-side-effect `dry-run` with the real manifest under the same interpreter, PYTHONPATH and working directory. Log captured, exit detected. The controller reported no Scientist or A40 call. |
| Host | On AC power. `caffeinate -i -s` keeps the Mac awake for the supervisor's lifetime. The Mac must stay online and its lid open: the agent runs on the Mac. |

## 3. Launch command (Mac; `<PROXY_HOST>` is the pod's `…@ssh.runpod.io` target, never committed)
```bash
cd /Users/Rami/Documents/dev/autopareto-camera-ready && PYTHONPATH=$PWD nohup caffeinate -i -s /opt/homebrew/bin/python3 -B scripts/campaign_supervisor.py run --supervisor-dir /Users/Rami/Documents/dev/autopareto/runs/pair-a-claude-opus55-t-seed3.supervisor --state-root /Users/Rami/Documents/dev/autopareto/runs/pair-a-claude-opus55-t-seed3 --notify --label opus55-T -- /opt/homebrew/bin/python3 -B -m study.contracts_2026.campaign_controller run --execute --contracts T --agent-backend claude --state-root /Users/Rami/Documents/dev/autopareto/runs/pair-a-claude-opus55-t-seed3 --baseline-record /Users/Rami/Documents/dev/autopareto/runs/pair-a-official-20260914-new-a40/provenance/baseline_registration_new_a40.json --manifest /Users/Rami/Documents/dev/autopareto/runs/pair-a-claude-opus55-t-seed3/provenance/claude_t_manifest.json --manifest-sha256 c6e3ca0647a90371edc08ad68a190c3c13e21d23efd165f2e4cc9bc96c959fa6 --scientist-bin /opt/homebrew/bin/claude --model claude-opus-5-5 --reasoning xhigh --scientist-timeout-seconds 600 --a40-host <PROXY_HOST> --a40-identity /Users/Rami/.ssh/id_ed25519 --a40-repository /opt/autopareto_trusted --a40-python /opt/autopareto_runtime/bin/python --a40-study-venv /opt/autopareto_runtime --a40-cache /var/cache/autopareto/pair_a_registered_cache --a40-state-root /var/lib/autopareto/pair_a_claude_opus55_t_seed3 --a40-paired-baseline-t-checkpoint /var/lib/autopareto/pair_a_claude_opus55_baseline_attempt2/nonstudy_tests/t_smoke_artifacts/result.checkpoint.pt --a40-paired-baseline-n-checkpoint /var/lib/autopareto/pair_a_claude_opus55_baseline_attempt2/nonstudy_tests/n_smoke_artifacts/result.checkpoint.pt > /Users/Rami/Documents/dev/autopareto/runs/pair-a-claude-opus55-t-seed3.supervisor.nohup.log 2>&1 &
```
- **Check progress:** `python3 scripts/campaign_supervisor.py status --supervisor-dir … --state-root …`.
- **Stream progress:** `… watch …`, which exits on the exit record.
- **Stop:** `kill -TERM <supervisor pid>` (the pid is in `status.json`). Avoid stopping during a Scientist call, because that pauses the position for reconciliation. Stopping during a remote run is free.
- **Resume:** the identical command.

## 4. Residual risks (each fails closed)
- **CLI auto-update mid-campaign.** Auto-update is enabled for the global CLI, and it updated today.
  - An update makes the next call report a version other than the frozen 2.1.318, so the call is
    rejected and the position halts for reconciliation.
  - Resuming would then need an amendment.
  - Prevention needs the PI's OK for a persistent setting: `DISABLE_AUTOUPDATER=1` in
    `~/.claude/settings.json` `env`, for the campaign's duration. Until then, do not start new
    Claude Code sessions while the campaign runs.
- **Scientist timeout.** The timeout is 600 s, frozen in the scientist configuration and the same as
  Codex. A slower xhigh call is killed and halts the position for reconciliation. The canary took 18 s.
- **Seat windows.** Per-call usage is unknown until attempt 1; the first `scientist_call_finished`
  events will give it. The 80% pause bounds the risk.
- **Mac sleep or network loss.** A Scientist call fails (position halts for reconciliation), or the
  remote wait breaks (resumable). Stalls alert after 45 min.
- **Pod stop.** This wipes the container disk, including the trusted state and checkpoints. Mirror
  with `fresh_a40_bootstrap.sh mirror` after the campaign; a stop mid-campaign means a
  reconciliation and an amendment.

## 5. Cost
- **A40:** about 7–8 h at $0.44/h ≈ $3.5. The pod has been up since 02:24 UTC, about $3.4 so far.
- **Claude:** 50 calls on the Team seat, plus one canary call today (list-price equivalent $0.10).
  It is billed to the seat, never to overage.

## 6. Launch (PI approval: "Yes, both", 2026-11-04)
- **Auto-update off.** `DISABLE_AUTOUPDATER=1` was added to `~/.claude/settings.json` `env`. Remove it after the campaign.
- **What was updating the CLI.**
  - The source was an interactive global-CLI session of the PI's on another project (`claude --name stack-supervisor`, started 04:03 UTC). Its updater ran every 30 min (08:31, 09:01, 09:31 and 10:01 UTC).
  - Each run did `npm view …@latest` followed by `npm install -g @anthropic-ai/claude-code@2.1.318`. That reinstalls even the same version, leaving the binary missing or half-written for about 4 s.
  - After the setting change, no npm activity occurred at 10:31 UTC, and the CLI stayed at 2.1.318.
- **Launch.** 10:33:59 UTC, with the command in §3. Supervisor pid 34318, controller pid 34324.
- **Gates.** All passed at launch. The controller created its state: 25 active positions, 25 inactive.
- **First call.** Attempt 1 (global position 2) started its first agent call at 10:34:12 UTC.

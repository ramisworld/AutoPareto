# 25 — Historical agent provenance from the author's Codex session logs (2026-11-09)

**Scope.** This document identifies which agent, model and reasoning effort proposed each historical attempt (July–August 2026). It also records what happened in the seed-0 A16→A17 gap. No experiment was run.

**Inputs (read-only).**
- The experiment registry `results/results.db` in the evidence tree.
- The author's local Codex session logs, `~/.codex/sessions/2026/07–08/*.jsonl.zst`.

The session logs contain access credentials (the pod's SSH login appears in the opening prompt of the seed-0 session). They therefore stay private, and nothing from them is quoted here beyond metadata and short paraphrases.

**Reproduce.**
- Script: `scripts/audit_historical_codex_sessions.py`.
- Output: `supporting/historical_codex_attribution.json`. It is metadata only: session file names, timestamps, model and effort per attempt.

## 1. Agent identity

**Tool.** Every session was recorded with originator **"Codex Desktop"**, i.e. the Codex desktop app, working in `/Users/Rami/Documents/dev/autopareto`. The Codex core versions were:
- 0.145.0-alpha.27/30 (AutoResearch);
- 0.146.0-alpha.3.1 (seed 42);
- 0.146.0-alpha.9.2 (seeds 0 and 1).

The author recalls using Codex CLI. The records show the desktop app, which runs the same Codex agent.

**How the agent worked.** It operated the protected runner on the pod over SSH: it wrote queued sources, launched runs in tmux, polled them, and called the runner's campaign-view and record-decision commands.

**Attribution method.** Each attempt is attributed to the latest tool call naming its campaign within 20 minutes before its `started_at`, preferring calls that launch work. Attempts with no such call fall back to a 90-minute window and are labelled "probable".

| Campaign | Attempts | Model / effort |
|---|---|---|
| AutoResearch, 50 attempts (exp 14–63) | 47 attributed + 3 probable | gpt-5.6-sol / high |
| AutoResearch follow-up, 5 attempts (exp 64–68; AR-ref = exp 68) | 5 attributed | gpt-5.6-sol / high |
| AutoPareto seed 42, A6–A25 (exp 93–112) | 20 attributed (Codex) | gpt-5.6-sol / high |
| AutoPareto seed 42, A1–A5 (exp 84–87; exp 92 replays A5) | Cursor's agent, verified from Cursor's local store (see below) | gpt-5.6-sol / reasoning **high** |
| AutoPareto seed 0 (exp 113–137; AP-Q = exp 134) | 25 attributed, one session | gpt-5.6-sol / high |
| AutoPareto seed 1 (exp 138–162) | 25 attributed, one session | gpt-5.6-sol / high |

- **The registry label agrees with the logs for every reported attempt.** The registry's `coding_model_name='gpt-5.6-sol'` and `reasoning_level='high'` are the runner's defaults.

**Seed-42 A1–A5 were proposed by Cursor's agent, verified on 2026-11-09.**
- **Why Cursor.** On 2026-07-28 Codex usage was nearly exhausted (the author's message at 02:55 UTC says 1% left). The author therefore ran seed 42 from Cursor.
- **Source.** Cursor's local store records the settings of each conversation, and every terminal command it executed.
- **Who launched A1–A4.** The conversation "AutoPareto seed-42 instructions" (created 03:08 UTC; model gpt-5.6-sol, parameter `reasoning: high`) executed:
  - the run commands for A1–A4 at 03:29, 03:47, 03:59 and 04:10, matching the registry starts of exp 84–87;
  - their record-decision commands;
  - the run command for the original A5 at 04:23.
- **Rolled-back attempts.** A second conversation, "Complete seed-42 campaign" (created 04:24; gpt-5.6-sol, `reasoning: medium`), ran attempts 6–8 and recorded exp 88–90. These are the records the seed-42 rollback removed (ids 88–90 absent; doc 06, P2-06 (i)). They are not in the paper's data.
- **No other runs.** A grok-4.5 conversation on the same morning executed no run or decision command; it only analysed.
- **The A5 replay.** On 2026-07-30 a Codex thread set to gpt-5.6-sol/medium replayed A5 on a new pod as exp 92, at the author's instruction to "copy the exact same changes" of the original. Its edit:
  - sets `KV_HEAD_RATIO` 1→2 as the original A5 did;
  - restores the parent's batch and learning rate;
  - is recorded in the registry as "keep TOTAL_BATCH_SIZE at 2**17 from parent 87 and set KV_HEAD_RATIO from 1 to 2".

  The proposal therefore came from the high-effort conversation. Only the replay was operated, and its decision recorded, from the medium thread.
- **Reproduce.** `scripts/audit_historical_codex_sessions.py` (`cursor_seed42` in the JSON output).

**Security.** The local Codex logs contain the pod's SSH login and a RunPod API key in plain text. The key was flagged to the author for rotation on 2026-11-09. Neither is reproduced here.
- **The classical-optimizer campaigns are out of scope.** TPE, CMA-ES, MOTPE, random search and NSGA-II were operated from Codex threads mostly set to low effort. The optimizers, not the agent, chose those parameters.

**Summary for the paper.** Every reported historical attempt was proposed by gpt-5.6-sol at high reasoning effort. Seeds 0 and 1, seed-42 A6–A25 and both AutoResearch campaigns used the Codex desktop app; seed-42 A1–A5 used Cursor's agent.

**Transcripts exist.** Session logs survive for every historical agent campaign. The paper's earlier statement that "no transcripts survive" was wrong; the correction is in the P2-02 and P2-13 drafts and in the Disclosure statement.

## 2. The seed-0 A16→A17 gap (internal record; not in the paper by the author's decision)

**Timing.** A16 (exp 128) finished at 14:52:00 UTC on 2026-08-03. A17 (exp 129) started at 22:21:52 UTC, 7.5 h later. The author's timezone is Pacific/Auckland, so this was 02:52–10:21 local time.

**Timeline from the seed-0 session** (`rollout-2026-08-03T19-38-18-…`, gpt-5.6-sol/high throughout the campaign):

| Time (UTC) | Event |
|---|---|
| 07:41 | One turn starts: run the registered campaign per `programs/program_autopareto.md` |
| 14:40:56 | Routine "sleep 55 s, then check the pod" call while A16 runs |
| 15:23 → 18:44 | No activity (3.35 h). The poll's result was logged at 15:23 with an 11 s wall time; the next event is a usage reading at 18:43:59 |
| 19:14–19:45 | The agent resumes, finds A16 completed, and records it as dominated by its parent (exp 125) |
| 19:45 → 20:16 | No further action |
| 20:16:55 | The turn is interrupted deliberately (Codex records "interrupted on purpose") |
| 22:20:53 | The author starts a new turn: "Continue where you left off. First inspect the registered campaign state; do not duplicate any attempt. Complete the remaining attempts sequentially…" |
| 22:21:52 | A17 starts. A17–A25, including AP-Q (A22), follow |

**What was ruled out.**
- *Usage limits:* the Codex weekly usage was 43–45% throughout, with no limit reached and no error events.
- *RunPod interruption:* A16 completed normally, the pod answered SSH at 19:14, and at 22:21 the registry showed no running job and no duplicate.

**Unexplained.** The silent interval stays unexplained. It is consistent with the Mac sleeping overnight, but macOS power logs only go back to 2026-11-02, so this cannot be verified.

**Scientific effect.** None:
- the program, evaluator, protected files and registry were unchanged;
- no attempt was duplicated, edited or re-run;
- the restart instruction carries no scientific steering.

**Paper.** On 2026-11-09 the author decided this is operational maintenance and leaves it out of the paper. P2-06's former sentence "The AP-Q lineage (seed 0) has no recorded intervention" was narrowed in the draft to "no recorded scientific intervention (program, evaluator and records unchanged)", which remains accurate.

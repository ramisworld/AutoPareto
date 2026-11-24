# Hostile reproducibility audit

Audit date: 2026-09-02
No training, benchmark, campaign, or N_STAR operation was run.

## Threat model

Assume a reviewer distrusts all prose and tests whether each headline result can be
traced to raw or structured evidence, whether regimes were mixed, whether failed
measurements were hidden, and whether the current pre-campaign instrument was
mistaken for a scientific result.

## Authority and checks

| Test | Evidence checked | Result |
|---|---|---|
| Old ledger is preserved before rewrite | `research_ledger_pre_rewrite_20260902.md`, capture JSON | PASS; byte-for-byte SHA match |
| Git/worktree is not cleaned | capture record and status | PASS; pre-existing dirty evidence preserved |
| Frozen parent and N_STAR hashes are unchanged | frozen manifests | PASS; `ab7c7923...bb3e`, `7212564a...4d4f` |
| Superseded invalid freeze retained | `n_star_freeze_20260902.json` | PASS; invalid artifact indexed |
| Historical/equal-token regimes separated | ledger, Phase 1 manifest, summaries | PASS |
| 1.81x/1.985x arithmetic reconciled | Phase 1 JSON and table | PASS; 1.985x is authoritative |
| ~307,982 ratio interpreted correctly | baseline row and historical source semantics | PASS; mixed ratio labelled misleading |
| Phase 1 exact-token reporting checked | 21 run summaries and per-run artifacts | PASS for reported counts; runtime stream hash gap retained |
| Runtime token-ID hash checked | preflight, token-stream audit, expected output | FAIL/MISSING; no claim of runtime hash verification |
| Checkpoint/source pairing checked | phase1 selection and checkpoint registry | PASS locally; weights remain private |
| Populated registry checked | recovered SQLite snapshot, read-only schema/counts | PASS for recovered snapshot; canonical local DB absent |
| Public CSV scope checked | `data/experiment_records.csv` row count | PASS; release snapshot explicitly not complete registry |
| Failed/unstable outcomes checked | SQLite statuses, inference summaries, logs | PASS; no fastest repetition promoted |
| Downstream evaluator drift checked | evaluator hashes and `.contiguous()` incident | PASS; provenance split retained |
| Agent attribution checked | `agent_actions`, trajectory audit | PASS as bounded attribution; full transcript UNKNOWN |
| Cost reproducibility checked | run counts and timing records | PARTIAL; dollars/energy/human time missing |
| Release/checklist checked | LICENSE, current official venue pages, registry | PARTIAL; release package not ready |
| Audit itself is non-scientific | task command history | PASS; no training/campaign launch |

## Missing/unsafe claims

The following claims are prohibited until the named evidence is restored:

- “The Phase 1 runtime token stream was hash verified.”
- “The public checkout contains the complete private experiment database.”
- “AutoPareto algorithmically beats TPE/MOTPE.”
- “AP292 is universally better than AutoResearch.”
- “AP292's gain is caused by depth alone.”
- “The result transfers to another GPU or scale.”
- “Exact agent reasoning/transcript is reconstructed.”
- “The repository contains complete energy, billing, or human-time accounting.”

## Reproduction verdict

An independent reviewer can reproduce the aggregate table arithmetic and inspect the
local source/checkpoint/data manifests. They cannot reproduce the full private
environment from the public checkout alone, verify the missing runtime token-ID
hash, inspect all raw per-example downstream traces, or recover the full AutoPareto
agent transcript. The evidence is therefore suitable for a bounded internal
pre-campaign freeze and a truthful research ledger, not yet a complete public
reproduction release.

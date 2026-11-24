# Research-ledger rewrite report

Date: 2026-09-02
Scope: documentation/provenance reconstruction only
Scientific training or campaign execution: none

Exact change class: documentation and new provenance registries only. No scientific
source, model recipe, optimizer, schedule, seed, tokenizer, dataset, benchmark,
inference adapter, sandbox, or N_STAR artifact was edited by this audit. The only
tracked code modification visible in the worktree, `protected/inference_adapter.py`,
predates this rewrite and was preserved.

## Ledger identity

| Version | SHA-256 | Bytes | Lines | Status |
|---|---|---:|---:|---|
| Preserved pre-rewrite ledger | `691ac32a97d323334effeae3fffbb459ef2993bec5459d853b3aad428337070a` | 56,532 | 940 | Immutable copy verified |
| Rewritten ledger | `f8213a916509a78f133a86b8663e962ff0d2b684dc14743d42c6a02dd2edcbed` | 31,549 | 510 | Reconciled ledger |

The old ledger remains at
`study/contracts_2026/provenance/research_ledger_pre_rewrite_20260902.md` and its
capture record is `research_ledger_pre_rewrite_record.json`. The rewrite is a
replacement with preserved historical provenance, not an append-only continuation.

## Structural changes

- Replaced duplicated/stale narrative sections with a single chronological ledger.
- Added explicit evidence priority and status vocabulary.
- Separated five-minute discovery, confirmation, downstream, deployment, equal-token
  Phase 1, and current pre-campaign T/N regimes.
- Added a complete failure/intervention ledger and non-results section.
- Added metric definitions and timing-boundary semantics.
- Added explicit human/agent attribution limits and no hidden-chain-of-thought claim.
- Added claim, hypothesis, source, artifact, data, tokenizer, checkpoint, experiment,
  compute, release, and venue registries.
- Added reproduction commands, release index, venue checklist, and hostile-audit
  checklist.

## Corrections made

1. The 1.81x AP292 decode prose claim is corrected to the table-derived
   `383.021825913658 / 192.93322787922392 = 1.985x`.
2. The historical `307,981.863 tokens/s` is labelled a mixed all-token/post-warm-up
   ratio; the correct post-warm-up rate is `288,841.748 tokens/s`.
3. The invalid ~45.1M-token protected calibration is retained as superseded evidence;
   it is not treated as valid N_STAR.
4. The exact historical step semantics are stated verbatim: “Steps 0–10 are
   scientifically executed but excluded from the historical steady-state timing
   boundary. Steady-state timing begins at step index 11.”
5. Runtime token-stream hash verification is not claimed: the preflight says it was
   not observed and the expected JSON is absent.
6. The populated recovered SQLite snapshot (318 experiments) is distinguished from
   the 284-row public CSV, the older 297-row prose, and the zero-byte local database.
7. Historical and equal-token downstream evaluator hashes/results are not merged.

## Status summary

| Area | Status | Evidence or missing item |
|---|---|---|
| Historical ledger preservation | PASS | Immutable copy and hash record |
| Source and candidate provenance | SUBSTANTIALLY VERIFIED LOCALLY | Candidate snapshots and source hashes present; some historical Git-only files are not in current checkout |
| Data/tokenizer static hashes | PASS | `data_manifest.json`, aggregate hash `1211827090aa7e669ec8bcd965ae1423dc90697f0e4928d0fce707cbc36c3f93` |
| Runtime token-ID sequence hash | UNRESOLVED | `results/phase1/token_stream_verification.json` missing |
| Checkpoint registry | PASS LOCALLY | 21 source/checkpoint pairs indexed |
| Experiment registry | PASS WITH PORTABILITY LIMIT | Populated recovered SQLite snapshot indexed; canonical `results/results.db` absent |
| Metrics and analysis | PASS WITH BOUNDS | 21-run table reconciled; n=3 and one-A40 limits explicit |
| Autonomous attribution | PARTIAL | Persisted hypotheses/lineage; full AutoPareto transcript unavailable |
| Compute/cost | PARTIAL | Nominal slots and run counts; dollars/energy/human time unavailable |
| Licenses/release | INCOMPLETE | LICENSE present; third-party data/weights and private evidence release review open |
| NeurIPS/AutoMLR readiness | PREPARATION ONLY | Checklist mapped; final paper/package not ready |
| Current study status | PRE-CAMPAIGN | Pair A/25×25 not launched; N_STAR unchanged |

## Output files

- `docs/RESEARCH_LEDGER.md`
- `study/contracts_2026/provenance/source_registry.json`
- `study/contracts_2026/provenance/data_manifest.json` (existing canonical manifest; unchanged)
- `study/contracts_2026/provenance/tokenizer_manifest.json`
- `study/contracts_2026/provenance/model_checkpoint_registry.json`
- `study/contracts_2026/provenance/experiment_manifest_index.json`
- `study/contracts_2026/provenance/claim_source_artifact_map.json`
- `study/contracts_2026/provenance/compute_ledger.json`
- `study/contracts_2026/provenance/reproducibility_audit.md`
- `study/contracts_2026/provenance/release_artifact_index.md`
- `study/contracts_2026/provenance/neurips_checklist_readiness.md`
- `study/contracts_2026/provenance/reproduction_commands.md`
- `study/contracts_2026/provenance/sources.bib`

## Decision

The research ledger is ready for a pre-campaign freeze because it now reports the
strongest claims, limitations, conflicts, missing artifacts, and exact provenance
without using unsupported evidence. The public/paper release is not ready until the
listed data, runtime-hash, raw-evidence, license, and final-paper issues are resolved.

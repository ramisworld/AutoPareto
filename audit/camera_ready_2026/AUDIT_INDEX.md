# AutoPareto camera-ready audit — Index

- **Audit date:** 2026-10-28.
- **Repository:** `/Users/Rami/Documents/dev/autopareto`, branch `autopareto/professor-ready-2026` at `6bcb69e`, with uncommitted changes.
- **Mode:** read-only.
  - No experiment, GPU job, paid API call or cloud resource was launched.
  - No existing source, configuration, dataset, checkpoint, result or manuscript file was modified.
  - All SQLite access used `mode=ro&immutable=1`.
  - The only writes are inside `audit/camera_ready_2026/`.
- **Purpose:** establish the scientific ground truth before the 2026-11-13 camera-ready, and scope the follow-up "Do AI-discovered optimizations survive scaling?".

## Reports

| File | What it contains | Prepared by |
|---|---|---|
| [`01_PROJECT_INVENTORY.md`](01_PROJECT_INVENTORY.md) | Map of code, data stores, checkpoints, runs, paper versions and git history; missing and inaccessible evidence | Delegated inventory audit |
| [`02_EXPERIMENT_REGISTRY.csv`](02_EXPERIMENT_REGISTRY.csv) | **403 rows**: all 318 database experiments (setup, baselines, discovery, crashes, OOMs, confirmations, ablations, the 201M panel) and 85 unique Pair-A attempts across 5 launch lineages. 66 columns: provenance, lineage, recipe constants parsed from the frozen sources, budgets, metrics, status and selection status. `UNKNOWN` marks fields the sources do not support. | Lead auditor (`supporting/scripts/build_registry.py`) |
| [`03_CLAIMS_AND_EVIDENCE.md`](03_CLAIMS_AND_EVIDENCE.md) | Every paper claim recomputed from raw records, with verdicts and confidence; findings the data support but the paper omits; a 20-item discrepancy log | Lead auditor |
| [`04_REVIEWER_RESPONSE_MATRIX.md`](04_REVIEWER_RESPONSE_MATRIX.md) | Each of the 10 reviewer and area-chair concerns: the manuscript passage, what the data establish, unjustified wording, the minimal fix, and the experiment if needed | Lead auditor |
| [`05_STATISTICAL_AND_CAUSAL_AUDIT.md`](05_STATISTICAL_AND_CAUSAL_AUDIT.md) | Decomposition of the ranking reversal; seed and session variance; a natural replicate; selection optimism; unselected candidates; frontier stability; Pair-A validity | Lead auditor (`supporting/scripts/statistical_audit.py`) |
| [`06_AUTONOMOUS_SEARCH_AUDIT.md`](06_AUTONOMOUS_SEARCH_AUDIT.md) | Search procedure per era; Pair-A forensics, including the harness defect behind the 3.2159-BPB runs; dated human-intervention timeline; selection policy (11 → 4 → 3) | Delegated agent-forensics audit, spot-checked by the lead auditor |
| [`07_SCALING_FEASIBILITY.md`](07_SCALING_FEASIBILITY.md) | How the code parameterizes scale; existing transfer evidence; costs at 4×/16×/64×; promotion blockers; a capped camera-ready pilot; a long-term study design | Delegated scaling audit, with a lead-auditor consistency note at the top |
| [`08_EXPERIMENT_PLAN.md`](08_EXPERIMENT_PLAN.md) | Ranked experiments (E0–E10) with designs, seeds, run counts, A40-hours, API tokens, analyses, outcome interpretations and stopping rules; redundant experiments; calendar to Nov 13 | Lead auditor |
| [`09_PAPER_REVISION_PLAN.md`](09_PAPER_REVISION_PLAN.md) | Section-by-section REPLACE/ADD/DELETE/MOVE instructions, draft text, mechanical corrections, page-budget check | Lead auditor |
| [`10_EXECUTIVE_BRIEF.md`](10_EXECUTIVE_BRIEF.md) | A standalone summary for the advising model and the team | Lead auditor |

## Supporting artifacts (`supporting/`)

| File | Contents |
|---|---|
| `claims_recomputation.json` | Machine-readable recomputation of the paper's numbers: Table 1, headline ratios, archive bests, fixed-time confirmations, baseline seed effects, ablation decomposition, rank correlations, downstream means, BLiMP configurations, correlations, lineage |
| `statistical_audit.json` | Paired tests and CIs, reversal table, exposure response, seed variability, token-exposure variation, selection optimism, top-candidate gaps, frontier by seed, Pair-A lineage summary |
| `registry_build_report.json` | Registry sources, consistency between the main and snapshot DBs, anomalies (missing ids 88–90, duplicated training-run rows, runs left in `running`) |
| `REPRODUCIBILITY_AND_PROVENANCE.md` | Per-artifact provenance (figures, tables, macros), public-release diff, reproducibility classification, environment eras, 17 mismatches (M1–M17), ranked minimal additions |
| `scripts/` | The lead auditor's scripts: `build_registry.py`, `recompute_claims.py`, `statistical_audit.py`. The delegated audits' helper scripts are under `scripts/delegated/`; raw git extracts were excluded. |

**To reproduce the core numbers**, run from `audit/camera_ready_2026/`:
```
python3 -I supporting/scripts/build_registry.py /Users/Rami/Documents/dev/autopareto .
python3 -I supporting/scripts/recompute_claims.py /Users/Rami/Documents/dev/autopareto supporting/claims_recomputation.json
python3 -I supporting/scripts/statistical_audit.py /Users/Rami/Documents/dev/autopareto supporting/claims_recomputation.json supporting/statistical_audit.json
```

## Coverage

| Area | Covered | Notes |
|---|---|---|
| SQLite stores | 12 populated SQLite files in the working tree, plus 19 that exist only in git (`01` §B). Analyses used the 2026-09-01 RunPod snapshot DB, which is a superset of `results/results.db` and byte-identical to the git copy; `results/phase1/downstream_200m.db` is 0 bytes. | Across 23 results snapshots, no row was deleted; ids 88–90 were removed before any snapshot captured them. 242 events were removed by the orphan-event cleanup. |
| Pair-A runs | All namespaces under `runs/` (history, position and manifest files; agent transcripts) | 5 launch lineages plus non-study validations |
| Code and harness | `train.py` family, frozen candidates, `study/contracts_2026/*`, evaluators, analysis pipeline | The harness that executed the panel differs from the recorded commit (REPRO M13) |
| Git | All local and remote branches, including `autopareto/research-infrastructure` (93 historical commits) | Pair-A code, `runs/`, `paper/` and `analysis/` are untracked on the current branch |
| Manuscript | `paper/main.tex`, generated macros and tables, the submitted PDF, backups and Overleaf zips | The submitted abstract matches only `paper_backups/automlr_overleaf_reconstructed/` (REPRO M1) |
| Statistics | Recomputed offline; no new training | n = 3 per cell limits resolution to about 0.005–0.012 BPB |

## Most important missing or inaccessible evidence

1. **RunPod persistent-volume snapshot** `/workspace/PRESERVED_20260917T021347Z`. The pod is stopped and the snapshot is remote; it was not accessed. It holds the dataset cache, tokenizer, executed Pair-A harness files and milestone checkpoints. No repository file mentions it; it is known only from the project memory note. All 21 panel `final.pt` checkpoints are recoverable locally from `.git/lfs` (`01` §C.3). The discovery checkpoints of AR-ref (exp 68) and MOTPE (exp 234) are missing.
1a. **The local-only archive branch** `autopareto/research-infrastructure` (877 files absent from the working tree) is the only copy of the archive DBs and the phase-1 run data. Back it up before any cleanup (`01` §0 item 1, §H).
2. **Historical agent transcripts and identity** for the 75 AutoPareto attempts and AR50. None exist. The model label is a runner default (06 A.5).
3. **The three N* calibration runs.** Described in `docs/RESEARCH_LEDGER.md` (lines 640–669) but not stored as data files.
4. **Per-item downstream outputs.** No item-level paired CIs are possible.
5. **Seed-42 rollback records** (exp 88–90 and the original A5–A8). Only live-event files survive (06 A.2).
6. **The OpenReview reviews.** Not machine-readable here; the team's list of concerns was used.
7. **A pinned training-data revision.** Training read the dataset's `main` branch (REPRO M14).

See `01_PROJECT_INVENTORY.md` §I and §K for the full inventory of missing and inaccessible items.

## Corrections to earlier, unaudited statements

These were made in an earlier advisory session and are flagged in the team's brief as "unverified leads".
- **"The Pair-A runs have 22 attempts per arm; the paper reports only 4."** The 22-attempt histories are a *separate restart* (2026-09-14 to 09-17; new A40; N* 85,458,944; new programs). The paper's lineage reached 8 attempts per arm, and only 4 existed at the paper's evidence cutoff.
- **"The fixed-token agent never changed batch size, a contract-induced blind spot."** This holds in one restart only. Across lineages, batch-size exploration appears in both arms, and is missing from one fixed-time arm (G3) and one fixed-token arm (G4). The G4 fixed-token arm is further confounded by a time-based LR schedule (05 §5). **No contract effect on agent behavior is supported.**
- **"Batch alone is worth about 0.1 BPB at about equal tokens."** Confirmed. In the seed-matched July ablation, batch alone gives −0.113 BPB, 97.8% of AutoResearch's 300-second gain (05 §3.6).

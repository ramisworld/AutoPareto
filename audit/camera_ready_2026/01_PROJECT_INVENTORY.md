# 01 — Project Inventory: AutoPareto (camera-ready audit, 2026)

- **Scope:** a read-only inventory of `/Users/Rami/Documents/dev/autopareto` (branch `autopareto/professor-ready-2026`, HEAD `6bcb69e`), its git database, sibling PDFs in `/Users/Rami/Documents/dev/`, and what is known about off-machine evidence.
- **Audit date:** 2026-10-28.
- **Camera-ready deadline:** 2026-11-13. Source: the user's memory file. No repository file records it.
- **Method:** Every SQLite file was opened only with `file:<path>?mode=ro&immutable=1`. Git was used only through read-only plumbing. No repository script, training job, GPU, network or pod was used.
- **Helper scripts and intermediate outputs:** `/private/tmp/claude-501/-Users-Rami-Documents-dev-autopareto/084ded0c-4778-442b-9b7f-c93e451f1f5d/scratchpad/audit/inventory/`. This includes `dbinventory.py`, `dbcompare.py`, `ckptmap.py`, `queuecmp.py`, `refcheck.py`, `dirstats.py`, and the sub-folders `forkE/` (runs/ namespaces) and `forkG/` (paper materials). Git-only databases were extracted, read-only, into `gitdbs/` there for analysis.
- **Labels:** **O** = OBSERVED, **D** = DERIVED (the calculation is shown), **I** = INFERRED, **U** = UNKNOWN, **M** = MISSING.
- **Times:** database times are UTC as stored. File mtimes are local time: NZST (UTC+12) until 2026-09-27, NZDT (UTC+13) after.
- **Secrets:** RunPod SSH endpoints and hostnames are never reproduced. Where they occur, they are counted only.
- **Missing input:** the plain-text PDF extraction named in the brief (`…/scratchpad/pdftext/automlr_autopareto.txt`) does not exist (**M**). The text was re-extracted to `…/inventory/forkG/pdftext/` and `…/inventory/forkE/automlr_autopareto.txt`.

---

## 0. Headline facts and discrepancies (summary)

1. **The working tree is not the research branch.** The checkout is `autopareto/professor-ready-2026`, which tracks 100 files. Of the 930 files tracked on `autopareto/research-infrastructure`, 877 are absent from the working tree (**O**). Absent items include:
   - 21 archive databases, 9 `results/preserved_checkpoints/*.json` manifests, 100 deployment-evaluation files, 11 downstream `results.json`;
   - 586 `results/phase1/**` files, plus the `scripts/`, `search/`, `tests/` and `dashboard/` sources.

   That branch exists **only locally**: no `origin/*` ref contains its tip `e74771d` (**O**). Git history is therefore the only copy of these files (§H).
2. **The canonical Phase-1 database is empty.** `results/phase1/downstream_200m.db` is **0 bytes** (mtime 2026-08-28 15:09). The populated copy exists in two places, and both are byte-identical (sha256 `d1166a3b84e6e91f…`, 62,943,232 B) (**O**):
   - `results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db`;
   - git blob `c84a683a` (research-infrastructure commit `e74771d`).
3. **The "missing runtime token-stream verification" claim is contradicted by local evidence.** `results/phase1/token_stream_verification.json` exists in git (`e74771d`, 2026-08-28 14:47) and in the RunPod snapshot. It reports `"status": "RUNTIME_VERIFIED"`, `observed: true` and `all_seed_stream_hashes_equal: true` (**O**). Three sources still say it is missing or unobserved (**O**):
   - `evaluation/phase1_token_stream_audit.json`, committed in the same commit, says `observed: false`;
   - `docs/RESEARCH_LEDGER.md` (L259–262, L445, L464);
   - `FORENSIC_RESEARCH_AUDIT.md` (L37, L343, L683, L722).
4. **The orphan-event cleanup deleted 242 `experiment_events` rows** between `safety-20260730T230009Z-before-orphan-event-cleanup.db` and `safety-20260731-before-post-seed42-preparation.db`.
   - The deleted ids are `9703–9786, 10063–10136, 10418–10501` (**O**). They reference experiment ids 89, 90 and 91 and training runs 28607, 29053 and 29504.
   - Raw JSONL shows they belong to **discarded original seed-42 attempts 6–8**, run on 2026-07-28 between 04:41 and 05:15Z. The id 91 had since been reused for a different experiment (**O**/**I**).
   - `PROJECT_STATE.md` calls them "orphaned database event duplicates" and mentions only a "lost attempt 5".
   - Experiments 88–90 appear in **no** snapshot. 357 further event ids were never captured in any snapshot (§B.3).
5. **Published-recipe checkpoints.** Local discovery checkpoints exist for baseline (exp 4), TPE (183), AP292/AP299/AP294 (134/137/136) and all 12 fixed-time confirmations (288–299).
   - **No local checkpoint exists** for the AutoResearch reference discovery (exp 68) or the MOTPE discovery (exp 234) (**M**). Downstream evaluation used substitutes: exp 77, the seed-2 re-run of exp 68's source, and exp 300, a seed-42 recovery re-training of exp 234's source.
   - All 21 matched-token Phase-1 `final.pt` files exist only as LFS objects in `.git/lfs/objects`, and all hashes verify (**O**).
   - The working-tree snapshot copy has 10 complete files, 1 **truncated** file (`phase1-autoresearch_compatible-seed4/final.pt`, 3,014,656 of 159,405,169 B) and 10 empty directories (**O**).
6. **Corrupted source.** `results/candidates/baseline-a40-seed0/0003-train.py` (exp 5) contains **2,178 NUL bytes** (sha `a3c94aeb…`). The other four baseline sources are identical to each other (`428e22f1…`) (**O**).
7. **All seven paper hash prefixes match local files** (§D).
8. **Why `.git` is 24.3 GB:** a Codex CLI turn-diff ref (`refs/codex/turn-diffs/checkpoints/…`, a tree from 2026-09-17 12:18) snapshotted the whole working tree, including 10.4 GB of `.pt` checkpoints and `runs/`.
   - The same ~10.3 GiB is stored twice: once in the pack `loose-4b362b….pack` and again as 6,155 duplicate loose objects (11.2 GB).
   - Add 2.25 GB of LFS objects, 0.44 GB of dangling loose objects and 0.28 GB of `tmp_obj` garbage.
   - Only 20.2 MiB on disk is reachable from real branches (§H).
9. **The Pair-A sample in the paper is not the 22-per-arm campaign.**
   - The paper's "4 attempts per arm" comes from `runs/pair-a-official-20260906-fixed-warmup-v2` (positions 1–8).
   - The 22-per-arm histories are in `runs/pair-a-official-20260914-new-a40-continuation-20260916-r2`. That is a **separate fresh restart** with a different baseline, N\*, program text and A40s, not a continuation (§E).
   - No repository document describes any namespace created on or after 2026-09-07.
10. **Which PDF was submitted is uncertain** (§G).
    - The designated submitted PDF `/Users/Rami/Documents/dev/automlr_autopareto.pdf` (sha `60b419b1…`, 10 pages, created 2026-09-06 22:59:19Z) matches no other local file.
    - `paper/master_v2/CHANGELOG.md` names a different version (V2).
    - The OpenReview deadline recorded locally ("Sep 06 2026 11:59AM UTC") is earlier than the creation time of every 10-page build.
    - `paper/main.tex` does not regenerate the V3 abstract.
    - `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf` is **M**, although the memory rule requires it.
11. **The public release lags the database.** `data/experiment_records.csv` (origin/main) holds 284 rows (ids 1–287). That equals `safety-20260810-after-discovery.db`, not `results.db`, which has 297 rows (**O**).
12. **Uncommitted edits to tracked files:**
    - `protected/inference_adapter.py`, a protected evaluator, mtime 2026-09-16 02:01, +125/−6 lines;
    - `docs/RESEARCH_LEDGER.md`, mtime 2026-09-03 17:38, +901/−940 lines.
13. **RunPod SSH endpoints appear in plain text.** By my regex they appear in at least 64 files under `runs/`, 5 under `study/contracts_2026/` and `docs/RESEARCH_LEDGER.md:853`; fork E counted 67 under `runs/`. They are not reproduced here and must be redacted before any release.

---

## A. Repository map

The working tree is on `autopareto/professor-ready-2026` (100 tracked files).
- `.gitignore` on that branch ignores only `.DS_Store`, `*.sqlite`, `*.db` and `__pycache__/`.
- Everything else not listed in that branch is untracked.

Sizes come from `du -sk`; file counts and mtimes come from `dirstats.py` (**O**).

### A.1 Top level

| Path | Size | Files (main types) | mtime range | Purpose | Tracked? |
|---|---:|---|---|---|---|
| `.git/` | 24,267 MB | — | 2026-07-19 → 2026-10-28 | Git database (§H) | — |
| `results/` | 10,912 MB | 2,404: .log 892, .json 408, .jsonl 375, .py 299, .txt 253, .pt 118, .db 12 | 07-20 → 09-10 | Experiment stores (DBs, checkpoints, logs, events, candidates, snapshots) | 17 files tracked (7 candidates, 10 `results/phase1/*`) |
| `runs/` | 1,270.7 MB | 2,846: .json 1,039, .jsonl 559, .py 539, .log 467, .cid 221 | 08-30 → 09-17 | Pair-A fixed-time vs fixed-token study, 27 namespaces (§E) | untracked |
| `.venv/` | 556 MB | — | 08-28 | Python environment (not inventoried) | — |
| `paper_backups/` | 434.0 MB | 3,159 (2,811 of them in a `.venv` copy) | 09-06 → 09-07 | Three paper backups plus `automlr_autopareto-before-1.4-fix.pdf` (§G) | untracked |
| `paper/` | 288.1 MB | 3,203 (`.venv` has 2,948 files, 139.7 MB) | 07-03 → 09-07 | Workshop manuscript, generated data, audit, master_v2 (§G) | untracked |
| `study/` | 13.5 MB | 296: .py 75, .pyc 152, .json 40, .md 20 | 08-28 → 09-17 | `contracts_2026` Pair-A harness, provenance, tests, tools | untracked |
| `tmp/` | 12.1 MB | 46 .png (`tmp/pdfs/delivery-*.png`) | 09-06 | Page renders used for paper QA | untracked |
| `queue/` | 2.8 MB | 104: .py 79, .pyc 25 | 07-28 → 09-04 | Frozen candidate inputs fed to runs (§D) | untracked |
| `plots/` | 1.6 MB | 15: .svg 7, .png 7, .md 1 | 08-28 | Public-release figures | tracked |
| `data/` | 0.9 MB | 3 .csv + README | 08-28 | Public CSV release (284 experiments; 2,784 inference rows; 233 deployment rows) | tracked; identical to origin/main |
| `scripts/` | 0.7 MB | 11 .py + 48 .pyc | 07-23 → 08-28 | Release and Phase-1 scripts. The .pyc files cover 38 modules whose sources exist only on research-infrastructure | 11 tracked |
| `__pycache__/` | 0.6 MB | 24 .pyc | 07-31 → 09-04 | Bytecode, including modules now absent (`runner`, `quantization_evaluator`, `change_effect_analysis`, `classical_campaign_analysis`) | ignored |
| `tests/`, `search/`, `dashboard/` | 0.3 / 0.1 / 0.0 MB | .pyc only (27 / 9 / 4) | 07-22 → 08-26 | Bytecode only; the `.py` sources are only in git (research-infrastructure) | — |
| `analysis/` | 0.3 MB | 35: .py 16 | 09-06 → 09-07 | Paper data pipeline (`build_dataset.py`, `analyze_*.py`, `generate_*.py`, `verify_evidence.py`, `master_v2/`) | untracked |
| `protected/` | 0.3 MB | 4 .py + 11 .pyc | 07-20 → 09-16 | Protected Phase-1 runtime and inference adapter. `inference_adapter.py` is modified and uncommitted | tracked |
| `docs/` | 0.1 MB | 5 .md | 08-28 → 09-03 | Research ledger and Phase-1 / downstream notes. `RESEARCH_LEDGER.md` is modified and uncommitted | tracked |
| `prelaunch_audit/` | 0.1 MB | 3 .md + `scratch/` (9 .py, 1 .c) | 09-02 | Pair-A pre-launch adversarial and repair audits; sandbox probe candidates | untracked |
| `evaluation/` | 0.1 MB | 5 .json | 08-28 | Downstream suite and Phase-1 manifests / audits | tracked |
| `configs/` | 0.1 MB | 18 .json + README | 08-28 | Public-release recipe configs (seeds 0, 1, 42) | tracked |
| `protocol/` | <0.1 MB | 2 .md | 08-28 | Public research protocol and ledger schema | tracked |
| `output/pdf/`, `worktrees/` | 0 | empty | 08-10 / 08-05 | Empty. The pod-side AutoResearch worktrees are not local | — |
| `audit/camera_ready_2026/` | — | This audit's deliverables, created 2026-10-28 from 00:44 by this and parallel audit agents | 10-28 | Audit outputs; not part of the research record and not inventoried | untracked |

Root files:
- Tracked code: `train.py`, `prepare.py`, `autopareto_db.py`, `inference_benchmark.py`, `deployment_benchmark.py`, `downstream_evaluator.py`, `benchmark_config.py`, `benchmark_stability.py`, `event_log.py`, `pyproject.toml`.
- Tracked docs: `README.md`, `RELEASE.md`, `REPRODUCIBILITY.md`, `LICENSE`, `PROJECT_STATE.md` (113 KB).
- Untracked:
  - `FORENSIC_RESEARCH_AUDIT.md` (2026-08-29 00:58) and `AUTOPARETO_TRAJECTORY_AUDIT.md` (08-29 09:36);
  - `validation_tmp_adversarial_{candidate,parent}.py` (09-02);
  - `autopareto_overleaf.zip` (145,341 B, 09-06 23:34) and `autopareto_final_overleaf.zip` (172,205 B, 09-07 00:13).

### A.2 Important subdirectories (**O**)

| Path | Size | Files | mtime range | Content |
|---|---:|---|---|---|
| `results/checkpoints/` | 9,614.9 MB | 107 .pt in 24 dirs (2 empty) | 07-20 → 08-12 | Fixed-time discovery and confirmation checkpoints (§C) |
| `results/snapshots/` | 1,044.0 MB | 598 | 08-14 → 09-06 | 3 pre-deployment DB snapshots, plus `runpod-<pod-id>-shutdown-20260901/` (933.9 MB): `inventory.json` (23 MB), `phase1_scientific_artifacts_no_checkpoints.tar.gz` (12.9 MB, 697 entries), `remote_worktree.patch` (10 file diffs), `extracted/` (575 files, 103 MB) and `scientific_final_checkpoints/` (21 dirs, 11 .pt, 797.8 MB) |
| `results/archive/` | 117.1 MB | 24 | 07-28 → 09-06 | 7 safety DBs with -shm/-wal sidecars (all WALs 0 B) and `setup-autopareto-seed42-20260730-tokenizer-cache/` (3 files) |
| `results/logs/` | 34.7 MB | 816 .log + `cache-verify.json` | 07-20 → 08-17 | Per-attempt training, inference, decode128 and deployment logs (§F) |
| `results/events/` | 30.0 MB | 282 .jsonl | 07-22 → 08-12 | Raw per-attempt event streams (§F) |
| `results/candidates/` | 6.4 MB | 252 .py in 30 dirs + 10 `._*` AppleDouble files | 07-20 → 08-17 | Frozen `train.py` per experiment (§D) |
| `results/samples/` | 0.3 MB | 251 .txt in 30 dirs | 07-20 → 08-12 | Generated text samples per experiment |
| `results/tmp/` | 0.3 MB | 48 .json | 07-28 → 07-31 | Seed-42 AutoPareto per-attempt train and inference JSON |
| `results/phase1/` | 0.1 MB | 11 (6 json, 2 md, 2 csv, 1 **0-byte** .db) | 08-28 | Public Phase-1 summaries (tracked) |
| `results/live/` | 0 | empty | 07-22 | — |
| `results/autopareto.sqlite` | 0 B | — | 07-29 15:04 | Empty file (**O**) |
| `paper/generated/` | 101.0 MB | 45 (34 json, 10 tex, 1 txt) | 09-06 → 09-07 | Exports of `downstream_200m.db` and analysis outputs (§G) |
| `paper/audit/` | 37.6 MB | 137 | 09-06 → 09-07 | Literature (11 papers in 3 formats), prompt refreshes, OpenReview group JSON, diffs of executed files, delivery manifest |
| `paper/master_v2/` | 1.2 MB | 28 | 09-07 | 14-page de-anonymized "master" rewrite |
| `paper_backups/paper-20260906-234643/` | 223.1 MB | 3,037 | 09-06 | Full `paper/` backup (V0) |
| `paper_backups/submitted-automlr-20260907-020829/` | 101.8 MB | 61 | 09-07 | Byte-identical to the current `paper/` sources (V2) |
| `paper_backups/automlr_overleaf_reconstructed/` | 101.7 MB | 60 | 09-07 | Source of the designated submitted PDF (V3) |
| `study/contracts_2026/provenance/` | 9.5 MB | 52 | 08-28 → 09-14 | Pre-study and N\* manifests, checkpoint registry, compute ledger, data and tokenizer manifests |
| `study/contracts_2026/historical_reference/` | 0.9 MB | 38 | 08-30 → 09-07 | Recovered historical `runner.py`, programs and scripts at commit `b8eab289` |

---

## B. Data stores

### B.1 Every SQLite file

Counts come from `dbinventory.py`, outputs in `…/inventory/db_worktree.json` and `db_git.json` (**O**). Abbreviations:
- **E** experiments, **TR** training_runs, **Ev** experiment_events, **C** campaigns, **CS** campaign_state;
- **IM** inference_measurements, **DM** deployment_measurements, **DS** deployment_sessions, **DR** downstream_evaluation_runs.

Every `-wal` sidecar in the working tree is 0 bytes, so the immutable reads are complete. The `-shm` files were last touched 2026-09-06 17:51–17:52 and 22:21, and `results.db-shm` on 09-10 12:13, showing earlier non-immutable opens (**O**).

**Working tree (12 populated + 2 empty)**

| # | Path | Bytes | sha256[:16] | Tables | E (max id) | TR (max id) | Ev (max id) | C / CS | Other non-zero tables | Earliest → latest timestamp |
|---|---|---:|---|---:|---|---|---|---|---|---|
| W1 | `results/archive/safety-20260730-before-attempt5-recovery.sqlite` | 7,704,576 | 7f5c301e2d69b2e4 | 12 | 87 (87) | 104 (27466) | 9,184 (9,459) | 13 / 10 | IM 528, DM 233, agent_actions 67, legacy 27, confirmation 6, ablation 6 | 2026-07-19 13:43:26 → 2026-07-30T03:02:56Z |
| W2 | `results/archive/safety-20260730T230009Z-before-orphan-event-cleanup.db` | 8,843,264 | 1d797f51725ef4d7 | 12 | 94 (97) | 112 (32295) | 10,984 (11,341) | 14 / 10 | IM 600, DM 233 | → 2026-07-30T22:59:07Z |
| W3 | `results/archive/safety-20260731-before-post-seed42-preparation.db` | 12,476,416 | d774b95c3d362062 | 12 | 109 (112) | 127 (37945) | 16,010 (16,609) | 14 / 10 | IM 780, DM 233 | → 2026-07-31T02:52:02Z |
| W4 | `results/archive/safety-20260803-before-dashboard-research-ledger.db` | 12,533,760 | 4fe632806f796b72 | 15 | 109 (112) | 127 | 16,010 | 16 / 12 | IM 780 (+3 empty downstream tables) | → 2026-07-31T02:52:02Z |
| W5 | `results/archive/safety-20260805T025129Z-before-seed1-closure.db` | 23,019,520 | 113a7ce4f08af0fb | 15 | 159 (162) | 177 (64101) | 30,017 (30,616) | 16 / 12 | IM 1,368 | → 2026-08-04T10:40:51Z |
| W6 | `results/archive/safety-20260805-before-classical-registration.db` | 23,040,000 | 3fefe3675752ab50 | 17 | 159 (162) | 177 | 30,017 | 16 / 12 | IM 1,368 (+2 empty quantization tables) | → 2026-08-04T10:40:51Z |
| W7 | `results/archive/safety-20260810-after-discovery.db` | 34,893,824 | 1d13b9434b9612b6 | 17 | 284 (287) | 302 (95985) | 44,606 (45,205) | 31 / 27 | IM 2,784, DM 233 | → 2026-08-07T15:10:59Z |
| W8 | `results/snapshots/results-before-final-deployment-20260814.db` | 38,211,584 | b109484096f09c99 | 17 | 297 (300) | 327 (100336) | 48,471 (49,070) | 44 / 40 | IM 2,940, DM 233, confirmation 18, DR 11, generations 198, task results 99 | → 2026-08-14T01:30:30Z |
| W9 | `results/snapshots/results-before-stabilized-canary-20260816.db` | 38,395,904 | fe6e47a65d1c3905 | 18 | 297 (300) | 327 | 48,471 | 44 / 40 | DS 2, DM 233 | → 2026-08-14T01:30:30Z |
| W10 | `results/snapshots/results-before-stabilized-deployment-20260817.db` | 38,785,024 | 79b3ff41c8a5ff93 | 18 | 297 (300) | 327 | 48,471 | 44 / 40 | DS 5, DM 368 | → 2026-08-16T05:29:28Z |
| W11 | **`results/results.db`** | 62,271,488 | 2c1a2cabd3abb7d6 | 18 | **297 (300)** | **327 (100336)** | **48,471 (49,070)** | 44 / 40 | IM 2,940, **DM 6,060**, DS 6, DR 11, generations 198, task results 99, confirmation 18, ablation 6, agent_actions 67, legacy 27; quantization 0 | 2026-07-19 13:43:26 → 2026-08-17T07:36:39Z |
| W12 | `results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db` | 62,943,232 | d1166a3b84e6e91f | 18 | **318 (321)** | 348 (120553) | 48,471 (49,070) | 45 / 40 | DM 6,060, DS 6, **DR 32, generations 576, task results 288** | → 2026-08-27T03:48:20Z |
| W13 | `results/phase1/downstream_200m.db` | **0** | — | — | — | — | — | — | — | mtime 2026-08-28 15:09 (**M**: canonical copy empty) |
| W14 | `results/autopareto.sqlite` | **0** | — | — | — | — | — | — | — | mtime 2026-07-29 15:04 |

Two further checks on W11 (**O**):
- `results.db` equals the pod's canonical DB recorded in `…/runpod-…/inventory.json` (`canonical_results_db`: sha256 `2c1a2cab…`, 62,271,488 B, snapshot taken 2026-08-31T14:42Z).
- No other SQLite files exist outside the `.venv` folders; I checked by header scan.

**Git-only databases (local branch `autopareto/research-infrastructure`; absent from the working tree)**

| # | Git path (blob) | Bytes | sha256[:16] | E (max) | TR (max) | Ev (max) | C | Latest timestamp | Note |
|---|---|---:|---|---|---|---|---:|---|---|
| G1 | `results/archive/2026-07-22-before-fresh-baselines.db` (4c8681c0) | 237,568 | d9bcfe893ca3ce70 | 7 (7) | 7 (7) | 73 (85) | 3 | 2026-07-22T03:55:18Z | 11 tables |
| G2 | `…/2026-07-22-before-legacy-label-correction.db` (30c11c20) | 442,368 | 54a0686618686ddc | 12 (12) | 12 (14897) | 328 (340) | 4 | 07-22T05:18:35Z | |
| G3 | `…/2026-07-22-before-readiness-metadata.db` (467e16e5) | 442,368 | b1150a3cab03c3b5 | 12 | 12 | 328 | 4 | 07-22T05:18:35Z | Same counts as G2 |
| G4 | `…/2026-07-24-diagnostic-attempt1-io-recovery/results.db.before-recovery` (85239e6b) | 5,304,320 | 450672b0ffbb8ec1 | 63 (63) | 63 (20784) | 6,740 (6,752) | 7 | 07-24 03:21:33 | It has a sidecar `results.db-wal.before-recovery` (1a02c9a9, 135,992 B): 33 frames of 4,096 B, 5 commit frames, 22 pages. The WAL was **not replayed**, per the immutable rule, so its 5 transactions are **U** |
| G5 | `…/2026-07-24-diagnostic-attempt1-before-metadata-recovery.db` (12aff3dd) | 5,308,416 | 277cbc1332c1ded3 | 64 (64) | 64 (21163) | 6,742 (6,754) | 7 | 07-24T04:45:11Z | |
| G6 | `…/2026-07-25-before-25x3-study.db` (6c7e33a0) | 5,775,360 | e662d23c825c1024 | 68 (68) | 68 (22477) | 7,347 (7,359) | 7 | 07-24T06:12:57Z | |
| G7 | `…/2026-07-26-before-local-completion-repair.db` (468527c0) | 5,787,648 | 4b4fa52a204b5e8e | 68 | 68 | 7,347 | 7 | 07-24T06:12:57Z | Same counts as G6; adds an empty DM table |
| G8 | `…/2026-07-27-before-confirmation-ablation.db` (303fffbc) | 5,787,648 | 69097a17455b5ece | 68 | 68 | 7,347 | 7 | 07-24T06:12:57Z | Named 07-27, but its content stops on 07-24 |
| G9 | `…/2026-07-27-before-autopareto-hardening-registration.db` (3aefbfc3) | 7,200,768 | f2deafb67e64c361 | 83 (83) | 100 (24450) | 8,826 (8,846) | 13 | 07-27T04:37:20Z | Confirmation 6, ablation 6, DM 233 |
| G10 | `…/2026-07-28-before-seed42-restart.db` (9d1a7eb5) | 7,204,864 | 200a846d20faa196 | 83 | 100 | 8,826 | 13 | 07-27T04:37:20Z | Same counts as G9 |
| G11 | `…/motpe-seed42-20260805-final-results.db` (f65c0bb1) | 31,109,120 | 58086f04fab6877c | 234 (237) | 252 (83010) | 40,181 (40,780) | 31 | 08-06T15:51:51Z | |
| G12 | `results/phase1/downstream_200m.db` (c84a683a, commit e74771d) | 62,943,232 | d1166a3b84e6e91f | 318 (321) | 348 | 48,471 | 45 | 08-27T03:48:20Z | **Identical to W12** |
| O1 | `…/2026-08-06-tpe-pretraining-infrastructure-failures-optimizer_state.db` (9c7d0f34) | 122,880 | fdff976e9a73b2ae | Optuna: study `tpe-seed42-20260805`, 4 trials (COMPLETE/RUNNING) | | | | 2026-08-05 03:59:18 → 04:00:05 | No matching `results.db` rows: the TPE experiments start at id 163 on 08-06 (**I**: pre-training false start) |
| O2 | `…/tpe-seed42-20260805-optimizer_state.db` (de4b3770) | 176,128 | de9c20576053eb1f | 25 trials | | | | 08-06 00:27:39 → 04:21:22 | |
| O3 | `…/cmaes-seed42-20260805-optimizer_state.db` (40ccc799) | 217,088 | 17652902fc96fbd3 | 25 trials | | | | 08-06 04:57:08 → 09:28:36 | |
| O4 | `…/motpe-seed42-20260805-optimizer_state.db` (c9f63f42) | 176,128 | 14d9b020bfc606d9 | 25 trials | | | | 08-06 10:45:57 → 15:51:53 | |
| O5 | `…/random-search-seed42-complete-optimizer_state.db` (2709f16e) | 237,568 | da15516c318c184f | 2 studies (MOTPE + random search), 50 trials | | | | 08-06 10:45:57 → 08-07 07:20:44 | |
| O6 | `results/optimizer_state.db` (b0bb3652) | 176,128 | 20962c2e2f470359 | NSGA-II only, 25 trials | | | | 08-07 10:28:18 → 15:11:00 | Final of 4 committed versions; the earlier versions are O2, O3 and O4 |

Also committed is `…/results.db-shm.before-recovery` (3 bytes). `paper/generated/db_*.json` are JSON exports of all 18 tables of W12 (fork G).

### B.2 Chronological lineage

Comparisons run on consecutive stores, keyed on primary ids; output is in `…/inventory/dbcompare_out.txt` (**O**). "None" means no row was added, removed or changed in experiments, TR, C, CS, Ev, confirmation, ablation, DS or DR.

| Transition | experiments | training_runs | campaigns / campaign_state | experiment_events | Other |
|---|---|---|---|---|---|
| G1→G2 | +8–12 | +5 | +`baseline-a40-cached-fresh-20260722`. `baseline-first`, `baseline-a40-async` and `baseline-a40-seed0` were **archived**, reason "superseded in dashboard by clean five-seed cached baseline suite" | +86–340 | |
| G2→G3 | none | | | | |
| G3→G4 | +13–63 | +51 | +`discovery-reference-seed42-20260722`, `autoresearch-seed42-20260723`, `autoresearch-diagnostic5-seed42-20260724` | +341–6752 | |
| G4→G5 | +64 | +21163 | diag5 ready→running | +6753–6754 | |
| G5→G6 | +65–68; exp 64 running→crash | +4; TR 21163 →completed | diag5 completed | +6755–7359 | |
| G6→G7→G8 | none | | | | |
| G8→G9 | +69–83 | +32 | +6: two ablations, `autopareto-seed42-20260727`, `confirmation-exp65/68-seeds012-20260725`, `setup-checkpoint-mount-20260727` | +7360–8846, **gaps 7368–69, 7507–09, 7647–49** | confirmation_runs +4–9; ablation_runs +1–6 |
| G9→G10 | none | | | | |
| G10→W1 | +84–87 | +4 | seed42 next 1→5 | +8847–9185 and 9441–9459, **gap 9186–9440** | |
| W1→W2 | **+91–97; 88–90 never present** | +8 | +`setup-autopareto-seed42-20260730-tokenizer-cache`; seed42 next 5→11 | +9460–61 and 9544–11341, **gap 9462–9543** | |
| W2→W3 | +98–112 | +15 | seed42 completed (26/25) | +11342–16609; **REMOVED 242** (§B.3) | |
| W3→W4 | none | | +`autopareto-seed0-20260731` and `autopareto-seed1-20260731` (ready) | | 3 empty downstream tables added |
| W4→W5 | +113–162 | +50 | seed0 and seed1 completed | +16610–30616 | |
| W5→W6 | none | | | | 2 empty quantization tables |
| W6→G11 | +163–237 | +75 | +15 classical campaigns (random-search, tpe, cmaes, motpe, nsgaii × seeds 0, 1, 42) | +30617–40780 | |
| G11→W7 | +238–287 | +50 | nsgaii-seed42 and random-search-seed42 completed | +40781–45205 | |
| W7→W8 | +288–300 | +25 | +13: 12 `autopareto-primary-*-seed{0,1,2}-20260810` and `downstream-recovery-motpe-exp234-seed42-20260812` | +45206–49070 | confirmation_runs +10–21; DR +1–8 and +11–13 |
| W8→W9 | none | | | | DS +1–2 (table created); DM unchanged at 233 |
| W9→W10 | none | | | | DS +3–5, sessions 1–2 changed; DM +234–368 |
| W10→W11 | none | | | | DS +6, session 5 changed; DM +369–6060 |
| W11→W12 | +301–321 | +120533–120553 | +`phase1-200m-downstream` (status still `planned`) | none | DR +14–34; generations +378; task results +189 |

Across all 23 populated stores, **no experiment, training_run or campaign row is ever deleted or renumbered** between snapshots. The changes are additions and status transitions only (**O**).

Additional checks (**O**):
- No `inference_measurements`, `legacy_metrics` or `agent_actions` rows disappear between sampled snapshots (G1, G2, G4, G5, G6, G9, W1, W2, W3, W6, W7, W11).
- No `deployment_measurements` rows disappear between W8 and W12.

### B.3 Deleted or renamed records (concrete ids)

**Experiments 88, 89, 90** are absent from every store, yet `sqlite_sequence.experiments = 90` already in W1 (**O**). Raw evidence for them survives outside the database (**O**):
- `results/archive/setup-autopareto-seed42-20260730-tokenizer-cache/events-attempt5-setup.jsonl` holds 82 events for exp 88 / TR 27902, 2026-07-28 04:27–04:36Z.
- `results/events/autopareto-seed42-20260727-0006.jsonl` holds exp 89 / TR 28607 (84 events, 04:41–04:48Z) **and** the re-run exp 93.
- `…-0007.jsonl` holds exp 90 / TR 29053 (74 events, 04:53–04:59Z) and exp 94.
- `…-0008.jsonl` holds exp "91" / TR 29504 (84 events, 05:07–05:15Z) and exp 95.

INFERRED reading (**I**):
- Seed-42 attempts 5–8 first ran on 2026-07-28 as experiments 88–91 and were discarded in the "attempt-5 recovery".
- They were re-run on 2026-07-30 as experiments 92–95.
- Id 91 was reassigned to the setup crash `setup-autopareto-seed42-20260730-tokenizer-cache` #5.
- `PROJECT_STATE.md` (L1229–1262) documents only "the lost attempt 5".

**242 events removed by the orphan-event cleanup** (between W2 at 2026-07-30T23:00:09Z and W3) (**O**):
- Ids: `9703–9786` (exp 89 / TR 28607), `10063–10136` (exp 90 / TR 29053), `10418–10501` (exp 91 / TR 29504). All have `source='live'`; event times run 2026-07-28T04:41:42Z → 05:15:24Z.
- The ids sit inside the id blocks of the 07-30 re-runs (exps 93, 94, 95). The JSONL files above are shared between the old and new attempts. Together these indicate the old events were re-ingested when attempts 6–8 were re-run (**I**).
- Before the cleanup, the exp-91 event stream mixed two different experiments (**O**).
- `PROJECT_STATE.md` L1294 says "242 orphaned database event duplicates … raw JSON event files were not deleted". The raw files are indeed present (**O**).

**357 event ids that never appear in any snapshot** (**O**). In `results.db`, 599 ids are missing, computed as 49,070 − 48,471 (**D**). Of these, 242 are the cleanup above; the other 357 never appear in any store:
- `15–26`, `7368–7369`, `7507–7509`, `7647–7649`;
- `9186–9440`: 255 ids sitting between exp 87's decision event (9185, 07-28 04:23Z) and its 07-30 re-benchmark events (9441+). This is the window of the discarded 07-28 attempts (**I**);
- `9462–9543`: 82 ids, the same count as the 82 archived exp-88 events (**I**).

**`downstream_evaluation_runs` ids 9 and 10** (**O**):
- They are absent from every store, while `sqlite_sequence = 13`.
- Interruption records `results/downstream_interruptions/run-0009-experiment-300.json` and `run-0010-experiment-300.json` survive in git (research-infrastructure).
- They were deleted between W7 and W8.

**Renames and reclassifications** (**O** in the DB, with the narrative per `PROJECT_STATE.md`):
- Exp 91 (experiment_number 5) sits under `setup-autopareto-seed42-20260730-tokenizer-cache`. PROJECT_STATE calls this a reclassification; it happened before W2, so no snapshot shows the old label (**U**).
- Archived campaigns: `baseline-first`, `baseline-a40-async`, `baseline-a40-seed0`, `setup-checkpoint-mount-20260727`, `setup-autopareto-seed42-20260730-tokenizer-cache`.

**Stale statuses in `results.db` and W12** (**O**):
- TR 14973 (exp 14) and TR 16716 (exp 36) are still `running`; their experiments are `interrupted`.
- Campaign `phase1-200m-downstream` is `planned`, although its 21 experiments are completed.
- 11 `campaign_state` rows remain `ready`: the 10 classical seed-0/1 campaigns, which have 0 experiments, and `discovery-reference-seed42-20260722`, which ran 1 of 5 attempts.

**Id sparsity** (**O**):
- training_runs: 327 rows with ids 7…100336; `sqlite_sequence` 120235 in `results.db`.
- legacy_metrics: 27 rows with sequence ≈32.7k.

  The likely cause is upsert-style writes consuming autoincrement values (**I**). This is not evidence of deletion.

**Store versus public release** (**O**):
- `data/experiment_records.csv` (284 rows, ids 1–287) matches W7.
- `data/inference_measurements.csv` (2,784 rows) matches W7 IM.
- `data/deployment_measurements.csv` (233 rows) matches deployment session 1.
- Exps 288–300 (12 confirmations + MOTPE recovery) and 301–321 (Phase-1) are **not** in the public CSVs.

---

## C. Checkpoints

### C.1 Counts, size and naming (**O**)

| Location | Files | Bytes | Naming |
|---|---:|---:|---|
| `results/checkpoints/<campaign_id>/<NNNN>.pt` | 107 in 22 non-empty dirs (+2 empty: `baseline-first`, `baseline-a40-async`) | 10,081,934,387 ≈ **10.08 GB (9.39 GiB)** | `NNNN` is the zero-padded `experiment_number`. The DB `experiments.checkpoint_path` records pod paths: `/workspace/autopareto/results/checkpoints/…` (274 rows), `/workspace/autopareto/results/preserved_checkpoints/…` (4), `/workspace/autopareto-diagnostics/…` (1, the recreated attempt-4 file), NULL (18) |
| `results/snapshots/runpod-…/scientific_final_checkpoints/phase1-<recipe>-seed<k>/final.pt` | 11 files in 21 dirs | 836,487,530 (797.8 MiB) | 10 complete; **`phase1-autoresearch_compatible-seed4/final.pt` truncated** at 3,014,656 B (sha `c323ee96…`, mtime 2026-09-01 02:58, an interrupted pull). Empty dirs: baseline 3/4/5, tpe_quality 3/4/5, motpe_quality 3/4/5, autoresearch_compatible 5 |
| `.git/lfs/objects/…` (pointers at `results/phase1/runs/phase1-*/final.pt` on research-infrastructure) | 21 | 2,248,733,805 (2.25 GB) | All 21 sha256 checks pass. They match the 21 hashes in `study/contracts_2026/provenance/model_checkpoint_registry.json` and the 21 Phase-1 DR `checkpoint_sha256` values |
| Pod only, per `runpod-…/inventory.json` (2026-08-31T14:42Z) | 188 under `results/phase1` | 21,428,546,444 | 47 `final.pt` (21 Phase-1 + 26 `NON_SCIENTIFIC-smoke-*`) and 141 `step-*.pt`. Of these, **167 are not local**, computed as 188 − 21 (**D**) |

Hash tables: `…/inventory/pt_hashes.txt` (all 118 working-tree `.pt`) and `ckptmap_out.txt`.

### C.2 Mapping to experiments (`ckptmap.py`, **O**)

| Campaign | Exps | Local `.pt` (exp ids) | Experiments with a recorded checkpoint path but no local file |
|---|---|---|---|
| baseline-a40-seed0 | 3–7 | 5 (3–7) | — |
| baseline-a40-cached-fresh-20260722 | 8–12 | 0 | 8–12 |
| discovery-reference-seed42-20260722 | 13 | 0 | 13 |
| autoresearch-seed42-20260723 (50 attempts) | 14–63 | 0 | 46 ids (15–35, 37–53, 56–63) |
| autoresearch-diagnostic5-seed42-20260724 | 64–68 | 0 | 64–68, **including reference exp 68** |
| confirmation-exp65-seeds012-20260725 | 72–74 | 0 | 72–74 |
| confirmation-exp68-seeds012-20260725 | 75–77 | 1 (77) | 75, 76 |
| ablation-batch-only / nonbatch-only | 78–83 | 0 | 78–83 |
| autopareto-seed42-20260727 | 84–87, 92–112 | 21 (92–112) | 84–87 |
| autopareto-seed0-20260731 | 113–137 | 24 (all except 129, which crashed) | — |
| autopareto-seed1-20260731 | 138–162 | 25 | — |
| tpe-seed42-20260805 | 163–187 | 4 (177, 183, 184, 185) | 19 |
| cmaes-seed42-20260805 | 188–212 | 4 (202, 206, 209, 210) | 21 |
| motpe-seed42-20260805 | 213–237 | **0** | 24, **including 234** |
| random-search-seed42-20260805 | 238–262 | 6 (243, 244, 246, 247, 259, 262) | 17 |
| nsgaii-seed42-20260805 | 263–287 | 4 (274, 275, 276, 282) | 19 |
| autopareto-primary-* (12 campaigns) | 288–299 | 12 | — |
| downstream-recovery-motpe-exp234-seed42-20260812 | 300 | 1 | — |

Integrity (**O**):
- 90 of the 107 local checkpoints have entries in git `results/preserved_checkpoints/*-manifest.json` (research-infrastructure), and **all 90 SHA-256 values match**.
- 17 have no manifest: exps 3, 5, 6, 7, 288–299 and 300.
- The 11 checkpoints referenced by `downstream_evaluation_runs` 1–8 and 11–13 all match their local files.
- The manifests list 179 experiments, 89 of which have no local checkpoint.
- No duplicate checkpoint files exist.

### C.3 Published recipes (**O** unless marked)

| Recipe | Discovery source exp | Discovery checkpoint local? | Fixed-time exact-source confirmations | 300-s downstream run checkpoint | Phase-1 200M seeds 3–5 (experiments 301–321) |
|---|---|---|---|---|---|
| Baseline | 4 (`baseline-a40-seed0` #2, seed 1) | **Yes**, `0002.pt` d8d03b3e | — (seeds 0–4 = exps 3–7 all local) | DR 1, exp 4 ✓ | exps 313–315: LFS ✓ (f958f870, 2c857112, 13c4953f); snapshot dirs empty |
| AutoResearch reference | 68 (`…diagnostic5…` #5) | **No (M)** | exp 75 (no file), 76 (incompatible, no file), 77 ✓ (87f50949) | DR 2, **exp 77** (substitute) | exps 310–312: LFS ✓ (846877696, e0ac7756, 7183ab49); snapshot seed3 ✓, seed4 **truncated**, seed5 empty |
| TPE | 183 (`tpe…` #21) | **Yes**, c931ca26 | — | DR 7, exp 183 ✓ | 319–321: LFS ✓; snapshot empty |
| MOTPE | 234 (`motpe…` #22) | **No (M)** | — | DR 11, **exp 300** (seed-42 recovery re-training of 234's source; 8865fdaa) | 316–318: LFS ✓; snapshot empty |
| AP292 | 134 (`autopareto-seed0` #22) | **Yes**, ebdfdd13 | 291/292/293, all local (292 = e913acf3) | DR 4, exp 292 ✓ | 301–303: LFS ✓ and snapshot ✓ |
| AP299 | 137 (#25) | **Yes**, 11753cb4 | 297/298/299 local (299 = 8a297f12) | DR 6, exp 299 ✓ | 307–309: LFS ✓ and snapshot ✓ |
| AP294 | 136 (#24) | **Yes**, bc7f28c8 | 294/295/296 local (294 = d5c29a5b) | DR 5, exp 294 ✓ | 304–306: LFS ✓ and snapshot ✓ |

Other missing checkpoints:
- Seed-42 AutoPareto attempts 1–4 (exps 84–87) are not local.
- The recreated attempt-4 checkpoint `/workspace/autopareto-diagnostics/attempt4-recreation/0004-recreated.pt` (sha `102fee96…` per PROJECT_STATE L1218) is pod-only (**M**).

---

## D. Candidate sources

### D.1 `results/candidates/` (**O**)

- **Mapping.** 30 directories hold 252 `.py` files. Every one maps 1:1 to a `results.db` experiment via `<campaign_id>/<NNNN>-train.py`.
- **AppleDouble files.** There are 10 `._*` AppleDouble files (163 B, "Mac OS X" headers), which are copy artifacts.
- **Experiments without a source file**, all failures:
  - autoresearch-seed42: 31 of 50 (exps 14, 21, 24, 27, 30, 31, 33, 35–39, 41–47, 49, 50, 53–60, 62, 63; all crash or interrupted);
  - exp 129 (crash: missing tokenizer cache);
  - OOM first attempts 163, 182 (TPE), 213 (MOTPE), 238, 249 (random search), 263, 272 (NSGA-II);
  - setup rows 1, 2, 69–71, 91;
  - confirmation exps 72–77 and ablation exps 78–83 do have files.
- **Corruption:** `baseline-a40-seed0/0003-train.py` (exp 5) contains 2,178 NUL bytes (sha `a3c94aeb8537…`). Files 0001, 0002, 0004 and 0005 are identical (`428e22f1…`). No other `.py` under `results/candidates`, `queue/` or `runs/` contains NUL bytes.
- **Exact-source confirmations are byte-identical** to their discovery sources:
  - exps 291–293 = `autopareto-seed0-20260731/0022` (55a7fd8b);
  - exps 294–296 = `0024` (6d0d8d1d);
  - exps 297–299 = `0025` (36e3ad91);
  - exps 288–290 = `0012` (c2a5e5dc).

### D.2 `queue/` (`queuecmp.py`, **O**)

| Queue dir | Files | Relation to `results/candidates` |
|---|---|---|
| `autopareto-seed42-20260727/` | 25 `N-train.py` | 25/25 identical |
| `autopareto-seed0-20260731/` | 25 `.py` + 25 `.pyc` | 24/24 identical. **`17-train.py` (exp 129) has no candidate counterpart**, so the queue is the only local source for exp 129 |
| `autopareto-seed1-20260731/` | 25 | 25/25 identical |
| `confirmation-exp65-seeds012-20260725/candidate-train.py` | 1 | = candidates 0001–0003 (b3072bc1) |
| `confirmation-exp68-seeds012-20260725/candidate-train.py` | 1 | = candidates 0001–0003 = **b3a961ee** (the AutoResearch reference) |
| `ablation-batch-only…/candidate-train.py` and `ablation-nonbatch-only…/candidate-train.py` | 1 each | = their candidates (d388c661 / 0863cad9) |

### D.3 Hashes of the seven published recipes (**O**; all prefixes match the paper)

| Paper label | File | SHA-256 | Exp | Paper prefix |
|---|---|---|---|---|
| Baseline | `results/candidates/baseline-a40-seed0/0002-train.py` | `428e22f114040f9fa9dd2d4da8b1ad005d7aa0be4dd3b395931c4537ba82f6b8` | 4 | 428e22f1 ✓ |
| AutoResearch | `results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py` | `b3a961eeea05dcec17c49cb1fa392dae70c1c87d520409da43d4f3fb9fd6c5b1` | 68 | b3a961ee ✓ |
| TPE | `results/candidates/tpe-seed42-20260805/0021-train.py` | `8547b2789c6635c5814ed67c0bb60256bf32bb24af0fcb744604ab59e17c76b2` | 183 | 8547b278 ✓ |
| MOTPE | `results/candidates/motpe-seed42-20260805/0022-train.py` | `62882a20b11c8c9b6f103f221dfc8621f43013a24c58c14da4ae5909e266e4b8` | 234 | 62882a20 ✓ |
| AP292 | `results/candidates/autopareto-seed0-20260731/0022-train.py` | `55a7fd8bf32924f2a35ebc095c5206f3503e671a9d7a9646b94979c8ae49259a` | 134 (label from confirmation row 292) | 55a7fd8b ✓ |
| AP299 | `results/candidates/autopareto-seed0-20260731/0025-train.py` | `36e3ad91a62810e4a907c1465adfbd1f07da446d0e63570c9e6a944b6066f12b` | 137 (row 299) | 36e3ad91 ✓ |
| AP294 | `results/candidates/autopareto-seed0-20260731/0024-train.py` | `6d0d8d1d9c7a08eeca1cd8d15b20b0bfdf5d84ee5255f56cd8a8092715290d10` | 136 (row 294) | 6d0d8d1d ✓ |

The git copies of these seven files on `professor-ready` (commit fa54db5) have the same hashes.

Runs/ Pair-A sources: fork E reports all 42 valid r2 attempt sources present and matching their recorded hashes. The two failed N-arm sources exist only under `pending/` (§E).

---

## E. `runs/` namespaces (Pair-A study)

Source: fork E (`…/inventory/forkE/`, scripts `ns_summary.py`, `prefix_cmp.py` and `manifest_check.py`), plus my spot checks (**O** unless marked).

Common facts for all namespaces:
- Schema `autopareto_pair_a_controller_v1`; study `autopareto-contracts-2026-pair-a`.
- 50 scheduled positions (25 per arm, interleaved N1, T1, N2, T2, …); schedule sha `ccd498d8…`.
- Arms: N = `pair-a-fixed-tokens-seed3`, T = `pair-a-fixed-time-seed3` (300 s).
- Every executed position used gpt-5.6-sol at high reasoning, Codex CLI 0.153.0, with a 600 s timeout. Exceptions:
  - the two gpt-5.6-terra/high namespaces;
  - the 900 s timeout in 20260909.

  The terra/medium setting in the 20260904 manifest and `docs/RESEARCH_LEDGER.md:853` was never executed.
- Earliest mtimes (08-30, and 09-05 18:38) are copy times, not activity dates.
- No `.pt` files are under `runs/`. Checkpoint paths point to the pod's `/var/lib/autopareto/…`.

| Namespace | MB / files | Activity (local) | Positions completed (N + T) and terminal state | Status |
|---|---|---|---|---|
| `pair-a` | 3.5 / 63 | 09-05 15:01–15:32 | 2 + 2; N1, T1 and T2 failed (`note_progress` TypeError); p05 running (KeyboardInterrupt) | Debug run. Its histories are byte-identical to `prestudy_debug/pair-a-interrupted-20260905` ("excluded from official analysis"). Holds the shared `provenance/` folder |
| `ABANDONED_NONSTUDY/pair-a-official-20260905` | 0.7 / 28 | 09-05 20:27–20:34 (terra/high) | N1 failed (KeyError 'N_STAR'); T1 running | Abandoned ("manually interrupted with Ctrl-C") |
| `pair-a-official-20260905-fresh` | 0.1 / 9 | 09-05 21:11 (terra/high) | 0 | Never launched |
| `pair-a-official-20260906-fresh` | 16.4 / 59 | 09-06 09:09–11:43 | 2 + 2; p05 running | Superseded |
| **`pair-a-official-20260906-fixed-warmup-v2`** | 26.4 / 100 | 09-06 14:07–16:33 | **4 + 4 (p1–8)**; p09 `reconciliation_required` | **The paper's "4 attempts per arm" prefix.** `paper/generated/budget_analysis.json` has `"current_namespace": "runs/pair-a-official-20260906-fixed-warmup-v2"`. History: 4 lines per arm, sha N `02129bb0…`, T `66290d1b…` |
| `…-continuation-20260907-preflight-{manifest-path,path,runtime-parent}-bug` (×3) | 26.7 / 102 each | states 09-07 12:36–12:49 | Same 4 + 4 records byte for byte; p09 running | Snapshots of failed preflights; no README |
| `…-continuation-20260907` | 38.3 / 122 | to 09-07 13:41 | 5 + 5; p11 reconciliation_required | Superseded |
| `…-continuation-20260907-new-a40-r2` | 96.2 / 183 | to 09-07 20:14 | 8 + 8; p17 reconciliation_required | Last state of lineage A; abandoned. `migration/abandoned_position_009.json` actually describes position 11 |
| `pair-a-continuation-preparation-20260907` | 0.0 / 3 | 09-07 | — | Replacement-A40 N\* of 88,080,384 was not adopted; 88,604,672 was kept |
| `pair-a-official-20260908-new-a40` | 0.2 / 17 | 09-08 19:09 | 0 (dry run) | Dry run only |
| `pair-a-official-20260909-new-a40` | 112.8 / 239 | 09-09 12:38 → 09-10 12:17 | 7 + 7; p15 reconciliation_required | Superseded; 5 operational amendments |
| `pair-a-official-20260912-fresh` and `-api-repair` | 0.1 / 10 each | 09-12 | 0 | Never launched |
| `preflight_pair_a_20260912_api_repair` | 0.1 / 7 | 09-12 | — | Passed dry run |
| `nonstudy_final_sol_high_smoke_20260912` | 0.4 / 22 | 09-12 01:32–01:34 | N1 failed | Validation, not study data |
| `nonstudy_api_repair_validation_20260912` (v1) | 0.2 / 17 | 09-12 01:43–01:45 | N1 reconciliation_required | Validation |
| `…_v2_20260912` | 0.4 / 19 | 01:47–01:50 | N1 reconciliation_required | Validation |
| `…_v3_20260912` | 0.4 / 19 | 01:52–01:55 | N1 reconciliation_required | Validation |
| `…_v4_20260912` | 4.2 / 35 | 02:00–02:30 | N1 and T1 completed (both dominated) | Validation |
| `pair-a-official-20260914-new-a40` | 161.0 / 292 | 09-15 08:18 → 09-16 03:23 | 10 + 9; p20 running (pod lost) | Predecessor of r2 |
| `pair-a-continuation-preparation-20260916` | 0.0 / 1 | 09-16 14:03 | — | Reuses N\* across the GPU change |
| `…-continuation-20260916` | 165.5 / 294 | state 09-16 14:10 | Same 19; no new results | Aborted continuation |
| **`pair-a-official-20260914-new-a40-continuation-20260916-r2`** | 534.5 / 597 | 09-15 08:18 → **09-17 12:59** | **22 + 22 (positions 1–44)**; p45 `reconciliation_required`; p46–50 pending | **Latest 22-per-arm histories.** `history.jsonl` has 22 lines per arm (N `ebb156f5…`, T `aa400ef2…`). N arm: 20 valid, plus failures at p37 and p39. `controller_state.json` is 210,512,257 B |
| `prestudy_debug` | 4.8 / 134 | 09-05 | 5 sub-runs, each with a PROVENANCE.md | Pre-study debug |
| `prestudy_validation` | 17.4 / 259 | 09-05 → 09-06 | 12 validation states, at most 1 position per arm | Pre-study validation |

Facts about the r2 campaign (**O** via fork E):
- Its p45 Scientist transcript contains a usage-limit message ("try again at Sep 19th").
- The 22-per-arm data is a **fresh restart**, not a continuation of the paper's prefix. Differences:
  - baseline `baseline-calibration-new-a40-20260914-run1` vs `baseline-calibration-repaired-20260903-run1`;
  - N\* 85,458,944 vs 88,604,672;
  - program hashes `825e53e4`/`8acffb40` vs `65a304a2`/`a1ea17a9`;
  - different A40s; positions 1–19 and 20–44 also ran on two different GPUs.
- No repo document describes namespaces from 09-07 onward. `docs/RESEARCH_LEDGER.md` was last edited 09-03.

---

## F. Logs, events and other result files (**O**)

| Store | Content | Coverage of the 297 `results.db` experiments |
|---|---|---|
| `results/logs/` (816 .log, 34.7 MB, 07-20 → 08-17) | `<campaign>-<NNNN>.log` (training), `-inference.log`, `-decode128.log` (146), `deployment-<session>-<exp>-context*.log`, setup/stabilized/smoke logs, `cache-verify.json` (pod cache paths) | 292/297. Missing: 14, 69–71, 91. Five logs are 0 B: `cache-prepare-vbsa.log`, `autopareto-preflight2.log`, `setup-deps.log`, `motpe-stop-retry.log`, `stabilized-retry.log` |
| `results/events/` (282 .jsonl, 30 MB, 07-22 → 08-12) | Raw event streams per attempt (`training_progress`, inference, decisions), plus `rebenchmark-*` and `downstream-recovery-*` | 276/297. Missing: 1–7, 14, 54, 55, 69–71, 91, 129, 163, 213, 238, 249, 263, 272. Contains the events of the discarded exps 88–91 (§B.3) |
| `results/samples/` (251 .txt) | One generated sample per experiment | 244/297 |
| `results/tmp/` (48 .json) | Seed-42 attempt train and inference JSON (07-28 → 07-31) | Seed 42 only |
| `results/archive/setup-autopareto-seed42-20260730-tokenizer-cache/` | `attempt5-setup.log`, `attempt5-setup-train.json`, `events-attempt5-setup.jsonl` | Discarded exp 88 and setup exp 91 |
| `results/snapshots/runpod-…/extracted/results/phase1/` (575 files) | 21 run folders (`config`, `telemetry.json`, `step_timing.jsonl`, …), 26 `NON_SCIENTIFIC-smoke-*` folders, stage B/C inference logs, `downstream_200m_evaluations/0014–0034/results.json`, `a40_timing_diagnostic/…/diagnostic.json` (3.3 MB, 2026-08-24, inference-only), `token_stream_verification.json` | Content-identical to git research-infrastructure `results/phase1/**` on all 573 shared paths; git also has 21 LFS pointers and `gpu.lock` |
| Git-only (research-infrastructure) | `results/deployment_evaluations/` (100), `results/downstream_evaluations/000{1–8},0011–0013/results.json` (~80 KB each: 9 task aggregates + 18 generations, **no per-example traces**), `results/downstream_interruptions/` (2), `results/preserved_checkpoints/` (9 manifests), 33 `results/*.json|csv|md|png` analyses (`final_deployment_stabilized_paper_table.csv`, `confirmed_pareto_frontier.json`, `discovery-freeze-20260810.json`, …), `results/archive/*.json` (3) | — |
| `data/*.csv`, `evaluation/*.json`, `study/contracts_2026/provenance/*.json` | Public CSVs (W7 state); Phase-1 manifests and token-stream audit; pre-study provenance (52 files: N\* freezes, checkpoint registry, compute ledger, data and tokenizer manifests) | — |
| `runs/**` | Scientist `stdout.jsonl` transcripts, `invocation.json`, `history.jsonl`, `position-*.json`, `.cid` container ids (221; containers were `--rm`, **I**: gone) | Pair-A only |

Agent transcripts (**O**):
- `agent_actions` has only 67 rows (exps 14–68, AutoResearch, 2026-07-22 → 07-24).
- The AutoPareto discovery campaigns have hypotheses, interpretations, decisions and diffs in `experiments`, but no transcript. `docs/RESEARCH_LEDGER.md` L114–117 and L447 mark it UNKNOWN.

---

## G. Paper materials

Source: fork G (`…/inventory/forkG/`, PyMuPDF from `paper/.venv` run with `-I`); I re-checked page counts and CreationDate.

| ID | Path | sha256[:16] | Pages | CreationDate (UTC) | What it is |
|---|---|---|---:|---|---|
| V0 | `paper_backups/paper-20260906-234643/main.pdf` | 49f8154cf399905a | 9 | 2026-09-06 11:34:32 | "AutoPareto: Separating Architecture from Resource Allocation…". No recipe hash prefixes. **V0 and V0c are the only builds created before the recorded deadline** |
| V0b | `autopareto_overleaf.zip` (459da467…) | — | — | — | V0 sources (main.tex f1a5a774) |
| V0c | PDF inside `paper_backups/paper-20260906-234643/working_source_bundle.zip` | cb0054fbbf16a9b3 | 9 | 10:21:07 | The only PDF covered by `paper/generated/pdf_qa.json` and `visual_qa.json` |
| V1 | PDF inside `autopareto_final_overleaf.zip` (916658d8…) | f46e4d2d654a4531 | 10 | 12:13:13 | New title "Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification". **The zip's `figures/` is empty, so it cannot recompile** |
| V1b | PDF inside `paper/working_source_bundle.zip` | 17be82cb677126a9 | 10 | 12:11:56 | Same sources as V1; README says "No submission has been performed" |
| V3-pre | `paper_backups/automlr_autopareto-before-1.4-fix.pdf` | 9609433ad395fe15 | 10 | 12:24:16 | V3 abstract with V2 body; Section 1.4 set `\footnotesize` |
| V2 | `paper/main.pdf` = `paper_backups/submitted-automlr-20260907-020829/main.pdf` | 88aaef390b318ed4 | 10 | 12:38:02 | Built from current `paper/main.tex` (2dd5d998…). Abstract: "A resource contract can act as a hidden research incentive" |
| **V3** | **`/Users/Rami/Documents/dev/automlr_autopareto.pdf`** | **60b419b1e8a22fa7** | **10** | **22:59:19** | Designated submitted PDF. Byte-identical to no other file. Source: `paper_backups/automlr_overleaf_reconstructed/` (main.tex = V2 plus 2 blank lines; different `generated/abstract.tex`; official template .sty) |
| V4 | `paper/master_v2/main.pdf` = `paper/master_v2/autopareto_master_v2.pdf` = `/Users/Rami/Documents/dev/autopareto_master_v2.pdf` | 2a4b34c0eccf7b18 | 14 | 2026-09-07 11:58:06 | De-anonymized, venue-neutral master. Adds the exposure confound (88.9M–251.8M discovery tokens, 2.83×), the full downstream table, an instrumentation section and an artifact-status table |
| — | `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf` | — | — | — | **M**, although the memory rule requires it |

Which version was submitted, and the competing evidence:
- `paper/audit/openreview_group.json` records "Deadline: Sep 06 2026 11:59AM UTC-0" and a `PC_Revision` invitation (**O**). Every 10-page build is later than that deadline (**O**).
- `paper/master_v2/CHANGELOG.md` names `submitted-automlr-20260907-020829`, i.e. V2. The same CHANGELOG says the submitted PDF rendered Section 1.4 in `\footnotesize`, which is true of V1 and V3-pre but **not** of V2 or V3 (**O**).
- Extracted text: V3 equals V3-pre except for hyphenation, and equals V2 except for the abstract and the affiliation lines (**D**, fork G).
- Which PDF OpenReview holds is **U**: no receipt or OpenReview copy exists locally.
- `paper/main.tex` plus `analysis/generate_reports.py` regenerate the V2 abstract, not V3's (**O**).

Headline numbers:
- Table 1 is identical in V0–V4, as are all macros:
  - decode ratio 1.99×; −41.7% parameters; −35.1% memory;
  - BPB cost 0.042034;
  - 201,326,592 tokens; 21 runs on seeds 3–5.
- How Pair-A is worded in the abstract:
  - V0: "begin a prospective … ablation";
  - V1: "four completed attempts per arm";
  - V2: "4-attempt prefix";
  - V3: not in the abstract (the body says "first four completed attempts of each arm (8 in total)");
  - V4: "exploratory prefix".
- The seven hash prefixes appear in V1–V4 only.

Other observations:
- **Generated data.** `paper/generated/` holds 45 files (101 MB).
  - `db_*.json` exports all 18 tables of W12.
  - `budget_attempts.json` (35.7 MB) has 28 rows: 8 current and 20 non-current. The non-current rows include duplicates, leaving 14 unique (fork E).
  - V2/V3 `\input` only `results_macros`, `abstract`, `main_results_table`, `primary_result` and `controls_table`.
- **Stale or broken artifacts.**
  - The QA files cover V0 only.
  - `claim_sentences.json` covers V2's abstract, not V3's.
  - `study/contracts_2026/pair_registration.json` is invalid JSON, per `parse_errors.json`.
  - `artifact_inventory.json` (3,331 entries, 09-06 18:34) is stale: `study/contracts_2026/provenance.py` is now absent and 23 harness files have changed.
  - `paper/main.bbl` holds 1 bibitem.
- **Missing references.**
  - A LaTeX comment points to `paper/part2_human_finalization.md`, which is **M** everywhere, including git history.
  - `paper/audit/delivery_manifest.json` refers to a PDF `e86cea72…` and a bundle `58ff517d…`; both are **M**.
- **Bibliography.** `references.bib` (12 entries) is identical across versions. `paper/audit/literature/` holds 11 retrieved papers in pdf, html and txt, plus `autoresearch_README.md`.
- **Private file.** `paper/AUTHOR_METADATA_PRIVATE.md` exists; its contents are not reproduced.

---

## H. Git

### H.1 Branches and refs (**O**)

| Ref | Tip / date | Commits | Root | Files at tip | Content |
|---|---|---:|---|---:|---|
| `master` (= `origin/master`, = `origin/HEAD`) | 228791f, 2026-03-25 | 36 | b11d6f2 "initial commit" (Andrej Karpathy, 2026-03-06) | 10 | **Upstream karpathy/autoresearch** (Karpathy plus contributor PRs #301, #342) |
| `origin/agenthub` | 7004de0, 2026-03-09 "hmmm" | 26 | b11d6f2 | — | **Upstream** Karpathy branch (1 ahead of / 11 behind master) |
| `origin/exp/H100/mar8` | 6c087cb, 2026-03-08 | 45 | b11d6f2 | 11 | **Upstream** autonomous-run branch (author "autoresearch"). `results.tsv`: "125 experiments, best val_bpb=0.969686" |
| `autopareto/research-infrastructure` (**local only**) | e74771d, 2026-08-28 14:47 | 129 (93 ahead of master) | b11d6f2 | 930 | All research infrastructure (07-20 → 08-28), committed archive DBs, Optuna DBs, `results/phase1/**` with LFS `.pt` pointers (`.gitattributes: *.pt filter=lfs`), deployment and downstream JSON, preserved-checkpoint manifests. Also a stray CV builder (`tmp/build_refined_cv.py`, `tmp/pdfs/*cv*.png`; personal, not research) |
| `origin/main` | 9cb262e, 2026-08-11 14:44 | 4 | **d21fa50 (orphan root)** | 40 | Public results release (below) |
| `autopareto/professor-ready-2026` (= origin) | 6bcb69e, 2026-08-28 15:33 | 6 | d21fa50 | 100 | origin/main plus fa54db5 "Publish professor-ready AutoPareto research state" (61 files) plus 6bcb69e "Correct abstract comparison wording" (`docs/ABSTRACT_DRAFT.md`) |
| `refs/codex/turn-diffs/checkpoints/2caa0642…/e6c05a08…/1789604280916/3cb05933-…` | tree e7237aa (not a commit), 2026-09-17 12:18 | — | — | 6,129 (12,225 MiB) | **Codex CLI working-tree snapshot**, including `results/checkpoints` and `runs/`. An empty ref dir `…/011b1f0e…/1789125717072` (09-11) suggests an earlier, deleted snapshot ref (**I**) |

Public-release commits on `origin/main`, all 2026-08-11 (**O**):
- `d21fa50` "initial results release": `.gitignore`, `README.md`, `configs/` (19).
- `a1dcafc` "measurement data": `data/` (4).
- `d2904b2` "plots and protocol": 6 plots, 2 protocol files.
- `9cb262e` "release reproducibility": LICENSE, RELEASE, REPRODUCIBILITY, `requirements-release.txt`, 3 scripts.

Remote (**O**):
- `origin https://github.com/ramisworld/autopareto.git` for fetch and push; no credentials embedded.
- `[lfs "https://github.com/ramisworld/autopareto.git/info/lfs"] access = basic`.
- `.git/lfs/cache/locks/refs/heads/autopareto/research-infrastructure/verifiable` exists. That could mean an LFS push of the local-only branch was attempted (**I**). Whether the LFS objects exist on GitHub is **U**.

### H.2 Why `.git` is 24.3 GB (**O**, with **D** totals)

| Component | Bytes | Notes |
|---|---:|---|
| `objects/pack/loose-4b362b49….pack` | 11,089,413,652 | 6,155 objects; created 2026-09-06 22:51, i.e. a `git maintenance` loose-objects pack (**I**); 4,656 objects reachable |
| Loose objects that duplicate that pack | 11,207,656,846 | All 6,155 pack objects also exist loose (`prune-packable: 6155`) |
| Dangling loose objects (unreachable, unpacked) | 443,417,975 | 774 objects |
| Other loose objects | 176,334,075 | 1,464 objects. **D**: 11,827,408,896 total loose − 11,207,656,846 − 443,417,975 |
| `objects/*/tmp_obj_*` garbage | 276,934,656 | 7 files, 2026-08-27 → 08-28 (interrupted writes) |
| `objects/pack/pack-268d….pack` | 547,703 | 193 objects (original upstream clone) |
| `lfs/objects` | 2,248,733,805 | 21 Phase-1 `final.pt` |
| **Total** | **25,443,038,712 ≈ 23.70 GiB** | **D**: sum of the rows; this matches `du` = 24,267 MiB |

How the contents break down (**O**):
- Reachable from all refs: 4,694 blobs, 11,852 MiB uncompressed and 9,503 MiB on disk.
- Reachable from branches and remotes only: 1,493 blobs, **201.7 MiB uncompressed and 20.2 MiB on disk**.
- So about 99.8% of on-disk blob bytes are reachable only via the Codex ref. **D**: 1 − 20.2 / 9,502.6.

Largest blobs (**O**):
- `runs/…-continuation-20260916-r2/controller_state.json`: 184.6 MiB, the 09-17 12:18 version.
- `results/checkpoints/nsgaii-seed42-20260805/0014.pt`: 169.3 MiB.
- 20 further checkpoints of 152.0 MiB each, e.g. `results/snapshots/runpod-…/scientific_final_checkpoints/phase1-autoresearch_compatible-seed3/final.pt`, `autopareto-seed42-20260727/0024.pt` and `baseline-a40-seed0/0005.pt`.
- Then `runs/…-r2/pair-a-fixed-tokens-seed3/history.jsonl` at 94.1 MiB.
- By extension: `.pt` 139 blobs / 10,412.6 MiB (all via the Codex ref); `.json` 669.5 MiB; `.jsonl` 532.6 MiB; `.db` 17 blobs / 127.2 MiB.

Answer to whether checkpoints and databases are committed:
- **Checkpoints are never committed to a branch.** They are reachable only through the Codex snapshot tree. The research-infrastructure branch holds only 21 LFS pointers.
- **Databases are committed only on the local-only research-infrastructure branch:** 19 blobs (G1–G12, O1–O6, WAL/SHM) totalling about 155 MB uncompressed.
- The Codex tree contains no `*.db`/`*.sqlite`, because the professor-ready `.gitignore` excludes them, but it does include their `-shm`/`-wal` sidecars.
- origin/main and professor-ready contain no DBs or checkpoints.

Other observations (**O**):
- 1,423 loose objects (born 09-07 → 09-17) and both packs had their **mtime freshened on 2026-10-28 between 00:41:35 and 00:41:38**, minutes before this audit began. No new objects or refs were created. **I**: some tool re-hashed the working tree.
- `git stash list` is empty; the only worktree is the main one.
- Working-tree modifications against HEAD: `docs/RESEARCH_LEDGER.md` (+901/−940) and `protected/inference_adapter.py` (+125/−6; adds `source_sha256`/`config_sha256`, `format_version` 1→2).

---

## I. External or inaccessible evidence

| Item | What is known | Source of the knowledge | Status |
|---|---|---|---|
| RunPod persistent-volume snapshot `/workspace/PRESERVED_20260917T021347Z` (8 GB) and `RESUME.md` | It copies the pod's ephemeral `/var/lib/autopareto`, `/var/cache/autopareto`, `/opt/autopareto_trusted` and `/opt/autopareto_runtime`. Pair-A positions 1–44 were complete and p45 was `reconciliation_required`; the pod was stopped (EXITED) on 2026-09-17. It probably holds the Pair-A checkpoints referenced by `runs/**` paths `/var/lib/autopareto/…` (**I**) | **Only** the user's memory file `~/.claude/projects/-Users-Rami-Documents-dev-autopareto/memory/autopareto-a40-pod-and-snapshot.md`. No repository file mentions `PRESERVED_2026` or `RESUME.md` (grep) | **Inaccessible**; not contacted |
| Earlier pod `/workspace/autopareto` (shutdown snapshot 2026-08-31T14:42Z) | `inventory.json` lists 70,704 `results/phase1` files: 188 checkpoints (21.4 GB) and about 70.5k non-checkpoint files (mostly compile caches). It also shows pod-side `results/preserved_checkpoints/…`, `/workspace/autopareto-diagnostics/…`, `/workspace/.cache/autoresearch`, and git HEAD `5bf86b57` with 10 dirty files (`remote_worktree.patch`) | `results/snapshots/runpod-<pod-id>-shutdown-20260901/{inventory.json,remote_worktree.patch}` | Volume state after shutdown is **U** |
| Provider-side agent logs (OpenAI / Codex) for the July–August discovery campaigns | Model gpt-5.6-sol at high reasoning (`campaigns` table). There are no transcripts: `agent_actions` has 67 rows, AutoResearch only. Billing and energy were not recorded (`study/contracts_2026/provenance/compute_ledger.json` lists `not_recorded`) | DB; `docs/RESEARCH_LEDGER.md` L114–117 and L447 | **M** locally. For Pair-A, Scientist transcripts are local in `runs/**/stdout.jsonl` (hash-matched, fork E) |
| Dataset and tokenizer caches | Pretraining data `karpathy/climbmix-400b-shuffle` at revision `915333b4…` (validation shard `shard_06542.parquet`); tokenizer `tokenizer.pkl` and `token_bytes.pt`. Downstream sets: `allenai/ai2_arc`, `baber/piqa`, `nyu-mll/blimp`, `EleutherAI/lambada_openai` (revisions in `downstream_task_results`). Hashes are in `study/contracts_2026/provenance/{data_manifest,tokenizer_manifest}.json` | `results/logs/cache-verify.json` (runtime `/root/.cache/autoresearch` → `/workspace/.cache/autoresearch`), DB rows | **Pod-only.** There is no `~/.cache/autoresearch` on this Mac, and the local HuggingFace cache holds none of these datasets (**O**) |
| OpenReview (NeurIPS 2026 AutoMLR, forum `U9ICjFvtN8`) | Poster acceptance, discussant format, camera-ready 2026-11-13 | Memory file `autopareto-automlr-camera-ready.md`; `paper/audit/openreview_group.json` holds only the venue-group metadata (deadline; last modified 2026-09-03T17:42Z) | Reviews, meta-review, decision and the as-hosted PDF are **M** locally |
| GitHub remote state for the local-only branch and LFS | — | — | **U** (not contacted) |

---

## J. Data that exists but is not in the manuscript

The manuscript checked is V3 (`/Users/Rami/Documents/dev/automlr_autopareto.pdf`, extracted text) together with `paper/main.tex`. Each item records the absence or the limited use I observed.

1. **Pair-A attempts 5–22 per arm** (`runs/pair-a-official-20260914-new-a40-continuation-20260916-r2`, 44 attempts). Also not reported:
   - lineage-A attempts 5–8 per arm (`…-continuation-20260907-new-a40-r2`);
   - the 7 + 7 of `…20260909-new-a40`.

   The paper reports only the 4 + 4 prefix and does not say a 22-per-arm restart exists.
2. **July batch ablations** (`ablation-batch-only-seeds012-20260725` and `ablation-nonbatch-only-seeds012-20260725`, exps 78–83, `ablation_runs` 1–6, source exp 68). The word "ablation" appears 0 times in the PDF text, `paper/main.tex` and `paper/generated/*.tex`.
3. **Fixed-time exact-source confirmations on seeds 0–2.** None of the 12 runs' values (1.0939, 1.0928, 1.0975, …) appears in the PDF; the lowest-memory role appears nowhere:
   - `autopareto-primary-*-seed{0,1,2}-20260810` = exps 288–299 (`confirmation_runs` 10–21). AP292/AP294/AP299 borrow their row numbers as labels only (PDF L94–97);
   - the lowest-memory role (exp 124 → 288–290);
   - the earlier `confirmation-exp65/68-seeds012-20260725` (exps 72–77; 4 of 6 inference-incompatible).
4. **AutoResearch 50-attempt campaign** (`autoresearch-seed42-20260723`, exps 14–63: 19 completed, 29 crash, 2 interrupted). It is referred to only as "a quality-only AutoResearch campaign/archive". Its failure history is not reported.
   - The archive-best "AutoResearch (1.0950)" in the PDF is **exp 65** (diagnostic5 #2, inference-incompatible per `confirmation_runs` 4–6), not the reference recipe exp 68 (**O**).
5. **diagnostic5 campaign** (`autoresearch-diagnostic5-seed42-20260724`, exps 64–68) is never named. Only exp 68's source is used, as "AutoResearch reference".
6. **Classical seeds 0 and 1:** 10 campaigns (`{random-search,tpe,cmaes,motpe,nsgaii}-seed{0,1}-20260805`) are registered as `ready` with 0 experiments. Only seed-42 archives exist and are reported.
7. **Nonstudy and pre-study runs:** `runs/nonstudy_api_repair_validation{,_v2,_v3,_v4}_20260912`, `nonstudy_final_sol_high_smoke_20260912`, `ABANDONED_NONSTUDY`, `prestudy_debug`, `prestudy_validation`, `preflight_pair_a_20260912_api_repair`.
8. **A40 timing diagnostic** (`results/phase1/a40_timing_diagnostic/phase1-a40-timing-diagnostic-20260824/diagnostic.json`, inference-only, 4 sessions, 6,745 telemetry samples; config `evaluation/phase1_a40_timing_diagnostic.json`, git-only).
9. **Multi-workload deployment characterization:** deployment sessions 1, 5 and 6 (6,060 `deployment_measurements` total; 16 workloads, context 256/896/1024/1536 × batch 1/2/4/8). Also `final_deployment_stabilized_paper_table.csv` (git-only). The paper uses only the batch-1, 256-in/256-out protocol.
10. **300-s-regime downstream evaluations:** `downstream_evaluation_runs` 1–8 and 11–13 (exps 4, 77, 289, 292, 294, 299, 183, 202, 300, 247, 275, covering CMA-ES, random search and NSGA-II representatives). The paper's downstream comparison uses the 200M Phase-1 panel (runs 14–34).
11. **26 `NON_SCIENTIFIC-smoke-*` Phase-1 runs** (2026-08-23 → 08-25; pod checkpoints, snapshot metadata).
12. **Discarded seed-42 attempts 5–8 of 2026-07-28** (exps 88–91, original) and the setup and infrastructure failures (exps 1, 2, 69–71, 91; the TPE Optuna false start O1).
13. **Quantization:** `evaluation/quantization_suite.json` (git-only) exists, but `quantization_runs` and `quantization_measurements` have 0 rows. It was planned and never run.
14. **The `discovery-reference-seed42-20260722` campaign** (1 of 5 attempts run) and the five-seed cached baseline suite (exps 8–12).

---

## K. Missing artifacts

Items referenced in docs, code or DB that are absent locally.

| Artifact | Referenced by | Local status |
|---|---|---|
| Per-example downstream traces (logits, choices) | `docs/DOWNSTREAM_EVALUATION.md:262`, `docs/RESEARCH_LEDGER.md:448`, `FORENSIC_RESEARCH_AUDIT.md:690` | **M**: never persisted. `results.json` files hold only 9 task aggregates and 18 generations |
| Checkpoints of the AutoResearch reference (exp 68) and the MOTPE discovery (exp 234) | `downstream_evaluation_runs` substitutes, `FORENSIC_RESEARCH_AUDIT.md:684` | **M** (substitutes exp 77 and exp 300 exist) |
| Checkpoints for seed-42 attempts 1–4 (exps 84–87) and `/workspace/autopareto-diagnostics/attempt4-recreation/0004-recreated.pt` | DB `checkpoint_path`, `PROJECT_STATE.md` L1217–1219 | **M** (pod-only) |
| 167 pod Phase-1 checkpoints (141 `step-*.pt`, 26 smoke `final.pt`) | `runpod-…/inventory.json` | **M** (pod-only) |
| A complete copy of `phase1-autoresearch_compatible-seed4/final.pt` in the working tree | Snapshot dir | Truncated; complete copy only in `.git/lfs/objects/e0/ac/e0ac7756…` |
| Populated `results/phase1/downstream_200m.db` at its canonical path | `docs/RESEARCH_LEDGER.md`, `docs/DOWNSTREAM_EVALUATION.md` | **0 bytes**; populated copies in the snapshot and git |
| `results/phase1/token_stream_verification.json` in the working tree | `evaluation/phase1_token_stream_audit.json` (`output`) | **Absent from the working tree but present** in git e74771d and the snapshot. The docs calling it missing are stale (§0 item 3) |
| `results/preserved_checkpoints/*.json`, `results/archive/2026-07-*.db`, `results/downstream_evaluations/**`, `results/deployment_evaluations/**`, `scripts/*` (41), `search/*.py`, `tests/*.py`, `dashboard/*`, `programs/*.md`, `runner.py`, `program.md` | `PROJECT_STATE.md` (52 backticked refs resolve git-only), bytecode in `__pycache__` | Git-only on the **local-only** branch (§H) |
| Raw logs and samples for exps 1–7, 14, 54–55, 69–71, 91, 129 and the OOM first attempts | DB `log_path`/`sample_path` | Partially **M** (§F coverage) |
| AutoPareto agent transcripts and provider logs | `docs/RESEARCH_LEDGER.md` L447 | **M**/**U** |
| Content of 5 WAL transactions in `2026-07-24-diagnostic-attempt1-io-recovery/results.db-wal.before-recovery` | Git | **U** (not replayed, per the read-only rule) |
| Original source of exp 5 (`baseline-a40-seed0/0003-train.py`) | DB `experiments` | **Corrupted** (NUL bytes); recoverable from identical siblings (**I**) |
| Candidate source of exp 129 in `results/candidates` | — | Only in `queue/autopareto-seed0-20260731/17-train.py` |
| Pair-A: `runs/pair-a/provenance/pair_a_active_provenance_20260904.json`, `runs/operational-transfer-smoke-20260902`, `runs/pair-a-seed3-baseline-training.json`, `runs/calibration_seed3.json`, top-level `runs/pair-a-fixed-*-seed3/history.jsonl`, pre-study smoke manifest `c41be2c6`, overwritten manifest versions `ec105648…` (r2 p1–5) and `b7321380…` (p38–39) | Study provenance, `scheduling.py`, registration (fork E) | **M** |
| `paper/part2_human_finalization.md`; delivery PDF `e86cea72…` and bundle `58ff517d…`; `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf` | LaTeX comment; `paper/audit/delivery_manifest.json`; memory rule | **M** |
| `study/contracts_2026/provenance.py` | `paper/generated/artifact_inventory.json` | **M** (since 09-06) |
| `…/scratchpad/pdftext/automlr_autopareto.txt` (named in the audit brief) | Audit brief | **M**; re-extracted under `forkG/` and `forkE/` |
| OpenReview reviews, decision, and the as-hosted PDF | Memory file | **M** locally |

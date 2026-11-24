# 11 — Validity Verification (independent re-check from primary records)

- **Date:** 2026-11-02. **Mode:** read-only. Nothing under `/Users/Rami/Documents/dev/autopareto` was modified; every SQLite file was opened as `file:…?mode=ro&immutable=1`; the WAL files are 0 bytes, so `immutable` loses no data. No training, GPU, network or git state change.
- **Inputs.** `R:` = `/Users/Rami/Documents/dev/autopareto`. `SNAP` = `R:results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/` (its `downstream_200m.db` is a superset of `R:results/results.db`, 318 vs 297 experiment rows). `RI:` = git branch `autopareto/research-infrastructure` (read with `git show`). The prior audit reports `01`–`10` were **not** used as evidence, only as a list of claims to test.
- **Scratch scripts** (reproducible): `/private/tmp/claude-501/-Users-Rami-Documents-dev-autopareto/4730a939-4447-436c-ab09-c479899521a8/scratchpad/validity/` — `phase1_check.py`, `pairA_scan.py`, `nsched.py`, `archive_check.py`, `manifest_match.py`, `dis310.py`, `build_flags.py`.
- **Companion overlay:** `supporting/historical_validity_flags.json` (99 flags; §5).
- **Labels:** OBSERVED = read directly from a record; DERIVED = computed from records; INFERRED = reasoned, not directly recorded; UNKNOWN = not determinable from local evidence.

---

## 0. Bottom line

1. **BPB is one function in every era.** [DERIVED] All 297 discovery/confirmation/ablation/classical/recovery rows (ids 1–300) carry a `protected_manifest_sha256` that I re-derived exactly from `RI:benchmark_config.PROTECTED_FILES` at 15 specific commits. In every case `prepare.py` = `4f2ba9cb…` (upstream `09ebea4`). Pair-A's protected evaluator registers the same hash. Phase 1 used `7132ec46…`, whose only change is a `device` argument for the dataloader. No evaluator bug was found. The bytes denominator is an approximation, constant across models (§1.2).
2. **Validation BPB and held-out BPB are not interchangeable.** [DERIVED] Held-out exceeds validation by 0.050–0.072 BPB, and the offset is **recipe-dependent**: 0.053 for the baseline against 0.064–0.069 for AR, AP292 and MOTPE in the 201M panel. Rankings differ between the two metrics. The held-out segments are not aligned to BOS; validation rows are (§1.7). This weakness in the measurement went unreported before.
3. **Phase-1 integrity holds.** [OBSERVED/DERIVED] For all 21 runs, the seed, source SHA, derived-source SHA, checkpoint SHA (training record = milestone manifest = evaluation record = downstream DB), token count and held-out value agree. All 21 `final.pt` objects are present in `R:.git/lfs` and hash-verified.
4. **The 3.2159 runs are confirmed as a harness re-exec artifact, with two new pieces of evidence** (§4a):
   - the defect is present in the cached **3.10 bytecode of an earlier `candidate_process.py`** (35,471-byte source, mtime 2026-09-05);
   - all 19 signature files show a **timing fingerprint**: +0.45–0.61 s before the last update, against ≤0.01 s in the 147 healthy files.
5. **Corrections to the prior audit:**
   - **(i)** The Pair-A G3 fixed-token arm is **not** schedule-confounded. All 7 attempts use `contract_progress`, which returns token progress. Only G4 is (§4, extra).
   - **(ii)** The seed-42 rollback deleted **579 event rows**, not 242: 255 were already gone before the earliest local snapshot, plus 82 and then 242 (§4c).
   - **(iii)** `experiments.git_commit` for 67 discovery rows (ids 91–112, 118–162) is the **upstream** master merge `228791fb`, which contains no `runner.py`. Code identity must come from the manifest digest instead.
   - **(iv)** `tokenizer_manifest.json` and the ledger say runtime token-stream verification is missing, but `SNAP:token_stream_verification.json` exists and says `RUNTIME_VERIFIED`.
   - **(v)** The tokenizer was **retrained on every fresh Pod** during discovery, and no hash was recorded until 2026-08-26.
   - **(vi)** The 4→3 narrowing to the 201M panel was fixed on **2026-08-25**, after the 300-s confirmation and downstream results were known.
6. **Checkpoint substitutions (exp 77 for 68, exp 300 for 234) do not touch any paper number.** [OBSERVED] Every downstream number in `paper/main.tex:68` comes from the 201M panel (§3).

---

## 1. BPB implementation

### 1.1 Code paths that produce reported BPB

| Era / quantity | Code | Executed version (evidence) | Model state | Data / length |
|---|---|---|---|---|
| Discovery, 300-s confirmations, ablations, classical, exp 300 (`experiments.val_bpb`) | `train.py:619–622` → `prepare.evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)` (`prepare.py:347–369`) | `prepare.py` **4f2ba9cb** for all ids 1–300 [DERIVED: `manifest_match.py` reproduces all 15 distinct `protected_manifest_sha256` values; e.g. ids 13–63 → `3d22539f`, ids 113–162 → `b8eab289`, ids 238–300 → `7813202a`] | in-process after training, `model.eval()`, bf16 autocast | `shard_06542`, 20,971,520 target positions |
| Pair-A protected validation (`protected_validation_bpb`) | `study/contracts_2026/train_launcher.py:190–221` (trusted parent reloads the checkpoint, then calls `prepare.evaluate_bpb`) | registered `prepare` **4f2ba9cb** (`train_launcher.py:38`; every one of the 166 position files: `protected_validation_provenance.prepare_sha256`) [OBSERVED] | reloaded, `model.eval()` (`protected/inference_adapter.py:81`), `inference_mode` + bf16 | same; batch 64/32/16 (119/41/6 files) |
| Phase-1 validation (DB `val_bpb`) | `protected/phase1_train_launcher.py:267–274` captures in-process `evaluate_bpb` | `prepare.py` **7132ec46** (`token_stream_verification.json:prepare_sha256`); diff vs 4f2ba9cb = `device=` argument only (`git diff 09ebea4 fa54db5 -- prepare.py`) [OBSERVED] | in-process | same |
| Phase-1 validation (paper, `phase1_equal_token_results.json`) | `scripts/evaluate_phase1_checkpoints.py:25–56` → `prepare.evaluate_bpb` on reloaded milestone/final checkpoints, batch from manifest (64) | `checkpoint_evaluations.json:evaluator = "prepare.evaluate_bpb"` [OBSERVED] | reloaded, eval, bf16 | same |
| Held-out BPB (paper Table 1, `downstream_task_results`) | `downstream_evaluator.py:480–586` (`download_heldout_shard`, `heldout_segments`, `evaluate_heldout_bpb`) | 300-s panel `97a59f93…` (= `RI:3e3645c`); phase 1 `696f9c46…` (= `fa54db5`). The diff is a `module=` reuse argument plus `contiguous()` and a finiteness check (`git diff 985544c 3e3645c`, `3e3645c e74771d`) [OBSERVED] | reloaded, eval, `inference_mode` + bf16 | `shard_06541`, 2,560 × 2,048 = 5,242,880 targets |

### 1.2 What the formula does (all paths)

- **Log-probabilities and units.** [OBSERVED] The candidate's `model(x, y, reduction='none')` returns per-token cross-entropy in **nats**. The total is divided by `math.log(2) * total_bytes` (`prepare.py:363–369`; `downstream_evaluator.py:552–561`). Natural-log-to-bits conversion is correct.
- **Aggregation.** [OBSERVED] Ratio of sums: total nats over total bytes, then ×1/ln 2. **Not** a mean of per-token or per-segment BPB. Held-out also gives a segment-bootstrap ratio CI (`downstream_evaluator.py:56–77`, 1,000 repetitions, seed 20260731).
- **Bytes denominator.** [OBSERVED] `token_bytes[id] = len(enc.decode([id]).encode("utf-8"))`, and special tokens get 0 (`prepare.py:184–195`).
  - [INFERRED] `tiktoken.decode` uses `errors="replace"`. A token that is a partial UTF-8 sequence therefore counts as 3 bytes (U+FFFD), not its true 1–2 bytes. The denominator is the sum of per-token decoded lengths, **not** the UTF-8 byte length of the original text.
  - The bias is a fixed property of the scored target sequence, so it cannot change rankings within one metric. It does scale validation and held-out by slightly different factors. Magnitude: [UNKNOWN] (the tokenizer and shards are not available locally).
- **Masking and special tokens.** [OBSERVED] `mask = token_bytes[y] > 0` zeroes the loss on targets with zero bytes: BOS `<|reserved_0|>` and the other three reserved tokens. Their bytes are 0 too, so they leave the denominator as well. The softmax still spans all 8,192 ids.
- **Packing and BOS.**
  - **Validation** [OBSERVED] (`prepare.py:276–341`): each document is encoded with a BOS prepended. Rows of 2,049 tokens are filled by best-fit, so every row starts with BOS. When no whole document fits, the shortest document is cropped and its remainder discarded. Inputs and targets are shifted by 1.
  - **Held-out** [OBSERVED] (`downstream_evaluator.py:503–524`): BOS-prefixed documents are concatenated into one stream and cut into consecutive, non-overlapping 2,049-token windows. Segments therefore start **mid-document, without BOS**. The first token of each window is never a target.
- **Validation length.** [DERIVED] `steps = EVAL_TOKENS // (B·2048)` with `EVAL_TOKENS = 40·524,288`. Every executed `DEVICE_BATCH_SIZE` is 16, 32 or 64 (252 frozen sources; Pair-A eval batches 64/32/16), and all of them divide 10,240. Every model is therefore scored on the **same 10,240 rows**, i.e. 20,971,520 positions, before excluding BOS targets.
- **Degenerate-model check.** [INFERRED] A near-uniform model scores 13 bits per non-special token, so BPB 3.2159 implies about 4.04 decoded bytes per scored validation token.

### 1.3 Tokenizer identity

- **Pair-A and Phase 1.** [OBSERVED] `tokenizer.pkl` `888c9fa8…` and `token_bytes.pt` `23517e05…`:
  - `study/contracts_2026/provenance/data_manifest.json`;
  - `SNAP:token_stream_verification.json:runtime_verification.tokenizer_files`, written on the phase-1 Pod on 2026-08-26;
  - every Pair-A manifest (`grep tokenizer_sha256 runs/…/provenance`).
- **Discovery era.** [OBSERVED] No tokenizer hash is recorded (no `experiment_events` payload carries one). The tokenizer was **rebuilt on each fresh Pod**: `R:PROJECT_STATE.md:586–588` (baseline Pod) and `:1120–1121` (seed-42 AutoPareto Pod: "trained the 8,192-token tokenizer").
- [INFERRED] **The tokenizers are functionally identical across Pods.** Checkpoints trained on different Pods still give coherent held-out BPB when evaluated on the 2026-08-13/14 Pod (`downstream_evaluation_runs` 1–13):
  - exp 4 (trained 2026-07-20): validation 1.2127, held-out 1.2625;
  - exp 77 (2026-07-27), exp 183 (08-06) and exp 300 (08-12) behave likewise.

  A tokenizer with different merges would permute the token ids and push BPB well above 3. Byte identity: [UNKNOWN].
- **Inconsistency.** [OBSERVED] `study/contracts_2026/provenance/tokenizer_manifest.json` (`status STATIC_BYTES_VERIFIED_RUNTIME_TOKEN_HASH_MISSING`) and `docs/RESEARCH_LEDGER.md:258` say `results/phase1/token_stream_verification.json` is missing. That file exists in `SNAP` with `status: RUNTIME_VERIFIED`, all seed stream hashes equal (`180e0a5b…`), and both accumulation regimes equal.

### 1.4 Shard identity and disjointness

- **Validation.** [OBSERVED] `shard_06542.parquet`, SHA `769fe59d…`, 91,699,537 bytes (`data_manifest.json`; pinned by `prepare.py:42–44`).
- **Held-out.** [OBSERVED] `shard_06541.parquet`, SHA `51da1a9b…` in all 32 `downstream_task_results` held-out rows and all 21 `checkpoint_evaluations.json` files. It is downloaded at revision `915333b4…` to `~/.cache/autopareto/downstream/` (`downstream_evaluator.py:480–500`), outside the training `DATA_DIR`. `prepare._document_batches` therefore can never pick it up.
- **Training.** [OBSERVED] Shards 00000–00009 only:
  - `data_manifest.json`;
  - `token_stream_verification.json:training_files`;
  - `PROJECT_STATE.md:586–587` ("10 training shards plus the pinned validation shard").
  - [DERIVED] Training, validation and held-out are file-disjoint. Overlap at the document level: [UNKNOWN].
- **Revision pinning.** [OBSERVED] Training and validation shards come from `…/resolve/main` (`prepare.py:41`). The paper's "Training and validation use … dataset revision `915333b4…`" (`paper/main.tex:135`) is enforced only for held-out and downstream data. Whether `main` equalled `915333b4` at each download: [UNKNOWN]. Discovery-era validation-shard bytes: [UNKNOWN] (re-downloaded per Pod, hash first recorded 2026-08-31).

### 1.5 Version drift between eras

[DERIVED] Each discovery-era protected digest maps to a commit whose `prepare.py` is 4f2ba9cb, so validation BPB was computed by byte-identical code from 2026-07-19 to 2026-08-12. Pair-A registered the same bytes. Phase 1 drifted only in the non-semantic `device` parameter. The held-out evaluator changed between the 300-s panel and phase 1 only in non-semantic ways. **No evaluator version drift affects any reported comparison.**

Two numerical paths do coexist for phase-1 validation [DERIVED]:

- the in-process value, stored in `experiments.val_bpb` for ids 301–321;
- the reloaded-checkpoint value, used in the paper.

They differ by 4×10⁻⁶ to 2.4×10⁻⁵ (`phase1_check.py`). The AP292−AR validation gap is +0.045073 by re-evaluation and +0.045083 in-process.

### 1.6 Are validation and held-out on comparable scales?

Same unit and same formula; **different data and different packing**. [DERIVED] (`SNAP:phase1_equal_token_results.json`; `downstream_task_results`):

| 201M panel (means, seeds 3–5) | val BPB | held-out BPB | held-out − val |
|---|---|---|---|
| AutoResearch (src exp 68) | 1.032661 | 1.099803 | 0.0671 |
| TPE (183) | 1.057199 | 1.119402 | 0.0622 |
| AP292 (src 134) | 1.077734 | 1.141837 | 0.0641 |
| MOTPE (234) | 1.079266 | 1.148253 | 0.0690 |
| Baseline (src exp 4) | 1.093438 | 1.146590 | **0.0532** |
| AP299 (137) | 1.097147 | 1.159309 | 0.0622 |
| AP294 (136) | 1.128379 | 1.190057 | 0.0617 |

- **201M panel.** [DERIVED] Validation ranks MOTPE ahead of the baseline by 0.0142. Held-out ranks the baseline ahead by 0.0017.
- **300-s downstream panel** (one checkpoint per model). [DERIVED] The gaps range from 0.0498 (exp 4) to 0.0689 (exp 289). AP292 (id 292) has the best validation BPB at 1.09394, while TPE (183) has the best held-out BPB at 1.15341.
- **Cause.** [INFERRED] Candidate explanations: (a) the non-BOS-aligned window starts, which models trained only on BOS-aligned rows never see, with sensitivity that depends on the architecture; (b) shard content; (c) cropping in validation, which over-represents short documents and document beginnings. Cause: [UNKNOWN] without GPU re-evaluation.
- **Implication.** Compare only within one metric. The paper's reversal sentence (`main.tex:66`) pairs held-out at 201M with discovery-time validation, but the reversal also holds **on the same validation shard** (+0.045073), so it does not depend on the metric switch [DERIVED].

---

## 2. Metric-to-checkpoint integrity

### 2.1 Published 300-s recipes (`SNAP` DB)

| Recipe | Row | Seed | Source SHA (frozen file = DB/selection) | Training run → checkpoint | Inference row | Verdict |
|---|---|---|---|---|---|---|
| Baseline | exp 4 | 1 | 428e22f1 (`baseline-a40-seed0/0002-train.py`) | tr 4 → `preserved_checkpoints/…/0002.pt`; downstream ck `d8d03b3e` | tr 4, re-measured 2026-07-22 (evaluator dfa298c4) | consistent [OBSERVED] |
| AR ref | exp 68 | 42 | b3a961ee | tr 22477 → `…diagnostic5…/0005.pt` (**weights not retained**) | tr 22477 | metrics consistent; checkpoint lost [OBSERVED] |
| TPE | 183 | 42 | 8547b278 | tr 71805 → `tpe…/0021.pt`, ck `c931ca26` | tr 71805 | consistent |
| MOTPE | 234 | 42 | 62882a20 | tr 82311 → `/tmp`-backed path (**weights lost**) | tr 82311 | metrics consistent; checkpoint lost |
| AP292 | 292 (src exp 134) | 1 | 55a7fd8b (×3 seeds = plan = phase 1) | tr 97988 → ck `e913acf3` | tr 97988 | consistent |
| AP299 | 299 (src 137) | 2 | 36e3ad91 | tr 100039 → ck `8a297f12` | tr 100039 | consistent |
| AP294 | 294 (src 136) | 0 | 6d0d8d1d | tr 98569 → ck `d5c29a5b` | tr 98569 | consistent |

- `confirmation_runs` ids 4–21 link each candidate to the right training run and seed [OBSERVED].
- Every confirmation and ablation experiment has **two** `training_runs` rows, one `confirmation`/`ablation` and one `discovery`, with identical values and checkpoint [OBSERVED]. For exps 70, 72, 73, 74 and 76 the two rows carry **conflicting statuses** (completed vs crash).
- **Exp 87** [OBSERVED]: its `val_bpb` comes from the 2026-07-28 run, while its checkpoint and speed/memory come from `attempt4-recreation/0004-recreated.pt` (2026-07-30). This is a cross-run metric pairing.

### 2.2 The 21 phase-1 runs

[OBSERVED/DERIVED; `phase1_check.py`] For every run:

- the seed agrees across `run_metadata`, `training_metrics`, the milestone manifest and the results JSON;
- the local frozen source hash = `training_metrics.source_sha256` = the model's `source_sha256` in the results JSON;
- the derived-source hash = the file hash = `checkpoint_evaluations.candidate_sha256` = `experiments.git_commit` (sic);
- the final checkpoint SHA is the same in `training_metrics`, the milestone manifest, `checkpoint_evaluations`, `phase1_200m_downstream_selection.json` and `downstream_evaluation_runs`;
- the `training_metrics`, `checkpoint_evaluations` and `final_inference` SHA-256s in the results JSON match the files;
- the held-out value in the results JSON = `checkpoint_evaluations` = `downstream_task_results`, bit-identical;
- tokens = 201,326,592 (384 × 524,288 for the baseline; 1,536 × 131,072 otherwise);
- **all 21 `final.pt` are in `R:.git/lfs/objects` and re-hash correctly**.

The model definitions in each `phase1_train.py` equal the frozen source except for added imports. Downstream evaluation, which loaded the *original* source, is therefore architecture-equivalent [DERIVED].

**Gaps:**
- `final_inference.json` carries no checkpoint hash; it is linked by path only (`scripts/run_phase1_equal_tokens.py:225–249`) [OBSERVED].
- The snapshot copy `scientific_final_checkpoints/phase1-autoresearch_compatible-seed4/final.pt` is truncated (3,014,656 bytes) [OBSERVED].
- Seed 3 ran on host `1ytt8c1e6pe6s8` and seeds 4–5 on `80e0b113f937` (`run_metadata.json`) [OBSERVED].
- The LR and weight-decay schedules follow token progress in all 21 runs (`phase1_train.py:584`; `protected/phase1_source_transform.py:145–150`). The Muon momentum ramp stays step-based (`frac = min(step/300, 1)`): it covers 78 % of the baseline's 384 steps but 20 % of the others' 1,536 steps. This is disclosed in `PROJECT_STATE.md` ("explicit remaining recipe difference") but **not in the paper**. [OBSERVED]

---

## 3. Checkpoint substitutions

**Why.** [OBSERVED]
- **AR reference.** Only four final checkpoints were kept on persistent storage: exps 4, 74, 77 and 80 (`R:PROJECT_STATE.md:1016–1020`; `RI:results/preserved_checkpoints/manifest.json`, role `experiment_68_median_confirmed`). The selection note reads "Best-BPB compatible run from the exp68 confirmation campaign" (`RI:results/downstream_selection.json`).
- **MOTPE.** The exp 234 weights were lost because `results/checkpoints` was a symlink to container-local `/tmp`, which a Pod stop cleared (`RI:results/preserved_checkpoints/motpe-seed42-20260805-manifest.json:cause`; `R:PROJECT_STATE.md:1573–1580`). Exp 300's description: "original experiment 234 weights were unavailable". Downstream run 11 for exp 300 followed two interrupted attempts (`RI:results/downstream_interruptions/run-0009/0010-experiment-300.json`: infrastructure shutdown, then a missing `tokenizer.pkl`).

**Identity.**

| | Source bytes | Seed | Protected training/eval code | Exposure (300 s) | val BPB |
|---|---|---|---|---|---|
| exp 68 → **exp 77** | identical, b3a961ee [OBSERVED] | 42 → **2** | manifests 0a9c3a07 (`95fb9cee`) vs 695066bc (`3799d11e`). Only `runner.py`/`autopareto_db.py` registry code and the `PROTECTED_FILES` list differ; `prepare.py`, `protected/train_launcher.py` and `inference_benchmark.py` are unchanged [DERIVED] | 88.87M vs 88.47M tokens | 1.09498 vs 1.10238 |
| exp 234 → **exp 300** | identical, 62882a20 [OBSERVED] | 42 = 42 | a77a908d (`940fe417`) vs d4366cef (`7813202a`). Only a `runner.py` checkpoint-storage guard differs [DERIVED] | 152.96M vs 154.93M tokens | 1.09811 vs 1.09751 |

Exp 77 is a different seed. Exp 300 is a nondeterministic re-run at the same seed with +1.3 % fixed-time exposure. Neither is the discovery model [DERIVED].

**Effect on the paper.** [OBSERVED] `paper/main.tex:68` (ARC-Easy 36.91/36.50, PIQA 58.29/58.67, LAMBADA 21.50/20.44, BLiMP 72.02/74.56) reproduces exactly from the **201M** rows (downstream runs 14–16 vs 23–25). For the 300-s rows the values are 36.57/36.83, 58.16/57.40, 20.90/19.64 and 71.39/70.49. No 300-s downstream or held-out number appears in `main.tex` or in any `\input` file: grep for 1.1617, 1.1649 and exp 77 found nothing. The 201M panel trained its own checkpoints from the byte-identical sources (b3a961ee, 62882a20), so neither substitution reaches it. Internal documents that describe the 11-model historical panel should label rows 2 and 11 as substitutes.

---

## 4. Re-verification of the key audit findings

| Finding | Verdict | Primary evidence |
|---|---|---|
| **(a)** Pair-A 3.2159 runs come from the `candidate_process.py` re-exec loop | **CONFIRMED, strengthened** | See (a) below |
| **(b)** Selection narrowed 11 → 4 → 3 | **CONFIRMED** (with date nuance) | See (b) below |
| **(c)** Seed-42 rollback: exp 88–90 deleted; 242 events deleted | **PARTIALLY CONFIRMED, undercounted** | See (c) below |
| **(d)** Historical agent identity is a runner default | **CONFIRMED** | See (d) below |
| **(e)** Fixed-time AP292 − AR = −0.0067 (seeds 0–2); 201M reversal +0.045 val / +0.042 held-out | **CONFIRMED, with caveats** | See (e) below |
| **(f)** Figure 1 width labels A07–A10 wrong | **CONFIRMED** | See (f) below |

**(a) Re-exec loop.** [OBSERVED/DERIVED]
- **Code, current version.** `study/contracts_2026/candidate_process.py:659–697` (`700937bb`) runs `while True: exec(training_code)` and leaves only on `CandidateStop`. A normal return re-runs setup and training.
- **Code, earlier version** [OBSERVED in bytecode]. `study/contracts_2026/__pycache__/candidate_process.cpython-310.pyc` (source size 35,471 bytes, mtime 2026-09-05 06:38 UTC) has the same control flow: offset 820 `JUMP_FORWARD`→1044 `JUMP_ABSOLUTE`→804 `SETUP_FINALLY`/`exec`.
  - [INFERRED] This is most likely the paper-lineage version `ca4d1eaa`, which the prior audit marked UNKNOWN. The paper-lineage manifests record it, the Python is 3.10, and it predates the 2026-09-06 freeze.
- **Signature.** The position-file signature `candidate_reported.step == 0 and optimizer_updates == 0` occurs in exactly **19/166** files and coincides with BPB > 3 (19/19 vs 0/147).
  - In those files the protected `optimizer_updates` are 169–1,267 (643–1,267 in the paper lineage).
  - Every healthy file has |step − updates| ≤ 1.
  - The 19 files are **8 unique runs**: the paper lineage's T3–T8 (positions 5, 7, 10, 11, 14, 16), `20260906-fresh` T2 and prestudy r10 T1.
- **New timing fingerprint.** In `protected_events`, the update boundary before the final update is 0.45–0.61 s slower than that run's median in all 19 files, and within ±0.01 s in all 147 healthy files. That delay matches the time to rebuild a model.
- The bf16 max logit error of 0.000244 is consistent with untrained logits.

**(b) 11 → 4 → 3.** [OBSERVED]
- **11.** `RI:results/autopareto_confirmation_selection.json` (commit `ea3b3d7`, 2026-08-05; SHA `72d58664…`) holds 11 representatives from 3 campaigns.
- **4.** `RI:results/autopareto_primary_confirmation_plan.json` (`cc756d1`, 2026-08-10; `source_selection_sha256` matches) keeps 124, 134, 136 and 137, all from the seed-0 campaign.
- **3.** The 201M panel uses only 134, 137 and 136; 124 (low memory) is dropped.
  - [OBSERVED] No mechanical rule for the drop is recorded. `R:PROJECT_STATE.md:163` says "human approval required before launch", and `evaluation/phase1_equal_token_preflight.json:prepared_on` is **2026-08-25**, after the 300-s confirmations (08-12) and the 300-s downstream results (08-13/14).
  - The paper (`main.tex:43`) lists only three roles.

**(c) Rollback.** [OBSERVED] `results/archive/safety-20260730-before-attempt5-recovery.sqlite` has `sqlite_sequence.experiments = 90` and `max(id) = 87`. `results/events/autopareto-seed42-20260727-0006/0007/0008.jsonl` hold 84, 74 and 84 events for ids **89, 90 and 91**, dated 2026-07-28T04:41–05:15Z.
- **Four rolled-back attempts, not three.** [DERIVED] Attempt 8 ran as id 91 on 07-28. Id 91 was later **reused** by the 2026-07-30 tokenizer-cache setup row, which is only possible if the restored DB never held the Jul-28 row 91. Mechanism: [UNKNOWN].
- **Event deletions** [DERIVED, from id gaps; AUTOINCREMENT ids are never reused]:
  - **255** events (ids 9186–9440, between event 9185 at 2026-07-28T04:23Z and event 9441 at 07-30T02:59Z) were deleted **before the earliest local snapshot**. Not in PROJECT_STATE.
  - **82** events (9462–9543) were deleted between the first two snapshots ("stale live-event rows … earlier attempt-5 record", `PROJECT_STATE.md:1259–1260`).
  - **242** events (9703–9786, 10063–10136, 10418–10501 = 84 + 74 + 84) were removed by the orphan cleanup (`PROJECT_STATE.md:1293–1295`). The prior audit's "242" is this last step only.
  - Total in the rollback window: **579**. Plus 20 earlier unexplained gaps, the final DB lacks 599 event ids.

**(d) Agent identity.** [OBSERVED] `RI:runner.py:1882–1883, 1902–1903` default to `--coding-model gpt-5.6-sol`, `--reasoning-level high`, as do the runner versions at commits 4c302e50, a2680be0, 75af7b7e, 3825cc34, b8eab289 and e69f79cd.
- The DB records `gpt-5.6-sol`/`high` for TPE, MOTPE, CMA-ES, NSGA-II and random search (125 rows), for confirmations (19), ablations (6) and setup rows. None of these involve an LLM.
- `agent_actions` has 67 rows, all for AutoResearch.

**(e) Numbers.** [DERIVED]
- **Fixed time.** Paired by seed (ids 291–293 minus 75–77): −0.005365, −0.005053, −0.009623; mean **−0.006680**, SD 0.00255.
- **Caveats on the fixed-time number:**
  - AR seed 1 (exp 76) is inference-incompatible. Without it the mean is −0.007494.
  - The AR and AP292 confirmations ran 15 days apart on different Pods. Same-seed baseline reruns 3 days apart moved by **+0.0050** on average (ids 8–12 vs 3–7, 1.6–3.1M fewer tokens), which is comparable in size.
  - Seed 0 is AP292's discovery seed.
- **201M panel.** Validation **+0.045073** (paired 0.0440, 0.0467, 0.0445). Held-out **+0.042034** (paired 0.0387, 0.0477, 0.0397). Both match `\BPBCost{}` = 0.042034.

**(f) Figure 1 labels.** [OBSERVED] `analysis/generate_figures.py:44` labels A7–A10 "width 384→320", "320→256", "256→320" and "320→384".
- The logs (`results/logs/autopareto-seed0-20260731-00NN.log`, `n_embd`) and sources (`DEPTH`, `ASPECT_RATIO`) give A6 384, A7 384, A8 256, A9 384 and A10 384. The parents are 119←118, 120←119, 121←120 and 122←121.
- The true changes are therefore: none (depth only), 384→256, 256→384 and none.
- A9 and A10 both have 19,661,000 parameters.
- The other width labels (A6, A18, A24, A25) are correct.

**Additional correction (Pair-A schedules).**
- **G3 is not schedule-confounded.** [OBSERVED] All 7 G3 (`20260909-new-a40`) fixed-token attempts compute `progress = contract_progress(CONTRACT_ID, …, N_STAR)`. That call returns `min(total_tokens/N_STAR, 1)` for contract N (`study/contracts_2026/study_runtime.py:24–36`; protected bridge `candidate_process.py:185–216, 397–406`), and the final protected progress value is 1.0 at N* = 85,458,944. Prior audit 05 §5 ("G3: none of 7") is wrong.
- **G4 is confounded.** [OBSERVED] 19 of 20 completed attempts use `min(total_training_time / TIME_BUDGET, 1.0)` and stop at N* after 232–296 s, against `program_fixed_tokens.md:12–14`. The exception is attempt 2.
- **The paper prefix (G2) is not affected.** [OBSERVED] N1–N8 use `min(total_tokens / N_STAR, 1.0)`.

---

## 5. Validity-flag overlay (`supporting/historical_validity_flags.json`)

**Schema.** Each flag has `record_locator`, `store`, `flag` (INVALID/CAVEAT), `reason`, `evidence`, `detected_by`, plus `evidence_status`, `category` and, where needed, `scope`. The overlay has **99 flags**:

| Category | INVALID | CAVEAT | Content |
|---|---|---|---|
| `pair_a_degenerate_3.2159` | 19 files (8 unique runs) | | re-exec artifact |
| `fixed_token_time_schedule` | 37 records (19 unique G4 N attempts) | | invalid only as Contract-N evidence (`scope`) |
| `seed42_rollback` | 4 (ids 88, 89, 90, Jul-28 91) | 10 | deleted event-id ranges, mixed JSONL, exps 87, 91, 92, 97 |
| `stale_running` | | 2 | `training_runs` 14973 and 16716 |
| `inference_incompatible` | | 6 | exps 64 (log-recovered BPB, truncated checkpoint), 65, 72, 73, 74, 76 |
| `artifact_corruption` | 2 | 1 | `baseline-a40-seed0/0003-train.py` (bytes 24064–26241 NUL; SHA a3c94aeb; prefix = 0001); truncated snapshot copy of AR seed-4 `final.pt`; exp 5 provenance |
| `provenance` | | 2 | upstream `228791fb` in `git_commit` for ids 91–112 and 118–162; tokenizer and shards per Pod, unhashed |
| `store_consistency` | | 8 | conflicting duplicate `training_runs`; `confirmation_runs` ids 1–3 absent; phase-1 column semantics and two validation values; 20 other deleted event ids |
| other | | 9 | path-only inference link; substitutions (runs 2, 11, interruptions 9–10, lost weights of 68 and 234); stale tokenizer manifest; G3 correction; Pod/seed confound in phase 1 |

**Scan for further corruption.** [OBSERVED] A NUL-byte scan of 13,706 text artifacts found only the baseline source; four literature `.txt` extractions have one NUL each and are irrelevant. `paper/generated/artifact_inventory.json:915–919` had already recorded the corrupt hash on 2026-09-06 without flagging it.

---

## 6. Remaining validity of claims

| Claim (paper location) | Status | Evidence |
|---|---|---|
| BPB is a fixed, protected, tokenizer-independent metric used identically across eras | VALID-WITH-CAVEAT | §1.1, §1.5. Same code in every era, but the bytes are decoded per-token lengths, not raw UTF-8 (§1.2) |
| Held-out BPB is "distinct from discovery validation BPB" (`main.tex:43`) | VALID | §1.6. It is more than distinct: the offset is recipe-dependent and rankings differ, which should be stated |
| Held-out on `shard_06541`, 2,560 segments, 5,242,880 tokens, SHA recorded (`:135`) | VALID | 32 DB rows plus 21 bundles, SHA `51da1a9b` |
| Training **and validation** use revision `915333b4` (`:135`) | UNVERIFIED | `prepare.py:41` downloads from `resolve/main`; no revision recorded (§1.4) |
| "Hashed project tokenizer" (`:135`) | VALID-WITH-CAVEAT | Hashed for Pair-A and phase 1 (`888c9fa8`). Discovery-era tokenizer rebuilt per Pod and unhashed; functionally equivalent [INFERRED] |
| 21 runs: source hash, exact 201,326,592 tokens, checkpoint-to-downstream pairing verified (`:105`) | VALID | §2.2, all 21 re-verified. Inference is linked by path, not hash |
| Seeds 3–5 disjoint from discovery and earlier confirmation seeds (`:43`) | VALID | `SNAP` JSON and DB |
| AP292 vs AR held-out cost 0.042034; 1.99× decode; −41.7 % params; −35.1 % memory (`:66`, macros) | VALID | Recomputed (§4e); hashes match `:133` |
| Discovery ordering reverses under matched exposure; H1 supported (`:66`) | VALID-WITH-CAVEAT | Holds on held-out (+0.0420) and on the same validation shard (+0.0451). The fixed-time side is −0.0067, with Pod and date confounds of about 0.005 and one inference-incompatible AR seed |
| Three AP recipes "selected before confirmation under the archive's primary roles" (`:43`, `:105`) | VALID-WITH-CAVEAT | Roles were fixed on 2026-08-05/10. The 4→3 panel cut (dropping exp 124) was set on 2026-08-25, after the 300-s results, with no recorded rule |
| Archive bests 1.0933/1.0950/1.0977/1.0981/1.1038/1.1077/1.1083 (`:41`) | VALID-WITH-CAVEAT | Values reproduce (ids 134, 68, 183, 234, 247, 275, 202). The AR "best completed" is exp 68, not AR's own retained exp 65 (incompatible) |
| Downstream ARC-E/PIQA/LAMBADA/BLiMP AP292 vs AR (`:68`) | VALID | Exact from 201M rows; the exp 77/300 substitutions are not used (§3) |
| Figure 1 lineage and edit labels (`:50–54`) | INVALID (labels A7–A10); VALID (path, BPB values 1.0966→1.1573→1.0957→1.0933) | §4f |
| Fixed-time prefix: "4 completed attempts", `\TimeBest` 1.103402 (Table 2, `:68`) | VALID-WITH-CAVEAT | T3 and T4 of the 4 are harness-invalid (§4a). Only T1 and T2 are informative; T4's edit responded to corrupted feedback |
| Table 2: "LR progress: counted-token fraction" for N (`controls_table.tex`) | VALID-WITH-CAVEAT | True for the G2 prefix because agent N1 added it, not because the harness enforced it. False for G4 |
| Agent = Codex CLI 0.153.0, `gpt-5.6-sol`, high (`:82`, `:141`) | VALID for Pair-A; UNVERIFIED for historical campaigns | §4d: the historical label is a runner default |
| Matched exposure differs from discovery only in exposure, seed and evaluation data (`:114`) | VALID-WITH-CAVEAT | Also: tokens-based vs time-based LR progress; a step-based momentum ramp (78 % vs 20 % of steps for the baseline); Pod/host confounded with seed 3 (§2.2) |
| "On one A40" for the panel | VALID-WITH-CAVEAT | Two Pods/hosts (seed 3 vs seeds 4–5) |
| Historical record completeness ("preserves crashes … rather than dropping them", `:105`) | INVALID for seed 42 | Four attempts and 579 events deleted (§4c); exp 87 metrics come from two different training runs |

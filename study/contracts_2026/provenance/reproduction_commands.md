# Reproduction commands and boundaries

These commands are deliberately split by evidence regime. The audit itself ran only
read-only inspection and hash/count checks; it did not train, benchmark, launch a
campaign, or modify N_STAR.

## Read-only aggregate verification

From the repository root:

```bash
shasum -a 256 docs/RESEARCH_LEDGER.md
shasum -a 256 study/contracts_2026/provenance/final_pre_study_frozen_manifest_20260901.json
shasum -a 256 study/contracts_2026/provenance/n_star_freeze_v3_20260902.json
python3 scripts/verify_release.py
python3 scripts/generate_release_plots.py
```

The last two commands regenerate/check public aggregate artifacts only. They do not
launch training.

## Inspect the recovered registry read-only

```bash
python3 - <<'PY'
import sqlite3
p = 'results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db'
c = sqlite3.connect('file:' + p + '?mode=ro', uri=True)
for (name,) in c.execute("select name from sqlite_master where type='table'"):
    print(name, c.execute('select count(*) from "' + name + '"').fetchone()[0])
c.close()
PY
```

## Verify static data/tokenizer manifests

```bash
python3 study/contracts_2026/tools/freeze_data_manifest.py --help
```

Use the existing canonical `study/contracts_2026/provenance/data_manifest.json` and
its aggregate hash. Runtime token-ID verification must not be claimed until the
missing `results/phase1/token_stream_verification.json` is restored.

## Gated scientific commands

The following are documentation references only and were not run in this audit:

```bash
python3 scripts/run_phase1_equal_tokens.py --stage B --execute
python3 study/contracts_2026/runner.py --help
```

Any future training/campaign invocation requires the frozen protocol, current
isolation preflight, appropriate hardware, and explicit campaign authorization. Do
not use this audit document as Scientist context; the campaign programs define the
sealed agent context.

## Independent reproduction limitations

The public checkout alone does not include the complete private harness, all raw
logs, the populated database at its canonical path, weights, private caches, or the
runtime token-stream hash output. Reproduction claims must be scoped accordingly.

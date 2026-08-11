# Release provenance

## Release scope

This is the initial public discovery-evidence release. It contains reviewed result exports, named-candidate parameter snapshots, static plots, and protocol documentation. It does not contain the raw SQLite evidence ledger, checkpoints, the autonomous-search implementation, or any unfinished evaluation output.

## Source evidence

The exports were produced from the private research repository at source commit:

`cc756d112006d5abf03205ea6be9c562b887f9aa`  
`Freeze discovery evidence and confirmation plan`  
`2026-08-10`

The public release begins from database counts of 284 experiment records, 302 training runs, 2,784 inference measurements, and 233 deployment measurements. The 3,017 inference-plus-deployment rows are measurements, not independent training runs.

## File hashes

SHA-256 hashes at release preparation:

| File | SHA-256 |
|---|---|
| `data/experiment_records.csv` | `851a32ef3376af950f16ca5620043418974b4664a59cf0507e646ffa7ae953ae` |
| `data/inference_measurements.csv` | `9ea74b42a44cf43a7ad0c788291875b8cca94f2d08e8bd44057c1f10310ef7df` |
| `data/deployment_measurements.csv` | `ab5fff9f46f2d46ca25370d290b2cfd4efbb8260e7582647c96ddf377b4eaf87` |
| `configs/manifest.json` | `906351590e8db39efc81c103e854a6cd2cf6cada0cc0ece91972a09c991f2953` |

Run `python3 scripts/verify_release.py` to validate row counts, headline rows, configuration references, and the public claim boundaries checked by this release.


# Phase 5 evidence (5A: Opus frontier at 201,326,592 tokens; 5B: inference ablations)

Pre-registration: `evaluation/PHASE5_PREREG.md` and the two plans (commit acc86d7). Audit record:
`audit/camera_ready_2026/24_PHASE5_OPUS_MATCHED_TOKENS_AND_INFERENCE_ABLATIONS.md`.

| Path | Contents |
|---|---|
| `A/runs/` | 10 training runs (9 Opus + the AR-ref drift control): records, logs, step timing, telemetry, mapped sources |
| `A/downstream/` | frozen downstream suite on 15 checkpoints (9 Opus deployed + 6 E1 final) |
| `B/runs_json.tar.xz` | the 39 raw protected-benchmark JSONs (`runs/NN_id.json`, 60 MB uncompressed) |
| `B/runs/*.log` | per-run benchmark logs |
| `B/contexts/`, `B/kernel_census/`, `B/driver.log` | generated variant contexts, kernel census, driver log |
| `workspace_logs/` | chain script, console log, launch time |
| `analysis/` | `phase5_analysis.json`, `PHASE5_TABLES.md`, `phase5_tables.tex`, figures |
| `SHA256SUMS` | all 337 evidence files as they were on the pod (including the archived JSONs) |
| `CHECKPOINTS.SHA256SUMS` | the 76 checkpoints (4.7 GB, kept off Git in verified Mac storage) |

Not in Git: the checkpoints and `A/downstream/phase5a_downstream.db` (the repository ignores `*.db`). Both are listed
in the checksum files and kept in the verified Mac backup. The analysis does not read the database.

Verify and reproduce:

    cd results/camera_ready_2026/phase5
    mkdir -p /tmp/p5 && tar -xJf B/runs_json.tar.xz -C /tmp/p5
    (cd /tmp/p5 && grep ' B/runs/.*\.json$' "$OLDPWD/SHA256SUMS" | sed 's# B/runs/# runs/#' | shasum -a 256 -c -)
    cd ../../.. && python3 scripts/analyze_phase5.py --out /tmp/p5_analysis

`analyze_phase5.py` reads `B/runs/*.json` when present and the archive otherwise; both give byte-identical
`phase5_analysis.json`, `PHASE5_TABLES.md` and `phase5_tables.tex` (checked 2026-11-09).

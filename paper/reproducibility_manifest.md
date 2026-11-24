# Internal audit: not an anonymized submission asset

# Reproducibility manifest

Analysis checkout HEAD: `6bcb69e6036d64bb0eca9b5b07d0b0fde1c58abf`. Existing uncommitted study code is scientifically material; HEAD alone does not identify executed code. Artifact SHA-256 inventory is generated/artifact_inventory.json. No experiment source was executed in the paper build.

## One-command build

```sh
./paper/refresh_paper.sh --cutoff 2026-09-06T23:59:00+12:00
```

Use --inventory to rehash the full input collection. The script never launches training or benchmarks. It emits headline_changes.md and preserves prior headline values. Python dependencies are pinned in paper/requirements.txt; Tectonic or latexmk is required.

## Training and evaluation sources

| Component | Exact source |
| --- | --- |
| Historical raw panel | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1 |
| Most complete DB | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| Historical training launcher | scripts/run_phase1_equal_tokens.py; protected/train_launcher.py |
| Historical equal-token manifest | evaluation/phase1_public_manifest.json |
| Token stream verification | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/token_stream_verification.json |
| Dataset and shards | study/contracts_2026/provenance/data_manifest.json; evaluation/downstream_suite.json |
| Tokenizer | study/contracts_2026/provenance/tokenizer_manifest.json |
| Current executed manifest | runs/pair-a-official-20260906-fixed-warmup-v2/provenance/final_pre_study_manifest_pair_a_20260906_fixed_warmup_v2.json |
| Current launch invocations | runs/pair-a-official-20260906-fixed-warmup-v2/*/scientist_diagnostics/*/invocation.json; controller_state.json |
| Current initial exact inputs | paper/audit/prompts/{T,N}/* |
| Current prompt diff | paper/audit/executed_program_md.diff |
| Historical source commands | scripts/run_phase1_equal_tokens.py; scripts/evaluate_phase1_checkpoints.py |
| Historical downstream settings | evaluation/downstream_suite.json; downstream_evaluator.py |
| Historical throughput | final_inference.json: 256 input / 256 generation; median over 30 repetitions, stabilization blocks persisted |
| Current throughput | eight prompts; 1024 input / 128 generation / 127 cached transitions; 8 baseline-candidate-baseline brackets |
| Memory | peak allocated inference memory; MiB = bytes / 1048576. Reserved and KV kept distinct. |
| Parameter/architecture | static source AST constants; parameter_count measured in training record |
| Historical environment | per-run run_metadata.json and final_inference.json GPU telemetry |
| Latest environment | executed manifest: Python3.12; torch2.9.1+cu128; CUDA12.8; one A40 |
| Cutoff | metadata completion time where present; conservative first-audit observation bound otherwise |

## Reproduction limits

No bare experiment ID is globally unique across budget relaunches. Snapshot DB duplicates are never counted as new trials. Source files and registered manifests can disagree; executed request bytes and result provenance take priority. Authoritative UTC completion timestamps are missing in the budget histories; new rows first observed after cutoff without timestamp evidence are excluded; first-observation timestamps persist by full record hash. GPU UUIDs and local paths are retained privately and excluded from the PDF. Historical raw individual inference repetitions are not present in every per-run bundle; final medians and pass summaries are retained, and the paper does not claim a complete independent timing replay.

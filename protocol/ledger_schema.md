# Ledger schema

The internal evidence ledger is SQLite. This release publishes CSV exports rather than the operational database. The public schema is described here so the exports can be interpreted without the database.

## Core entities

| Entity | Purpose |
|---|---|
| `campaigns` | Search method, phase, attempt limit, status, training seed, frozen program metadata, and baseline linkage. |
| `experiments` | One candidate attempt: status, source/evaluator metadata, hypothesis, parentage, training result, headline inference summary, and decision. |
| `training_runs` | Individual retraining executions associated with an experiment. |
| `inference_measurements` | Repeated inference measurements with workload dimensions, timing, memory, KV-cache, and correctness fields. |
| `deployment_measurements` | Workload-characterization measurements across context lengths and batch sizes. |
| `agent_actions` | Recorded agent proposals and decisions. |
| `experiment_events` | Append-only operational events used for audit and recovery. |
| `confirmation_runs` and `ablation_runs` | Follow-up studies separated from discovery evidence. |
| `downstream_evaluation_runs` and `quantization_runs` | Reserved tables for the frozen but currently unrun evaluation suites. |

## Public exports

`data/experiment_records.csv` contains the summary fields necessary to reconstruct the discovery and control plots. `data/inference_measurements.csv` and `data/deployment_measurements.csv` contain repeated measurements and workload columns.

The release contains 284 experiment records, 302 training runs, 2,784 inference measurements, and 233 deployment measurements. The 3,017 inference-plus-deployment rows are not 3,017 independent model trainings.


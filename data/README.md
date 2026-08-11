# Data

These CSV files are public exports from the audited `results/results.db` ledger used for this release. They are sufficient to reproduce the plots in `/plots` and the tables in the repository README.

| File | Rows | Contents |
|---|---:|---|
| `experiment_records.csv` | 284 | One row per recorded experiment, including discovery candidates, baselines, controls, confirmations, ablations, and failures. |
| `inference_measurements.csv` | 2,784 | Cached-decode and other inference repetitions, including context length, generated tokens, batch size, bfloat16 dtype, warm-up flag, memory, and correctness fields. |
| `deployment_measurements.csv` | 233 | Workload-characterization repetitions, including prefill, decode, allocated and reserved memory, KV-cache fields, and correctness status. |

All published performance measurements use one NVIDIA A40 and bfloat16. The README names the workload for every headline value: 256 input tokens, 256 generated tokens, batch size 1, and 10 measured cached-decode repetitions. Deployment rows use their own explicit workload columns and should not be substituted for that headline measurement.

The raw SQLite ledger is intentionally not included in this first results release. It contains operational records and will be released with the full code and paper.


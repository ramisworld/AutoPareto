# Camera-ready E0 evidence: two separate sets

Both sets were copied back from the fresh A40 pod (`GPU-4702993a-fd40-d985-292e-6398bd236183`) and are
hash-identical to the pod copies. Training checkpoints (`*.pt`) and compile caches stay on the pod.
Interpretation: `audit/camera_ready_2026/14_E0_RESULT.md`. No file in either set is ever edited; they
are committed read-only.

| Set | Plan | Verdict | Files |
|---|---|---|---|
| `E0/` + `camera_ready_E0_*` (top level) | Original `evaluation/camera_ready_e0_e1_plan.json`, sha256 `806a6397…c829340` | **HALT**: run 2 made 666 updates, outside [669, 681]. `e0_verdict.json` sha256 `bd953480…f8d9`. | 32 + 3 logs |
| `amendment1/E0/` + `amendment1/camera_ready_E0_amendment1_*` | Amendment 1, `evaluation/camera_ready_e0_e1_plan_amendment1.json`, sha256 `89f29015…60dc` (PI-approved, post hoc: run 2's update count is diagnostic) | **PASS**. `e0_verdict.json` sha256 `517750d6…d91e`. Run 2 made 668 updates, which v1 would also have halted. | 32 + 3 logs |

The two `data_identity.json` records differ only in `recorded_utc`; both pass on the same token
stream. Each set also has an empty `gpu.lock`, left by the executor's GPU lock.

The executed-code `__pycache__/*.pyc` files (3 per set) stay on disk and on the pod; `.gitignore` keeps them out of git.

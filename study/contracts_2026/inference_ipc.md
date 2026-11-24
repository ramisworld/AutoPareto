# Protected inference IPC boundary

`study/contracts_2026/inference_benchmark.py` is the trusted evaluator. It
starts one disposable `inference_model_host.py` process for BF16 correctness and
benchmark execution and a second one for FP32 adjudication. Communication uses
an anonymous Unix `socketpair`; no TCP socket or inherited network route is
used.

The trusted parent owns the fixed token-ID prompts, checkpoint/source pairing,
correctness comparisons, host monotonic timing, decode-block aggregation,
bootstrap interval, stability status, memory field selection, and Pareto
decision inputs. Candidate code runs only in the model-host process and cannot
inspect the evaluator's Python frames, objects, campaign history, or acceptance
state. The host returns observations only; malformed, non-finite, out-of-order,
or schema-invalid replies fail closed.

The protected Design-B semantic workload is unchanged: batch 1, BF16, eight
fixed 1024-token prompts, prefill-produced `y1`, and 127 cached decode
transitions. The unavoidable operational change is that the authoritative host
clock surrounds the parent-to-model-host decode RPC. Consequently the reported
decode interval includes the anonymous IPC request/response overhead. This is
recorded explicitly in `timing_boundary.ipc_boundary` and is not silently
treated as the historical in-process timing. CUDA-event duration is diagnostic
only.

Two limitations remain explicit. Ordinary Linux processes cannot hide token IDs
from the process that must execute the model, so the model host can observe the
delivered benchmark tensors. Also, CUDA allocator counters belong to the model
host's CUDA context; the parent validates their type and uses them for the
registered metric, but cannot hardware-attest them against a malicious CUDA
program without restricting the research space or using confidential GPU
hardware. The study threat model therefore excludes malicious hardware/CUDA
attacks while preventing evaluator-state and result-arithmetic forgery.

# Opus 5.5 Contract-T arm: failure and compute ledger

Generated 2026-11-06T02:40:39Z by `scripts/campaign_compute_ledger.py` from durable records (read-only).

- Attempts completed: **27**; valid results: **25**; scientific failures: attempts [2, 9].
- Frozen 25-attempt result: 23 valid of 25 completed scheduled attempts; failed [2, 9]. Extension attempts: [26, 27].
- 5 Scientist call(s) timed out. A40 receipts from the first pod (attempt 1's run) were lost with that pod; attempt 1's evaluation time comes from its result file.
- Scientist calls: 63 (54 used, 9 extra, 0 in progress); 383.5 min in total, 86.5 min extra, 33.3 min in extension attempts.
- A40 invocations with a durable receipt: 28 (26 counted, 2 extra); 367.9 wall-min, 6.6 wall-min extra, 36.1 wall-min in extension attempts.

## Attempts

| Attempt | Pos | Ext | Outcome | BPB | Decode × | Mem MB | Parent | Category | Sci calls used/extra | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 |  | completed | 1.17909 | 1.000 | 353.6 | baseline-calibration-new-a40-20260914-run1 | optimization_batch_size | 2/0 | retained |
| 2 | 4 |  | scientific_failure |  |  |  | attempt-01 | architecture_depth | 2/0 | CandidateExitedBeforeStop: candidate training body returned before the protected stop |
| 3 | 5 |  | completed | 1.16222 | 1.327 | 303.2 | attempt-01 | architecture_depth | 2/0 | retained; 1 transfer-drop requeue(s) |
| 4 | 7 |  | completed | 1.12731 | 1.931 | 255.0 | attempt-03 | architecture_depth | 2/0 | retained |
| 5 | 10 |  | completed | 1.11766 | 2.486 | 231.9 | attempt-04 | architecture_depth | 2/0 | retained |
| 6 | 11 |  | completed | 1.13912 | 3.644 | 201.5 | attempt-05 | architecture_depth | 2/0 | retained |
| 7 | 14 |  | completed | 1.09953 | 2.474 | 231.9 | attempt-05 | optimization_batch_size | 2/0 | retained |
| 8 | 16 |  | completed | 1.09949 | 2.571 | 177.5 | attempt-07 | inference_adapter_memory | 2/0 | retained |
| 9 | 18 |  | scientific_failure |  |  |  | attempt-08 | inference_adapter_decode_speed | 2/6 | RuntimeError: candidate cache_nbytes reported 7864320 bytes but protected state traversal found 7907344 bytes; 3 Scientist timeout requeue(s) |
| 10 | 20 |  | completed | 1.09995 | 8.617 | 450.2 | attempt-08 | inference_adapter_decode_speed | 2/0 | retained |
| 11 | 22 |  | completed | 1.09983 | 8.426 | 163.2 | attempt-10 | inference_adapter_memory | 2/0 | retained |
| 12 | 23 |  | completed | 1.11307 | 8.595 | 163.2 | attempt-11 | optimization_batch_size | 2/0 | quality_valid_speed_uncertain |
| 13 | 26 |  | completed | 1.09977 | 10.150 | 135.5 | attempt-11 | inference_weight_storage_memory | 2/0 | retained |
| 14 | 27 |  | completed | 1.09985 | 16.769 | 135.5 | attempt-13 | inference_adapter_decode_speed | 2/0 | retained |
| 15 | 29 |  | completed | 1.10242 | 17.088 | 135.5 | attempt-14 | optimization_learning_rate | 2/0 | quality_valid_speed_uncertain |
| 16 | 31 |  | completed | 1.10687 | 16.932 | 112.8 | attempt-14 | architecture_attention_kv_heads_memory | 2/0 | quality_valid_speed_uncertain |
| 17 | 33 |  | completed | 1.09915 | 17.016 | 127.0 | attempt-14 | architecture_attention_window | 2/0 | retained |
| 18 | 35 |  | completed | 1.09965 | 24.739 | 122.6 | attempt-17 | inference_kv_cache | 2/2 | quality_valid_speed_uncertain; 1 Scientist timeout requeue(s) |
| 19 | 38 |  | completed | 1.09911 | 29.572 | 122.6 | attempt-18 | inference_decode_attention_kernel | 2/1 | quality_valid_speed_uncertain; 1 Scientist timeout requeue(s) |
| 20 | 40 |  | completed | 1.09690 | 27.567 | 122.6 | attempt-19 | architecture_attention_temperature | 2/0 | retained |
| 21 | 42 |  | completed | 1.09751 | 27.602 | 118.5 | attempt-20 | inference_prefill_logits_memory | 2/0 | retained |
| 22 | 44 |  | completed | 1.09759 | 32.002 | 118.5 | attempt-21 | inference_decode_kernel_fusion | 2/0 | quality_valid_speed_uncertain |
| 23 | 46 |  | completed | 1.09620 | 30.840 | 118.5 | attempt-22 | architecture_embedding_token_shift | 2/0 | retained |
| 24 | 47 |  | completed | 1.09946 | 31.403 | 101.6 | attempt-23 | architecture_value_embedding_sharing_memory | 2/0 | retained |
| 25 | 50 |  | completed | 1.13301 | 29.886 | 118.5 | attempt-23 | architecture_attention_output_gate | 2/0 | dominated |
| 26 | 51 | yes | completed | 1.09622 | 32.245 | 116.4 | attempt-23 | inference_prefill_attention_memory | 2/0 | quality_valid_speed_uncertain |
| 27 | 52 | yes | completed | 1.09623 | 30.535 | 103.9 | attempt-26 | inference_embedding_table_int8_storage_memory | 2/0 | retained |

## A40 invocations

| Attempt | Invocation | Remote state root | Status | Wall min | Use | Failure |
|---|---|---|---|---|---|---|
| 2 | `pair-a-004-ccb492e33…` | `pair_a_claude_opus55_t_seed3` | FAILED | 0.0 | extra | candidate worker failure origin is unknown; manual reconciliation is required |
| 2 | `pair-a-004-4b7202a1a…` | `pair_a_claude_opus55_t_seed3_continuation_20261105b` | COMPLETE | 5.8 | counted |  |
| 3 | `pair-a-005-d1b4ebe86…` | `pair_a_claude_opus55_t_seed3_continuation_20261105b` | COMPLETE | 11.2 | counted |  |
| 4 | `pair-a-007-804de455b…` | `pair_a_claude_opus55_t_seed3_continuation_20261105b` | COMPLETE | 10.9 | counted |  |
| 5 | `pair-a-010-a1d265bd6…` | `pair_a_claude_opus55_t_seed3_continuation_20261105b` | COMPLETE | 10.8 | counted |  |
| 6 | `pair-a-011-7e953fb57…` | `pair_a_claude_opus55_t_seed3_continuation_20261105b` | COMPLETE | 10.5 | counted |  |
| 7 | `pair-a-014-39873589e…` | `pair_a_claude_opus55_t_seed3_continuation_20261105b` | COMPLETE | 10.7 | counted |  |
| 8 | `pair-a-016-f6b218a79…` | `pair_a_claude_opus55_t_seed3_continuation_20261105b` | FAILED | 6.6 | extra | candidate worker failure origin is unknown; manual reconciliation is required |
| 8 | `pair-a-016-7368fbd33…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 10.7 | counted |  |
| 9 | `pair-a-018-42202d14b…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | FAILED | 6.6 | counted | candidate worker failure origin is unknown; manual reconciliation is required |
| 10 | `pair-a-020-eb94870ce…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 12.1 | counted |  |
| 11 | `pair-a-022-df6f09ce8…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 12.0 | counted |  |
| 12 | `pair-a-023-0ab6c88d1…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 12.0 | counted |  |
| 13 | `pair-a-026-a006d17e8…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 12.3 | counted |  |
| 14 | `pair-a-027-715738fe5…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 14.9 | counted |  |
| 15 | `pair-a-029-e2bee9d18…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 14.2 | counted |  |
| 16 | `pair-a-031-fea06ab6b…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 14.6 | counted |  |
| 17 | `pair-a-033-f9f103fb5…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 14.1 | counted |  |
| 18 | `pair-a-035-3d220b8ad…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 16.3 | counted |  |
| 19 | `pair-a-038-5e5be2881…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 17.6 | counted |  |
| 20 | `pair-a-040-acb6c4bde…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 17.4 | counted |  |
| 21 | `pair-a-042-f359cc70a…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 17.2 | counted |  |
| 22 | `pair-a-044-d1032dad9…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 18.9 | counted |  |
| 23 | `pair-a-046-89dda8b37…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 18.8 | counted |  |
| 24 | `pair-a-047-4ecdfe2b5…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 17.1 | counted |  |
| 25 | `pair-a-050-4ebb4623c…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 18.1 | counted |  |
| 26 | `pair-a-051-98a4b01db…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 17.4 | counted |  |
| 27 | `pair-a-052-657c9c13c…` | `pair_a_claude_opus55_t_seed3_continuation_20261105c` | COMPLETE | 18.7 | counted |  |

## Extra Scientist calls

| Attempt | Phase | Status | Min | Started | Reason |
|---|---|---|---|---|---|
| 9 | select_parent | COMPLETE | 9.0 | 2026-11-05T09:33:45Z | completed, superseded by a later call after a requeue |
| 9 | edit_child | FAILED | 10.0 | 2026-11-05T09:42:48Z | Scientist execution exceeded its bounded timeout |
| 9 | select_parent | COMPLETE | 4.7 | 2026-11-05T10:27:21Z | completed, superseded by a later call after a requeue |
| 9 | edit_child | FAILED | 10.0 | 2026-11-05T10:32:03Z | Scientist execution exceeded its bounded timeout |
| 9 | select_parent | COMPLETE | 4.7 | 2026-11-05T10:56:03Z | completed, superseded by a later call after a requeue |
| 9 | edit_child | FAILED | 10.0 | 2026-11-05T11:00:46Z | Scientist execution exceeded its bounded timeout |
| 18 | select_parent | COMPLETE | 8.0 | 2026-11-05T16:58:48Z | completed, superseded by a later call after a requeue |
| 18 | edit_child | FAILED | 15.0 | 2026-11-05T17:06:49Z | Scientist execution exceeded its bounded timeout |
| 19 | select_parent | FAILED | 15.0 | 2026-11-05T20:34:51Z | Scientist execution exceeded its bounded timeout |

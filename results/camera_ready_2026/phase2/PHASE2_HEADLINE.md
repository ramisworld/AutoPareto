# Phase 2 headline comparison (generated; do not edit by hand)

| Model | Budget | Pair-A val BPB | Held-out BPB | LAMBADA | BLiMP | PIQA | ARC-E | Decode speedup (stable) | E2E speedup | Peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| Opus 5.5 attempt 23 | 300 s (197M tok) | 1.0962 | 1.158 | 0.231 | 0.711 | 0.572 | 0.349 | 31.8× (2/3) | 4.6× | 113 |
| Opus 5.5 attempt 24 | 300 s (198M tok) | 1.0995 | 1.161 | 0.219 | 0.722 | 0.577 | 0.345 | 30.9× (3/3) | 5.3× | 97 |
| Opus 5.5 attempt 27 | 300 s (198M tok) | 1.0962 | 1.158 | 0.228 | 0.712 | 0.589 | 0.372 | 31.5× (1/3) | 5.3× | 99 |
| Pair-A baseline | 300 s (85M tok) | 1.2011 | 1.249 | 0.170 | 0.652 | 0.573 | 0.348 | 1× (2/3) | 1× | 337 |
| AutoResearch | 300 s (88M tok) | — | 1.168 | 0.203 | 0.702 | 0.560 | 0.353 | — | — | — |
| AutoPareto AP292 | 201M tok, seeds 3–5 | — | 1.142 ± 0.006 | 0.215 ± 0.019 | 0.720 ± 0.004 | 0.583 ± 0.003 | 0.369 ± 0.004 | — | — | — |
| AutoResearch | 201M tok, seeds 3–5 | — | 1.100 ± 0.002 | 0.204 ± 0.009 | 0.746 ± 0.006 | 0.587 ± 0.014 | 0.365 ± 0.014 | — | — | — |
| Phase-1 baseline | 201M tok, seeds 3–5 | — | 1.147 ± 0.005 | 0.205 ± 0.006 | 0.714 ± 0.008 | 0.574 ± 0.004 | 0.353 ± 0.015 | — | — | — |

Fixed-time rows: one training run each (Opus and Pair-A baseline: Pair-A harness, seed 3; AutoResearch: Phase-1 harness, seed 0). Matched-token rows: mean ± sd over seeds 3–5 (August panel, reused). Pair-A val BPB is the search objective (protected evaluator), re-measured on pod4. Held-out BPB and the four tasks use the frozen downstream suite (evaluator 696f9c46, suite 3dce2c73). Single-run 95% intervals: held-out BPB ±0.007, LAMBADA ±0.012, BLiMP ±0.003 (micro), PIQA ±0.023, ARC-E ±0.019. Speedups are decode-only (bracketed, vs the paired Pair-A baseline, median of 3 repeats; stable = normalized drift ≤ 0.02) and end-to-end (1,024-token prefill + 128 generated tokens, vs the baseline recipe measured as a candidate on the same pod). Peak = peak allocated CUDA memory during inference.


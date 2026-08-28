# Phase 1 equal-token results

All values are from three preregistered seeds at exactly 201,326,592 training tokens. Timing is A40-only. Decode throughput is the median of 30 repetitions that passed the frozen adaptive stabilization and cache-correctness protocol.

| Model | Held-out BPB mean ± SD | Steady-state-equivalent s mean ± SD | Complete scientific wall s mean ± SD | Stabilized decode tok/s mean ± SD |
|---|---:|---:|---:|---:|
| baseline | 1.146590 ± 0.004673 | 688.50 ± 0.73 | 809.69 ± 13.58 | 196.82 ± 2.45 |
| autoresearch_compatible | 1.099803 ± 0.001943 | 696.16 ± 0.63 | 819.53 ± 5.03 | 192.93 ± 1.38 |
| tpe_quality | 1.119402 ± 0.004271 | 540.53 ± 0.48 | 665.25 ± 30.04 | 258.89 ± 3.42 |
| motpe_quality | 1.148253 ± 0.003102 | 398.23 ± 0.03 | 538.86 ± 4.60 | 270.01 ± 5.40 |
| autopareto_292 | 1.141837 ± 0.006035 | 385.02 ± 0.22 | 512.61 ± 32.34 | 383.02 ± 4.90 |
| autopareto_299 | 1.159309 ± 0.000370 | 312.78 ± 0.39 | 446.53 ± 21.16 | 488.26 ± 8.30 |
| autopareto_294 | 1.190057 ± 0.002179 | 295.68 ± 6.45 | 413.83 ± 26.29 | 495.49 ± 3.95 |

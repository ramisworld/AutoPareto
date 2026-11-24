# E1: token exposure versus architecture (EXPLORATORY decomposition)

Not pre-registered. The pre-registered verdicts are in `../analysis/e1_analysis.md`.
Plan sha256 `89f2901588fc…`; 24 of 24 runs selected; excluded or missing: none.
Cross-check of paired deltas against the pre-registered analysis: {'prereg_analysis': 'results/camera_ready_2026/amendment1/E1/analysis/e1_analysis.json', 'present': True, 'max_abs_difference': 0.0, 'passed': True}.

## Table A. BPB by recipe and budget (mean ± SD over seeds 6–8) and paired Δ = AP-Q − AR-ref (mean [95% t-CI])

| Budget | Tokens AR-ref / AP-Q | AR-ref held-out | AP-Q held-out | Δ held-out | Δ validation | Seeds with Δ<0 |
|---|---|---|---|---|---|---|
| 88.6M tokens (676 updates) | 88.6M / 88.6M | 1.1550 ± 0.0043 | 1.1935 ± 0.0038 | +0.0385 [+0.0222, +0.0547] | +0.0418 [+0.0343, +0.0493] | 0/3 |
| 157.3M tokens (1,200 updates) | 157.3M / 157.3M | 1.1152 ± 0.0039 | 1.1548 ± 0.0045 | +0.0396 [+0.0226, +0.0565] | +0.0430 [+0.0341, +0.0519] | 0/3 |
| 201.3M tokens (1,536 updates) | 201.3M / 201.3M | 1.0995 ± 0.0042 | 1.1410 ± 0.0036 | +0.0415 [+0.0287, +0.0544] | +0.0445 [+0.0364, +0.0527] | 0/3 |
| 300 s fixed time | 88.0M / 158.1M | 1.1569 ± 0.0042 | 1.1547 ± 0.0040 | -0.0022 [-0.0205, +0.0160] | -0.0044 [-0.0124, +0.0036] | 1/3 |

## Table B (held-out BPB). Splitting the 300 s gap into exposure, architecture and schedule

Negative = favours AP-Q. Components add up exactly to the observed gap.

| Component | Mean [95% t-CI] | Per seed (6, 7, 8) |
|---|---|---|
| Observed 300 s gap G | -0.0022 [-0.0205, +0.0160] | +0.0037, -0.0105, +0.0001 |
| Exposure: AR-ref moved from its 300 s tokens to AP-Q's | -0.0406 [-0.0423, -0.0388] | -0.0403, -0.0414, -0.0401 |
| Architecture: AP-Q − AR-ref at AP-Q's 300 s tokens | +0.0396 [+0.0227, +0.0565] | +0.0454, +0.0321, +0.0412 |
| Schedule residual (fixed-time vs token-cell path) | -0.0012 [-0.0019, -0.0006] | -0.0015, -0.0012, -0.0010 |
| Alt. exposure: AP-Q's own gain from its extra tokens | -0.0394 [-0.0417, -0.0372] | -0.0385, -0.0403, -0.0394 |
| Alt. architecture: at AR-ref's 300 s tokens | +0.0384 [+0.0222, +0.0547] | +0.0437, +0.0311, +0.0405 |
| Interaction (architecture gap change with exposure) | +0.0011 [-0.0002, +0.0025] | +0.0017, +0.0010, +0.0007 |

| Ratio | Mean [95% t-CI] | Per seed |
|---|---|---|
| 300 s token ratio T_A/T_R | 1.796 [1.789, 1.803] | 1.798, 1.793, 1.798 |
| Steady-state throughput ratio ρ (300 s runs) | 1.810 [1.802, 1.818] | 1.812, 1.806, 1.812 |
| Token multiple k AP-Q needs to match AR-ref's 300 s quality | 1.787 [1.291, 2.283] | 1.958, 1.568, 1.834 |

Extrapolated curve points (outside 88.6M–201.3M): {6: ['AR_at_T_R', 'AP_at_T_R'], 7: ['AR_at_T_R', 'AP_at_T_R'], 8: ['AR_at_T_R', 'AP_at_T_R']}. k inversion: {6: 'in_range', 7: 'in_range', 8: 'in_range'}.

## Table B (validation BPB). Splitting the 300 s gap into exposure, architecture and schedule

Negative = favours AP-Q. Components add up exactly to the observed gap.

| Component | Mean [95% t-CI] | Per seed (6, 7, 8) |
|---|---|---|
| Observed 300 s gap G | -0.0044 [-0.0124, +0.0036] | -0.0038, -0.0079, -0.0016 |
| Exposure: AR-ref moved from its 300 s tokens to AP-Q's | -0.0474 [-0.0494, -0.0453] | -0.0468, -0.0470, -0.0483 |
| Architecture: AP-Q − AR-ref at AP-Q's 300 s tokens | +0.0430 [+0.0341, +0.0520] | +0.0425, +0.0397, +0.0469 |
| Schedule residual (fixed-time vs token-cell path) | -0.0001 [-0.0015, +0.0013] | +0.0005, -0.0006, -0.0002 |
| Alt. exposure: AP-Q's own gain from its extra tokens | -0.0461 [-0.0464, -0.0458] | -0.0462, -0.0460, -0.0461 |
| Alt. architecture: at AR-ref's 300 s tokens | +0.0418 [+0.0343, +0.0493] | +0.0420, +0.0387, +0.0447 |
| Interaction (architecture gap change with exposure) | +0.0013 [-0.0008, +0.0033] | +0.0006, +0.0010, +0.0022 |

| Ratio | Mean [95% t-CI] | Per seed |
|---|---|---|
| 300 s token ratio T_A/T_R | 1.796 [1.789, 1.803] | 1.798, 1.793, 1.798 |
| Steady-state throughput ratio ρ (300 s runs) | 1.810 [1.802, 1.818] | 1.812, 1.806, 1.812 |
| Token multiple k AP-Q needs to match AR-ref's 300 s quality | 1.699 [1.538, 1.860] | 1.701, 1.633, 1.763 |

Extrapolated curve points (outside 88.6M–201.3M): {6: ['AR_at_T_R', 'AP_at_T_R'], 7: ['AR_at_T_R', 'AP_at_T_R'], 8: ['AR_at_T_R', 'AP_at_T_R']}. k inversion: {6: 'in_range', 7: 'in_range', 8: 'in_range'}.

## Seed consistency (held-out)

- Largest within-recipe SD in any cell: 0.0045.
- Per-seed Δ curve shape over 88.6M → 157.3M → 201.3M: {6: 'increasing', 7: 'increasing', 8: 'increasing'}.

## Replication of the Phase-1 panel at 201.3M tokens (seeds 3–5 → fresh seeds 6–8)

| Metric | Panel seeds 3–5 | E1 seeds 6–8 | Welch E1 − panel |
|---|---|---|---|
| held-out BPB | +0.0420 [+0.0297, +0.0544] | +0.0415 [+0.0287, +0.0544] | -0.0005 [-0.0120, +0.0110] |
| validation BPB | +0.0451 [+0.0415, +0.0486] | +0.0445 [+0.0364, +0.0527] | -0.0005 [-0.0075, +0.0064] |


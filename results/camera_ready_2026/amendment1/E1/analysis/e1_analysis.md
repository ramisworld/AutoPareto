# E1 analysis (pre-registered)

Delta = BPB(AP-Q) - BPB(AR-ref); mean with 95% t-interval over seeds [6, 7, 8].

## heldout_bpb

| cell | n | mean Delta | 95% CI | per seed |
|---|---|---|---|---|
| tokens_088604672 | 3 | +0.0385 | [+0.0222, +0.0547] | +0.0437, +0.0311, +0.0405 |
| tokens_157286400 | 3 | +0.0396 | [+0.0226, +0.0565] | +0.0454, +0.0321, +0.0412 |
| tokens_201326592 | 3 | +0.0415 | [+0.0287, +0.0544] | +0.0465, +0.0362, +0.0418 |
| time_300s | 3 | -0.0022 | [-0.0205, +0.0160] | +0.0037, -0.0105, +0.0001 |

- Crossover (mean curve, log-token interpolation): none in range (all_positive)
- Seed bootstrap: exact enumeration of all 3^3 = 27 equally likely seed resamples; P(crossover in range) = 0.0
- H-E1a: inconclusive; H-E1b: contradicted; outcome: exposure_explains_fixed_time_advantage

## validation_bpb

| cell | n | mean Delta | 95% CI | per seed |
|---|---|---|---|---|
| tokens_088604672 | 3 | +0.0418 | [+0.0343, +0.0493] | +0.0420, +0.0387, +0.0447 |
| tokens_157286400 | 3 | +0.0430 | [+0.0341, +0.0519] | +0.0425, +0.0397, +0.0468 |
| tokens_201326592 | 3 | +0.0445 | [+0.0364, +0.0527] | +0.0440, +0.0416, +0.0480 |
| time_300s | 3 | -0.0044 | [-0.0124, +0.0036] | -0.0038, -0.0079, -0.0016 |

- Crossover (mean curve, log-token interpolation): none in range (all_positive)
- Seed bootstrap: exact enumeration of all 3^3 = 27 equally likely seed resamples; P(crossover in range) = 0.0
- H-E1a: supported; H-E1b: contradicted; outcome: exposure_explains_fixed_time_advantage

## 300 s cell vs seeds 0-2

- E1 mean -0.0044 [-0.0124, +0.0036] vs historical -0.0067; Welch difference +0.0023 [-0.0045, +0.0090] -> consistent

## Schedule checks

- AP-Q: time_300s - tokens_157286400: -0.0001 [-0.0014, +0.0012]; effect detected: False; fixed-time tokens {'6': 158334976, '7': 157679616, '8': 158334976}
- AR-ref: time_300s - tokens_088604672: +0.0019 [+0.0005, +0.0033]; effect detected: False; fixed-time tokens {'6': 88080384, '7': 87949312, '8': 88080384}

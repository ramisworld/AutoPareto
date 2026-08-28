# Phase 1: equal-token verification results

This is the current fixed-token verification panel. Every recipe received exactly
`201,326,592` training tokens on an NVIDIA A40 at seeds 3, 4, and 5. The project
uses “200M” as the campaign label, but the recorded token count is
`201,326,592`; this document uses the exact recorded value.

The tables below were verified from:

- `results/phase1/phase1_equal_token_results.csv` and `.json`;
- the private Phase 1 SQLite database, downstream runs 14–34; and
- the private per-run raw JSON files under `results/phase1/downstream_200m_evaluations/`.

The public branch retains only the aggregate result summaries and frozen protocol
manifests. It intentionally excludes the database, raw bundles, checkpoints, logs,
and telemetry. The public protocol manifest is
`evaluation/phase1_public_manifest.json`; the checked-in aggregate downstream values
used by the figures are in `results/phase1/public_downstream_metrics.json`.

No values in this document were rerun or estimated. Means and SDs are across seeds
3–5; SD means sample standard deviation. Lower is better for validation/held-out
BPB and time; higher is better for throughput and downstream accuracy.

### Count reconciliation

`phase1_200m_downstream_summary.json` reports 32 runs, 288 task rows, and 576
generation rows because that summary includes the 11 historical downstream rows copied
into the Phase 1 database. The new campaign itself is `phase1-200m-downstream` and
contains 21 runs, 189 task rows, and 378 generation rows. This document reports only
those 21 new equal-token runs.

## Panel and architecture

| Model | Parameters | Layers | Width | Q/KV heads | Window pattern | Total/device batch | Grad. accum. | Source/role |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Baseline | 50,332,176 | 8 | 512 | 4 / 4 | SSSL | 524,288 / 64 | 4 | Untouched original `train.py` |
| AutoResearch | 50,332,176 | 8 | 512 | 4 / 4 | SSSS | 131,072 / 64 | 1 | Strongest compatible AutoResearch reference |
| TPE | 39,846,284 | 6 | 512 | 4 / 4 | SSSS | 131,072 / 64 | 1 | Frozen TPE quality representative |
| MOTPE | 26,345,772 | 6 | 384 | 3 / 3 | L | 131,072 / 64 | 1 | Frozen MOTPE quality representative |
| AutoPareto 292 | 29,360,392 | 4 | 512 | 4 / 4 | SSSL | 131,072 / 64 | 1 | Confirmed-quality AutoPareto recipe |
| AutoPareto 299 | 26,214,662 | 3 | 512 | 4 / 4 | SSSL | 131,072 / 64 | 1 | Shallow-wide fast-quality recipe |
| AutoPareto 294 | 17,891,526 | 3 | 384 | 3 / 3 | SSSL | 131,072 / 64 | 1 | Efficiency-boundary recipe |

All Phase 1 recipes used the frozen token budget, device batch 64, and sequence length
2,048. AutoResearch and AP292 therefore have identical batch settings in the central
comparison; the baseline is the explicit exception shown above. The selection
manifest is `results/phase1/phase1_200m_downstream_selection.json`; full candidate and
checkpoint hashes were stored there and in the private SQLite run rows.

## Per-seed training and inference results

`Scientific wall` includes the full measured training workload. `Steady equivalent`
is the preregistered sustained-rate estimate. `Decode` is the stabilized cached
decode median from 30 synchronized repetitions. `Allocated` and `reserved` are peak
inference CUDA memory; `KV` is incremental KV-cache memory for the measured workload.

| Model | Seed | Tokens | Scientific wall (s) | Steady equivalent (s) | Train tok/s | Val BPB | Held-out BPB | Decode tok/s | Prefill ms | Alloc MiB | Reserved MiB | KV MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3 | 201,326,592 | 824.916 | 688.627 | 292,360 | 1.089689 | 1.143110 | 199.609 | 5.191 | 250.882 | 278 | 3.984 |
| Baseline | 4 | 201,326,592 | 798.841 | 689.161 | 292,133 | 1.097355 | 1.151900 | 195.036 | 5.357 | 250.882 | 312 | 3.984 |
| Baseline | 5 | 201,326,592 | 805.307 | 687.712 | 292,748 | 1.093269 | 1.144758 | 195.817 | 5.353 | 250.882 | 312 | 3.984 |
| AutoResearch | 3 | 201,326,592 | 825.127 | 696.233 | 289,166 | 1.032588 | 1.100921 | 193.990 | 5.391 | 250.882 | 278 | 3.984 |
| AutoResearch | 4 | 201,326,592 | 818.076 | 696.753 | 288,950 | 1.033224 | 1.100929 | 193.437 | 5.475 | 250.882 | 312 | 3.984 |
| AutoResearch | 5 | 201,326,592 | 815.396 | 695.505 | 289,468 | 1.032172 | 1.097560 | 191.373 | 5.490 | 250.882 | 278 | 3.984 |
| TPE | 3 | 201,326,592 | 663.980 | 541.075 | 372,086 | 1.053741 | 1.114508 | 257.882 | 4.119 | 205.881 | 256 | 2.988 |
| TPE | 4 | 201,326,592 | 635.869 | 540.296 | 372,623 | 1.058750 | 1.122379 | 256.091 | 4.176 | 205.881 | 242 | 2.988 |
| TPE | 5 | 201,326,592 | 695.907 | 540.208 | 372,684 | 1.059105 | 1.121317 | 262.710 | 4.071 | 205.881 | 256 | 2.988 |
| MOTPE | 3 | 201,326,592 | 537.231 | 398.235 | 505,547 | 1.079109 | 1.150809 | 276.141 | 3.857 | 145.568 | 216 | 2.241 |
| MOTPE | 4 | 201,326,592 | 535.287 | 398.205 | 505,585 | 1.076726 | 1.149146 | 265.964 | 4.053 | 145.568 | 170 | 2.241 |
| MOTPE | 5 | 201,326,592 | 544.055 | 398.257 | 505,519 | 1.081962 | 1.144802 | 267.915 | 3.996 | 145.568 | 216 | 2.241 |
| AutoPareto 292 | 3 | 201,326,592 | 546.657 | 385.161 | 522,708 | 1.076589 | 1.139616 | 382.024 | 2.879 | 162.880 | 222 | 1.992 |
| AutoPareto 292 | 4 | 201,326,592 | 482.293 | 385.135 | 522,742 | 1.079909 | 1.148668 | 378.697 | 2.877 | 162.880 | 222 | 1.992 |
| AutoPareto 292 | 5 | 201,326,592 | 508.880 | 384.761 | 523,251 | 1.076705 | 1.137228 | 388.344 | 2.869 | 162.880 | 222 | 1.992 |
| AutoPareto 299 | 3 | 201,326,592 | 455.483 | 313.176 | 642,855 | 1.097279 | 1.159419 | 482.644 | 2.324 | 142.380 | 198 | 1.494 |
| AutoPareto 299 | 4 | 201,326,592 | 422.371 | 312.792 | 643,644 | 1.096373 | 1.158897 | 484.342 | 2.339 | 142.380 | 198 | 1.494 |
| AutoPareto 299 | 5 | 201,326,592 | 461.745 | 312.388 | 644,477 | 1.097788 | 1.159611 | 497.789 | 2.278 | 142.380 | 198 | 1.494 |
| AutoPareto 294 | 3 | 201,326,592 | 443.221 | 303.109 | 664,204 | 1.127616 | 1.190911 | 500.056 | 2.248 | 106.942 | 168 | 1.121 |
| AutoPareto 294 | 4 | 201,326,592 | 392.546 | 291.439 | 690,801 | 1.128388 | 1.191678 | 493.223 | 2.294 | 106.942 | 168 | 1.121 |
| AutoPareto 294 | 5 | 201,326,592 | 405.734 | 292.499 | 688,299 | 1.129135 | 1.187580 | 493.195 | 2.304 | 106.942 | 168 | 1.121 |

## Mean ± SD across seeds

| Model | Val BPB | Held-out BPB | Scientific wall (s) | Steady equivalent (s) | Train tok/s | Decode tok/s | Prefill ms | Alloc MiB | Reserved MiB | KV MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1.093438 ± 0.003836 | 1.146590 ± 0.004673 | 809.688 ± 13.578 | 688.500 ± 0.733 | 292,413.534 ± 311.346 | 196.821 ± 2.446 | 5.300 ± 0.095 | 250.882 ± 0.000 | 300.667 ± 19.630 | 3.984 ± 0.000 |
| AutoResearch | 1.032661 ± 0.000530 | 1.099803 ± 0.001943 | 819.533 ± 5.027 | 696.163 ± 0.627 | 289,194.669 ± 260.449 | 192.933 ± 1.380 | 5.452 ± 0.053 | 250.882 ± 0.000 | 289.333 ± 19.630 | 3.984 ± 0.000 |
| TPE | 1.057199 ± 0.003000 | 1.119402 ± 0.004271 | 665.252 ± 30.039 | 540.526 ± 0.477 | 372,464.266 ± 328.634 | 258.894 ± 3.423 | 4.122 ± 0.052 | 205.881 ± 0.000 | 251.333 ± 8.083 | 2.988 ± 0.000 |
| MOTPE | 1.079266 ± 0.002622 | 1.148253 ± 0.003102 | 538.858 ± 4.605 | 398.233 ± 0.026 | 505,550.269 ± 32.894 | 270.006 ± 5.401 | 3.969 ± 0.101 | 145.568 ± 0.000 | 200.667 ± 26.558 | 2.241 ± 0.000 |
| AutoPareto 292 | 1.077734 ± 0.001884 | 1.141837 ± 0.006035 | 512.610 ± 32.344 | 385.019 ± 0.224 | 522,900.350 ± 303.969 | 383.022 ± 4.900 | 2.875 ± 0.005 | 162.880 ± 0.000 | 222.000 ± 0.000 | 1.992 ± 0.000 |
| AutoPareto 299 | 1.097147 ± 0.000717 | 1.159309 ± 0.000370 | 446.533 ± 21.158 | 312.785 ± 0.394 | 643,658.811 ± 810.950 | 488.258 ± 8.298 | 2.313 ± 0.032 | 142.380 ± 0.000 | 198.000 ± 0.000 | 1.494 ± 0.000 |
| AutoPareto 294 | 1.128379 ± 0.000760 | 1.190057 ± 0.002179 | 413.834 ± 26.290 | 295.683 ± 6.454 | 681,101.340 ± 14,686.517 | 495.491 ± 3.953 | 2.282 ± 0.030 | 106.942 ± 0.000 | 168.000 ± 0.000 | 1.121 ± 0.000 |

## Downstream evaluation: per seed

Every 200M checkpoint received the same frozen nine-metric suite. The values below
come from `downstream_task_results` for experiment IDs 301–321. `L-EM` is LAMBADA
exact match; `L-PPL` is LAMBADA perplexity; `ARC-N` and `PIQA-N` are the
length-normalized variants; BLiMP macro and micro are numerically equal here because
all 67 categories contain 1,000 pairs.

| Model | Seed | Held-out BPB | L-PPL | L-EM | ARC | ARC-N | PIQA | PIQA-N | BLiMP macro | BLiMP micro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3 | 1.143110 | 14.538743 | 0.201436 | 0.359848 | 0.336279 | 0.569097 | 0.579434 | 0.722597 | 0.722597 |
| Baseline | 4 | 1.151900 | 14.798131 | 0.201048 | 0.335859 | 0.331229 | 0.577258 | 0.577258 | 0.712254 | 0.712254 |
| Baseline | 5 | 1.144758 | 13.705270 | 0.211721 | 0.362795 | 0.332492 | 0.574538 | 0.573993 | 0.706313 | 0.706313 |
| AutoResearch | 3 | 1.100921 | 15.238148 | 0.194256 | 0.357744 | 0.316498 | 0.575082 | 0.556039 | 0.746209 | 0.746209 |
| AutoResearch | 4 | 1.100929 | 13.941088 | 0.212498 | 0.380892 | 0.334596 | 0.601741 | 0.581066 | 0.739388 | 0.739388 |
| AutoResearch | 5 | 1.097560 | 13.276109 | 0.206482 | 0.356481 | 0.321970 | 0.583243 | 0.580522 | 0.751179 | 0.751179 |
| TPE | 3 | 1.114508 | 12.372591 | 0.228411 | 0.367424 | 0.338384 | 0.583787 | 0.569641 | 0.741284 | 0.741284 |
| TPE | 4 | 1.122379 | 14.126521 | 0.209975 | 0.347222 | 0.320286 | 0.575626 | 0.551687 | 0.725761 | 0.725761 |
| TPE | 5 | 1.121317 | 15.246905 | 0.204347 | 0.356481 | 0.315236 | 0.569641 | 0.560936 | 0.731866 | 0.731866 |
| MOTPE | 3 | 1.150809 | 14.101344 | 0.230157 | 0.350589 | 0.342172 | 0.570729 | 0.569097 | 0.718672 | 0.718672 |
| MOTPE | 4 | 1.149146 | 14.027937 | 0.226470 | 0.373737 | 0.353956 | 0.579434 | 0.565288 | 0.726134 | 0.726134 |
| MOTPE | 5 | 1.144802 | 17.055443 | 0.189210 | 0.337963 | 0.322811 | 0.573449 | 0.562568 | 0.729373 | 0.729373 |
| AutoPareto 292 | 3 | 1.139616 | 12.333667 | 0.232486 | 0.371212 | 0.355219 | 0.579978 | 0.579978 | 0.716493 | 0.716493 |
| AutoPareto 292 | 4 | 1.148668 | 15.311066 | 0.216767 | 0.364899 | 0.340067 | 0.584875 | 0.574538 | 0.720403 | 0.720403 |
| AutoPareto 292 | 5 | 1.137228 | 14.980755 | 0.195614 | 0.371212 | 0.329545 | 0.583787 | 0.582699 | 0.723627 | 0.723627 |
| AutoPareto 299 | 3 | 1.159419 | 15.786818 | 0.207064 | 0.338805 | 0.306818 | 0.571817 | 0.562568 | 0.700746 | 0.700746 |
| AutoPareto 299 | 4 | 1.158897 | 14.804391 | 0.209975 | 0.365741 | 0.345539 | 0.565832 | 0.550054 | 0.712925 | 0.712925 |
| AutoPareto 299 | 5 | 1.159611 | 16.221183 | 0.205899 | 0.365741 | 0.324495 | 0.580522 | 0.577258 | 0.700537 | 0.700537 |
| AutoPareto 294 | 3 | 1.190911 | 17.949961 | 0.196779 | 0.351431 | 0.339646 | 0.568009 | 0.565832 | 0.704716 | 0.704716 |
| AutoPareto 294 | 4 | 1.191678 | 18.013210 | 0.206094 | 0.338384 | 0.321549 | 0.567465 | 0.562568 | 0.691418 | 0.691418 |
| AutoPareto 294 | 5 | 1.187580 | 17.157338 | 0.194644 | 0.355219 | 0.330387 | 0.571817 | 0.559848 | 0.683896 | 0.683896 |

## Downstream evaluation: mean ± SD

| Model | Held-out BPB | L-PPL | L-EM | ARC | ARC-N | PIQA | PIQA-N | BLiMP macro | BLiMP micro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1.146590 ± 0.004673 | 14.347381 ± 0.571008 | 0.204735 ± 0.006053 | 0.352834 ± 0.014775 | 0.333333 ± 0.002628 | 0.573631 ± 0.004155 | 0.576895 ± 0.002738 | 0.713721 ± 0.008240 | 0.713721 ± 0.008240 |
| AutoResearch | 1.099803 ± 0.001943 | 14.151782 ± 0.997844 | 0.204412 ± 0.009295 | 0.365039 ± 0.013744 | 0.324355 ± 0.009282 | 0.586688 ± 0.013660 | 0.572543 ± 0.014295 | 0.745592 ± 0.005920 | 0.745592 ± 0.005920 |
| TPE | 1.119402 ± 0.004271 | 13.915339 ± 1.448747 | 0.214244 ± 0.012587 | 0.357043 ± 0.010113 | 0.324635 ± 0.012171 | 0.576351 ± 0.007101 | 0.560754 ± 0.008979 | 0.732970 ± 0.007820 | 0.732970 ± 0.007820 |
| MOTPE | 1.148253 ± 0.003102 | 15.061575 ± 1.727131 | 0.215279 ± 0.022652 | 0.354097 ± 0.018143 | 0.339646 ± 0.015725 | 0.574538 ± 0.004453 | 0.565651 ± 0.003279 | 0.724726 ± 0.005488 | 0.724726 ± 0.005488 |
| AutoPareto 292 | 1.141837 ± 0.006035 | 14.208496 ± 1.632028 | 0.214956 ± 0.018502 | 0.369108 ± 0.003645 | 0.341611 ± 0.012906 | 0.582880 ± 0.002571 | 0.579071 ± 0.004155 | 0.720174 ± 0.003573 | 0.720174 ± 0.003573 |
| AutoPareto 299 | 1.159309 ± 0.000370 | 15.604131 ± 0.725848 | 0.207646 ± 0.002099 | 0.356762 ± 0.015552 | 0.325617 ± 0.019385 | 0.572724 ± 0.007387 | 0.563293 ± 0.013616 | 0.704736 ± 0.007093 | 0.704736 ± 0.007093 |
| AutoPareto 294 | 1.190057 ± 0.002179 | 17.706836 ± 0.476929 | 0.199172 ± 0.006089 | 0.348345 ± 0.008832 | 0.330527 ± 0.009050 | 0.569097 ± 0.002372 | 0.562749 ± 0.002997 | 0.693343 ± 0.010543 | 0.693343 ± 0.010543 |

## Interpretation supported by this panel

AutoPareto 292 is the safest current balanced candidate: compared with the equal-size
AutoResearch reference, it has 41.7% fewer parameters, about 1.81× higher stabilized
cached decode throughput, and about 45% lower steady-state-equivalent training time,
but held-out BPB is worse by 0.0420. Its measured downstream results remain close on
ARC-Easy, PIQA, LAMBADA, and BLiMP, although BLiMP is lower than AutoResearch
(0.7202 versus 0.7456).

AutoPareto 299 and 294 are more aggressive efficiency points. They are faster and
smaller than 292 but have progressively worse held-out BPB and downstream quality.
The panel therefore supports an efficiency–quality frontier, not a claim that
AutoPareto universally beats AutoResearch.

## Provenance and scope warnings

- The 200M downstream database also contains the 11 historical downstream rows copied
  from the original panel. The 21 new 200M rows are the campaign
  `phase1-200m-downstream`, experiments 301–321, downstream runs 14–34.
- This panel does not include 200M reruns for CMA-ES, Random Search, NSGA-II, or
  AutoPareto 289.
- The training and downstream evaluator hashes, checkpoint hashes, and candidate
  hashes are recorded per run. The downstream evaluator SHA differs from the original
  five-minute panel because the tensor-layout fix changed the evaluator artifact.
- These are fixed-token A40 results. They are not claims about larger models, other
  GPUs, or general-purpose language quality.

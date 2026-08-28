# Phase 1 stabilized inference rebenchmark

Pre-Stage-B measurement-protocol correction. Historical values are preserved; the standardized headline is the median of 30 stabilized synchronized-wall-clock decode repetitions. Maximum throughput is diagnostic only.

| Model | Source | Checkpoint | Provenance | Historical legacy tok/s | Stabilized median tok/s | Status | Correctness |
|---|---:|---:|---|---:|---:|---|---|
| baseline | 4 | 4 | historical selected checkpoint for experiment 4 | 158.3154 | UNAVAILABLE | unstable_environment | True |
| autoresearch_compatible | 68 | 77 | exact-recipe recovery/confirmation checkpoint for source experiment 68; original checkpoint unavailable | 161.4235 | UNAVAILABLE | unstable_environment | True |
| tpe_quality | 183 | 183 | historical selected checkpoint for source experiment 183 | 212.2945 | UNAVAILABLE | unstable_environment | True |
| motpe_quality | 234 | 300 | exact-recipe recovery checkpoint for source experiment 234; original checkpoint unavailable | 226.0848 | 268.3438 | completed | True |
| autopareto_292 | 292 | 292 | confirmed selected checkpoint for experiment 292, exact recipe sourced from discovery experiment 134 | 318.2745 | UNAVAILABLE | unstable_environment | True |
| autopareto_299 | 299 | 299 | confirmed selected checkpoint for experiment 299, exact recipe sourced from discovery experiment 137 | 361.1765 | UNAVAILABLE | unstable_environment | True |
| autopareto_294 | 294 | 294 | confirmed selected checkpoint for experiment 294, exact recipe sourced from discovery experiment 136 | 404.9996 | UNAVAILABLE | unstable_environment | True |

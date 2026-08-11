# Research protocol

## Question

Can an autonomous coding researcher discover Transformer configurations with useful trade-offs among validation BPB, cached decode throughput, GPU memory, and parameter count when it operates under a fixed training budget?

## Discovery loop

For each attempt, the agent:

1. Reads only the current campaign history and Pareto frontier.
2. Selects a valid non-dominated parent.
3. States one falsifiable hypothesis.
4. Makes one candidate source change to the Transformer architecture or training configuration.
5. Retrains from scratch for the fixed budget.
6. Runs protected cached-inference evaluation.
7. Records metrics, parentage, source/evaluator hashes, failures, and its decision.
8. Updates the campaign Pareto frontier before proposing another attempt.

## Fixed conditions

- Hardware: one NVIDIA A40.
- Numerical format: bfloat16 for training and headline inference measurements.
- Training: 300 measured seconds from scratch per attempt.
- Discovery budget: 25 attempts per AutoPareto campaign; three completed campaigns, with training seeds 42, 0, and 1.
- Objectives: minimize validation BPB, maximize cached decode throughput, minimize peak allocated inference CUDA memory; parameter count is recorded for interpretation.
- Headline inference workload: 256 input tokens, 256 generated tokens, batch size 1, 10 measured cached-decode repetitions.
- Baselines: a same-seed baseline is recorded alongside each AutoPareto campaign.
- Failure accounting: crashes, OOMs, timeouts, cache incompatibilities, and protection failures consume an attempt.
- Isolation: campaigns cannot inspect one another's histories.

## Validity gate

Cached generation must pass correctness validation. A candidate that fails cached inference cannot contribute a throughput or memory result. A stricter fp32 math-SDPA path is used only to adjudicate ambiguous bfloat16 cache-correctness outcomes; it is not a performance measurement mode.

## Interpretation rules

Discovery campaigns map candidate trade-offs. They do not establish a confirmed frontier, cross-seed generalization of individual recipes, downstream task performance, quantized performance, or scale transfer. Those distinctions are stated beside the results in the README.

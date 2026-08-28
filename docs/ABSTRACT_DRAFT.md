# Abstract draft — AutoPareto

Autonomous machine-learning systems are usually optimized for a single validation
loss, even when the deployed model must also satisfy latency and memory constraints.
AutoPareto is an auditable autonomous research loop that edits a protected language
model training program, trains candidate recipes under a fixed compute budget, and
retains candidates using a quality–inference-speed–inference-memory Pareto
criterion. We study whether this loop can discover useful deployment trade-offs on a
single GPU without selecting models using downstream results.

In the original five-minute discovery campaign, the agent produced multiple
architecturally distinct quality–speed–memory trade-offs alongside AutoResearch and
classical-search controls. We then evaluated a frozen seven-recipe panel at exactly
201,326,592 training tokens using three seeds per recipe. The strongest current
balanced AutoPareto candidate, AP292, has 29.36M parameters, a mean stabilized cached
decode throughput of 383.02 tokens/s, and a mean steady-state-equivalent training time
of 385.02 s. The equal-size AutoResearch reference has 50.33M parameters, 192.93
tokens/s, and 696.16 s respectively. AP292 therefore demonstrates a substantial
efficiency point, but its held-out BPB is worse (1.14184 versus 1.09980) and its BLiMP
accuracy is lower (0.72017 versus 0.74559). ARC-Easy, PIQA, and LAMBADA exact match
are comparatively close on the measured panel. AP299 and AP294 extend the frontier
toward higher throughput and lower memory with additional quality degradation.

The qualifying autonomous-agent result is therefore a reproducible, auditable
quality–efficiency frontier discovery—not a claim that AutoPareto universally beats
AutoResearch or improves general language quality. The artifact preserves candidate
lineage, training and inference measurements, checkpoint hashes, failures, frozen
benchmark definitions, raw aggregate results, and seed-level verification data. The
results support the hypothesis that multi-objective autonomous search can discover
smaller and materially faster model recipes with near-reference performance on several
downstream tasks, while also showing that the quality cost is real and task-dependent.

## Open questions before submission

- Which individual architectural or training changes cause AP292’s efficiency, after
  controlling for parameter count, batch size, and training schedule?
- Does the AP292 frontier transfer from the A40 to an RTX 3090 and H100?
- Does the result survive a final untouched shard and a larger 100–300M-scale pilot?
- Are classical-search comparisons fair under matched seeds, budgets, and checkpoint
  availability?
- Can we add a reproducible per-example benchmark trace without changing the frozen
  headline results?
- Which workshop track and submission rules apply, and what claims remain supportable
  after the final transfer and ablation evidence?

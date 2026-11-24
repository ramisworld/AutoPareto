# Next experiment plan: a replicated resource-contract study

## Why the current design cannot answer the question

The completed paired study contains four attempts under contract T and four under contract N, on a single training seed, with each arm forming one adaptive trajectory. Attempt k depends on attempts 1..k-1, so the eight records are one observation per arm, not eight. Extending the same two trajectories to 25 attempts each would produce a longer anecdote: it increases within-trajectory resolution while leaving the between-trajectory variance, which is the quantity a contract effect must be measured against, entirely unestimated.

## Proposed allocation of roughly 50 attempt slots

Run **five independent campaigns per contract, ten attempts each** (2 x 5 x 10 = 100 attempts at 300 s or 88.6M tokens). If only 50 slots are available, run **five campaigns per contract of five attempts each**, accepting shallower trajectories in exchange for replication. Replication is the binding constraint, not depth.

Independence across campaigns requires, at minimum, a distinct training seed per campaign and a fresh campaign history; the same five seeds should be used in both arms so the comparison is paired at the campaign level.

| Factor | Setting |
| --- | --- |
| Arms | T (300.000 protected seconds) and N (88,604,672 counted tokens) |
| Campaigns per arm | 5, paired by training seed across arms |
| Attempts per campaign | 10 (or 5 under a 50-slot budget) |
| Seeds | 5 fresh seeds, none used in discovery, verification, or the current prefix |
| Held fixed | Staged source bytes, agent model and reasoning setting, timeout, editable surface, evaluator protocol, hardware image, parent-eligibility rules |
| Varied | Stopping rule and schedule-progress variable only |

## Pre-declared outcomes

Declare these before launch, with the campaign as the unit of analysis:

1. **Capacity trajectory.** Parameter count of each attempt, summarized as the campaign mean and the sign of the first two edits.
2. **Exposure.** Counted tokens per attempt. Under T this is an outcome; under N it is fixed by construction, which is itself the contrast.
3. **Edit composition.** Fraction of attempts in each source-diff category (architecture, optimizer, batching, schedule, optimization logic), classified by the existing audit rules with the classifier frozen before launch.
4. **Achieved frontier.** Hypervolume of the campaign's Pareto set over held-out quality, throughput, and memory, computed against a fixed reference point.
5. **Final recipe quality under matched exposure.** Each campaign's best recipe frozen and retrained at 201,326,592 tokens on three fresh seeds, which is the only comparison that is contract-neutral.
6. **Hypothesis content.** Rate at which the agent's recorded hypothesis explicitly cites throughput, exposure, or update count as its rationale.

## Analysis

Compare arms with a paired test across the five seed-matched campaign pairs, reporting the effect size and its interval rather than a significance verdict at n = 5. Report between-campaign variance within each arm explicitly; if it is comparable to the arm difference, the honest conclusion is that the study is underpowered, and that is a publishable result about the difficulty of measuring contract effects.

## Cross-contract generalization (secondary, if slots remain)

Take each arm's best recipe and evaluate it under the other arm's contract. If fixed-time campaigns systematically produce recipes that lose their advantage under fixed tokens, while fixed-token campaigns produce recipes that hold up under both, that is a directional result about what each contract selects for, and it is measurable with roughly ten additional runs.

## Explicit non-goals

Do not extend the existing two trajectories and report them as a larger sample. Do not pool the current prefix with new campaigns; protocols and warm-up handling differ. Do not claim a causal contract effect from any design that lacks campaign-level replication.

# Next experiments and evidence plan

This is a prioritised plan. No experiment in this document has been run as part of the
current documentation update.

## Before the abstract is finalized

### 1. Complete the benchmark and result audit

**Question:** Are every headline number, seed, checkpoint hash, evaluator version,
split, and failure trace internally consistent?

**Strengthens the paper:** A clean audit trail with no unexplained metric or scope
disagreement. The current documentation separates the original 5-minute panel from the
201,326,592-token panel and flags that per-example benchmark traces were not persisted.

**Weakens the paper:** Any unresolved database/result mismatch, missing checkpoint
provenance, or discovery result presented as if it were equal-token evidence.

### 2. Extract seed-level uncertainty and paired contrasts

**Question:** How large are recipe differences relative to seed variation, especially
for held-out BPB and downstream accuracy?

**Strengthens the paper:** Predeclared paired contrasts, confidence intervals, and a
clear statement of which differences are practically large and which overlap seed
variation.

**Weakens the paper:** AP292’s apparent gains disappear under seed-level uncertainty,
or the current three-seed means are dominated by one favorable seed.

### 3. Freeze the qualifying result and Pareto figure

**Question:** What single claim can be defended without overselling the evidence?

**Strengthens the paper:** The current safest claim: AP292 is a substantially smaller,
faster, lower-memory recipe with downstream results close to AutoResearch on several
tasks, but worse held-out BPB and BLiMP.

**Weakens the paper:** Choosing a different “winner” after viewing downstream results,
or collapsing heterogeneous metrics into an undocumented quality score.

## Before the final paper

### 4. Controlled architecture and training-recipe ablation

**Question:** Is AP292’s frontier position caused by its architecture/depth, its
parameter count, or its optimizer/schedule and other training-recipe changes?

AP292 and AutoResearch already use the same total batch size (`131,072`), device batch
size (`64`), and gradient accumulation (`1`) in the 201,326,592-token verification.
Batch size is therefore not a confound for that central comparison, although it is not
matched across every model in the seven-recipe panel (the baseline uses a different
accumulation regime).

**Strengthens the paper:** Matched-parameter and matched-training-budget ablations
hold batch size fixed while isolating architecture/depth from optimizer, learning-rate,
attention-layout, and other recipe changes, and reproduce the efficiency gain.

**Weakens the paper:** The gain disappears when parameter count or training recipe is
controlled, leaving only an uninformative smaller-model effect; or it does not survive
an explicit architecture/depth ablation.

### 5. A40 → RTX 3090/H100 transfer

**Question:** Is the measured speed/memory frontier a recipe property or an A40/kernel
artifact?

**Strengthens the paper:** AP292/AP299/AP294 preserve their relative ordering and
quality trade-offs across at least one additional GPU, with synchronized protocol and
cache correctness.

**Weakens the paper:** The ranking reverses across hardware or the speed advantage is
specific to one kernel/runtime path.

### 6. Fair classical-search Pareto analysis

**Question:** Does AutoPareto discover a better frontier than Random Search, TPE,
CMA-ES, MOTPE, and NSGA-II under genuinely matched budgets and seeds?

**Strengthens the paper:** Matched repeated classical controls produce a statistically
clearer frontier advantage or show a reproducible complementarity between search
strategy and deployment objectives.

**Weakens the paper:** A classical method reaches the same frontier with the same
compute, or the comparison is confounded by missing checkpoints, different seeds, or
unequal attempt budgets.

### 7. Final untouched-shard evaluation

**Question:** Do the recipe rankings survive an independently protected dataset slice?

**Strengthens the paper:** AP292’s quality–efficiency position persists on a new shard,
with the same frozen evaluator and no selection feedback.

**Weakens the paper:** Held-out BPB or downstream rankings change materially, indicating
validation/shard overfitting or an unstable quality advantage.

### 8. Optional 100–300M scale-transfer pilot

**Question:** Does the discovered recipe remain useful as training and model scale
increase?

**Strengthens the paper:** AP292-like changes retain a measurable efficiency advantage
without a growing quality penalty at larger scale.

**Weakens the paper:** The advantage vanishes, quality degradation grows, or the recipe
does not train/infer robustly beyond the current small model.

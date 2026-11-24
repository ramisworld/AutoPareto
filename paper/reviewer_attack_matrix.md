# Internal audit: not an anonymized submission asset

# Reviewer attack matrix

| Attack | Assessment | Evidence / repair |
| --- | --- | --- |
| Fixed time favoring small models is obvious | Valid against a generic novelty claim | No such novelty claim; T did not shrink in first four attempts. |
| This is not novel | Partly valid | Discuss Primer, MONAQ, Ferreira and resource allocation; propose contract-level question without a first claim. |
| Sample too small | Valid | One incomplete paired trajectory; descriptive early contrast only. |
| Adaptive dependent trials | Valid | No candidate-level significance tests; seed SD only for frozen-recipe retraining. |
| Budgets computationally unfair | Valid if called compute matched | Report elapsed time and tokens separately; no matched-FLOP claim. |
| Quality is just more tokens | Plausible in historical discovery | Equal-token verification exposes a quality cost; milestones are diagnostic; missing crossover admitted. |
| Cherry-picked candidates | Selection limits remain | All historical trials retained; all seven frozen representatives and all tasks reported; no universal winner. |
| Not autonomous | Autonomy bounded | Show source-linked hypotheses and edits; humans define objectives/harness and request confirmation. |
| Humans introduced key ideas | Valid for budget question | Attribute question to human design; qualifying agent result is recipe lineage. |
| Classical search spaces differ | Valid | No perfectly fair optimizer comparison; no aggregate superiority claim. |
| Multi-objective undefined | Addressed | Define directions and dominance; distinguish historical search from descriptive mean frontier; no hypervolume. |
| Downstream explanations post-hoc | Valid | Report complete task pattern; no causal cognitive explanation; no per-example logits. |
| Throughput noisy | Valid and observed | Historical stabilized protocol separate from current bracketed path; all eight early current speeds uncertain. |
| No hardware/scale generalization | Valid | Scope to A40 workloads and small models. |
| Proprietary agent irreproducible | Valid limitation | Preserve model/config/staged prompts; hidden provider state unavailable. |
| Prompt tells agent answer | Checkable only for visible text | Exact executed prompt diff saved; shared text inspected; no visible directional capacity claim; hidden context unavailable. |
| Token budget opportunistic | Not fully resolved by labels | N_STAR=88604672 recorded before latest launch; multiple calibration revisions and reused baseline disclosed; no claim of pristine preregistration. |
| Benchmark artifact not ML result | Important alternative | Center verified recipe-level trade-off; new study execution mismatches and speed uncertainty explicitly limit causal interpretation. |
| Abstract hides the quality loss | Addressed | State that discovery quality does not survive equal-token held-out evaluation. |
| Abstract is a result dump | Addressed | Keep one central result and one durable deployment result. |
| Research question appears too late | Addressed | Open the abstract and introduction with the contract question. |
| Why should a reviewer care | Addressed | The contract changes exposure and can change the apparent discovery. |
| Title promises optimizer superiority | Addressed | Center the title on separating architecture from resource allocation. |
| AutoPareto merely makes models smaller | Partly valid | Lineage shows failed compression, width recovery, attention, batching, and optimizer edits. |
| Depth reduction is obvious | Rejected as the full account | The first four-layer branch loses quality; later width and optimizer changes recover it. |
| Attempt 25 is cherry-picked | Addressed | Show every seed-0 proposal, parent, failure, BPB, and speed. |
| Attempt 25 is AP292 | Potential confusion | Separate A25 historical throughput from AP292 equal-token confirmation. |
| Validation BPB is optimistic | Valid | Use validation only for discovery; headline quality uses held-out BPB. |
| Held-out data leaked | No artifact evidence | Protected evaluation is outside the editable surface and hashed. |
| Evaluation code was editable | Rejected for current controller | Only train.py is writable; historical boundary is described from recovered code. |
| The agent saw the other arm | Visible inputs say no | Separate histories and exact staged input diffs are retained. |
| Provider sampling differs | Unresolved | Record model and reasoning without claiming bitwise agent reproducibility. |
| Matched seed creates dependence | Design choice | It controls initialization; trajectories remain adaptive units. |
| One seed overstates quality | Addressed | Seven frozen recipes are retrained at three new seeds. |
| Three seeds are underpowered | Valid | Report point estimates and SD without significance claims. |
| No confidence intervals | Reasonable | Retain seed values; avoid asymptotic inference at n=3. |
| Pareto dominance uses means | Addressed | Call it a descriptive mean frontier and publish seed values. |
| Metric directions are wrong | Tested | Automated tests cover lower and higher directions and dominance. |
| Memory is vague | Addressed | Define peak allocated CUDA memory and separate reserved and KV fields. |
| KV memory is inferred | Rejected | Use measured per-repetition KV fields. |
| Warm-up creates speed result | Addressed | Confirmation uses stabilized repetitions; current brackets remain separate. |
| Context length creates speed result | Valid scope | State the exact 256 input and 256 generation headline workload. |
| Flash Attention comparison is unfair | Addressed | Frozen recipes share one inference protocol; repaired current timing is separate. |
| A40 evidence is narrow | Valid | Scope claims to one A40 and this workload. |
| A40 is cheaper without evidence | Avoided | Describe the hardware, not a market price. |
| Training and decode speed are confused | Addressed | Label training ktok/s and cached decode tok/s separately. |
| CMA-ES and TPE are unfairly compared | Valid | Treat fixed-vector results as diagnostics under unequal spaces. |
| AutoResearch differs beyond objective | Valid | Disclose prompt, commit, wrapper, and feedback differences. |
| AutoResearch source was modified | Checkable | Freeze and hash the recovered compatible source. |
| Downstream suite is post-selected | Timing unresolved | Report the complete recovered suite without claiming preregistration. |
| Only favorable downstream tasks appear | Rejected | Show every predefined displayed metric, including losses. |
| Length normalization manufactures wins | Addressed | Show raw and normalized metrics together. |
| BLiMP explanation is storytelling | Constrained | Use configuration clusters and recipe-level correlations, with no causal claim. |
| Depth alone explains BLiMP | Not supported | BLiMP tracks held-out BPB more strongly than depth in the seven-recipe panel. |
| Correlation proves mechanism | Rejected | Label the analysis descriptive and request controlled ablations. |
| There are no ablations | Partly valid | Name the budget-contract ablation and historical component ablations accurately. |
| Fixed token is compute matched | False | Call it exposure matched; compute, updates, and time vary. |
| Batch size changes updates | Addressed | Report tokens, updates, batch, and schedule progress. |
| Warm-up trains the final model | Rejected | Discard it and verify reconstructed initial-state hashes. |
| Warm-up used a separate process | False for execution | State same-worker reconstruction and disclose the prompt mismatch. |
| Failures disappeared | Rejected | Keep crashes, corrupt rows, uncertain speeds, and poor outcomes. |
| Duplicate IDs inflate counts | Tested | Namespace identities by run root. |
| Late evidence crosses cutoff | Tested | Use the explicit cutoff and persistent first-observation bounds. |
| Hypothesis was preregistered late | Valid concern | Distinguish retrospective audit freeze from registered manifests. |
| Controller description is aspirational | Addressed | Describe executed gates and disclose deviations. |
| Codex cannot be reproduced | Valid limit | Release harness inputs and outputs while disclosing provider state. |
| Workshop page limits are violated | Tested | Check Part 1, Part 2, and Part 3 independently. |
| Part 2 is not human-written | Submission blocker | Keep it provisional until the human author rewrites and verifies it. |
| Author identity breaks blinding | Tested | Keep main.pdf anonymous and identifiers in private dossiers. |
| References are decorative | Addressed | Use primary work to delimit search space and resource questions. |
| Novelty depends on a first claim | Avoided | Claim an audited separation, not first-of-kind status. |
| Conclusion exceeds evidence | Guarded | Scope to the measured hardware, scale, artifacts, and workload. |

Evidence for all local assessments: normalized records and exact sources in evidence_inventory.md; external claims: audit/reference_metadata.json. Unresolved criticisms remain limitations, not hidden exclusions.

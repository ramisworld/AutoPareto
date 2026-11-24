# Confirmation decision 1: include the Codex G4 comparator (frozen before the Opus campaign)

Decided 2026-11-04, before any attempt of `pair-a-claude-opus55-t-seed3` existed. At the commit
that adds this file, neither the campaign state root nor its derived manifest existed, so no Opus
result could have influenced this decision.

The confirmation pre-registration `evaluation/CLAUDE_CAMPAIGN_CONFIRMATION_PREREG.md`
(sha256 `adf8ce713c57e55a9250bcfac3ed2f8a17703b462c416489a0c9a6041fc80968`) is unchanged.
This file exercises its §3 option: "Optional, approval required: the Codex G4 T-arm best-quality
candidate on the same 12-run design, for a same-harness agent comparison."

**Approval.** The PI, in chat on 2026-11-04: "I want to proceed with the 25-attempt Opus campaign
and include the Codex G4 comparator in the independent confirmation … Freeze the G4 comparator
decision before observing Opus results."

## 1. The comparator: identity and selection
The comparator was selected by applying the pre-registration's role-1 rule (§2) to the Codex G4
T arm. The rule takes the lowest protected validation BPB among `pareto_eligible` completed attempts,
collapses duplicate sources to the earliest attempt, and breaks ties toward the earlier attempt.

| Field | Value |
|---|---|
| Campaign | `runs/pair-a-official-20260914-new-a40-continuation-20260916-r2`, arm `pair-a-fixed-time-seed3` (history.jsonl sha256 `aa400ef2…0afa`) |
| Selected | Attempt 14 (global position 27), `pair-a-fixed-time-seed3-attempt-14` |
| Source | sha256 `b612b6e824569be6f95ef2684071c64b6e78de6c8bce2422deaff2b3e2cae2d8` (26,194 B). Byte copy frozen at `evaluation/confirmation_sources/codex_g4_t_attempt14.train.py`. |
| Discovery result (seed 3, `GPU-35264a0f…`) | Validation BPB 1.1016129764; 126,615,552 tokens in 966 updates; `quality_valid_speed_valid`; inference `stable`, correctness passed |
| Runner-up (not used) | Attempt 15, 1.1037991 |

Of the 22 T-arm attempts, 20 were `pareto_eligible`; attempts 13 and 20 were not. The Codex campaign
halted after 22 attempts per arm, with an N-arm position in `reconciliation_required`, so the T arm
has 22 attempts, not 25.

## 2. Design (the pre-registered 12 runs plus 6)
| Factor | Levels |
|---|---|
| Recipe | Opus winner (pre-registration §2); Pair-A seed-3 baseline (`b2fb8788…`); **Codex G4 attempt 14 (`b612b6e8…`)** |
| Contract | T: 300 s; N: N\* = 85,458,944 tokens |
| Seeds | 6, 7, 8 |
| Harness | The same protected Pair-A worker, evaluator and bracketed inference benchmark, with no agent, on the same pod (`GPU-4702993a…`) |
| Runs | 3 × 2 × 3 = 18 |

**Run order.** Fixed in advance, so that drift within the session is balanced across recipes. Seeds run
in the order 6, 7, 8. Within a seed, contract T runs before N. Within each seed and contract, the
recipe order rotates by seed:
- seed 6: baseline, Opus, G4;
- seed 7: Opus, G4, baseline;
- seed 8: G4, baseline, Opus.

**If no Opus attempt is eligible**, the G4 and baseline runs (12) still proceed, and G4 against
baseline is reported.

**The G4 source runs unchanged** under the current, fixed harness, exactly like the Opus winner and the
baseline. Under the N contract it must pass the same contract-API check as the Opus winner. Any
failure follows the pre-registration's §6: one re-run on the same seed, otherwise it is reported as
failed. No seed or run is added in response to results.

## 3. Analysis
For each contract, with paired differences per seed and mean with t-based 95% CI (n = 3, t = 4.303),
using the same decision rule as pre-registration §5:
1. **Primary (unchanged):** Opus winner − baseline.
2. **Secondary:** G4 − baseline.
3. **Secondary, the agent comparison:** Opus winner − G4. A CI entirely below 0 means the Opus
   winner is confirmed better; a CI entirely above 0 means the G4 winner is confirmed better;
   otherwise the comparison is inconclusive.

The secondary comparisons are reported with unadjusted CIs and are labelled as secondary.
Pre-registration §4 metrics and the inference and memory ratios are reported for all three recipes.

## 4. Disclosures fixed now
* **Unequal search budget.** Codex G4 had 22 T attempts; the Opus campaign has 25, unchanged as
  approved. The agent comparison is between each campaign's selected output, not a budget-matched
  search comparison. The write-up will report:
  - whether the Opus winner came from attempts 1–22;
  - the best eligible Opus discovery BPB within attempts 1–22.

  Both are descriptive only: no extra confirmation runs are pre-registered for this.
* **One search per agent.** There is one campaign per agent, so search-to-search variance is not
  measured. The confirmation removes seed and hardware differences from the comparison of the two
  selected recipes, but cannot say how often each agent would find such a recipe.
* **Hardware.** G4 was discovered on two other A40s. Its discovery BPB is not compared with
  numbers from this pod; only the same-pod confirmation runs are.
* **Other differences.** Effort setting, CLI and model differ by design: Codex `gpt-5.6-sol`/high
  in a container, against Claude `claude-opus-5-5`/xhigh on the host with file tools only
  (`audit/camera_ready_2026/12_GPU_READINESS_REPORT.md` §2.3).

## 5. Cost
Six more runs: about 6 × 11 min ≈ 1.1 A40-h plus evaluation, about $0.5. The confirmation total
becomes about 3.8 A40-h.

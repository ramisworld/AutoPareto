# Part 2, revision 6: provisional text for your review

**Status: AI-drafted, not approved.** The workshop expects Part 2 to be human-written. Claude restructured every passage tagged `r6` below in revision 6 (2026-11-15). Treat it as editorial suggestions: review each passage, rewrite it in your own words, and only then remove its `\humanauthor[...]` wrapper in `paper/main.tex`. To see the tagged passages as red boxes in the PDF, set `\showhumanmarkerstrue` (Part 2 then runs about one page long; that view is only an editing aid).

- **Word diff against revision 5:** `PART2_REVISION6.diff` (`[-removed-]`, `{+added+}`).
- **Revision 5 source and PDF:** `source_before_revision6/`.
- **Numbers** below are the values of the generated macros, and section, table and appendix numbers are those of the built PDF.

## What changed and what needs your approval

| ID | Section | Before revision 6 | Revision 6 change | Needs your approval |
|---|---|---|---|---|
| P2-02 | 2.1 Agent and Harness | approved by you 2026-11-10 | Your first two sentences kept verbatim. New: model vs coding application vs protected runner. Historical and Pair-A paragraphs split. Version numbers left to Appendix C. | Yes (substantive) |
| P2-03 | 2.2 Tools | approved 2026-11-10 | Grouped by era (historical discovery, matched-token verification, Pair-A). Controller sentence dropped (step list covers it). | Yes |
| P2-04 | 2.3 Research Loop | approved 2026-11-10 | Each step labelled Controller or Agent. Step 5 adds "verify the checkpoint". Function names moved to Appendix C. Figure pointer sentence dropped. | Yes |
| P2-06 | 2.4 Human Interventions | approved 2026-11-12 | One labelled paragraph per intervention: what happened, why, what changed, effect on conclusions. New fact stated: the 900 s timeout came after three consecutive agent-call timeouts at attempt 9 (audit doc 20, sections 13 and 14). Two-A40 detail shortened. | Yes (substantive) |
| P2-07 | 2.5 Verification | AI draft | Opens with why audits matter, then six checks as a list. | Yes |
| P2-08 | Table "Selection funnel" | AI draft | Moved from Part 2 to the appendix (now Table 6, next to the full funnel). | Yes (placement) |
| P2-09 | Selection and the winner's curse | AI draft | Explains the winner's curse and why we retrained. Exp 109 limitation kept. Exp 65 sentence moved to the FAQ, with "indistinguishable" changed to "do not differ resolvably". | Yes |
| P2-10 | Variability | AI draft | Pair-A and ablation speed-noise sentences moved to Appendix C (Inference and memory protocol). | Yes |
| P2-11 | Data integrity | AI draft | Wording unchanged, moved to Appendix C. | Placement only |
| P2-17 | 2.6 Reproducibility | AI draft | Adds https://github.com/ramisworld/AutoPareto and says every level needs files that are not yet public. | Yes (substantive) |
| P2-12 | 2.7 Significance | AI draft | Shorter. Removed the sentence repeating the abstract. | Yes |
| P2-13 | 2.8 Interpretability | AI draft | Width-rounding example moved to Appendix G. Session-log sentence moved to 2.1. | Yes |
| P2-14 | 2.9 Limitations | AI draft | Shorter "addressed" clause. Pilot caveat removed here because 2.10 states it. | Yes |
| P2-15 | 2.10 Pair-A | AI draft | Split into The question, What the arms explored, A harness defect. The N* rule moved to Appendix C. Batch sentence simplified. | Yes (substantive) |
| P2-16 | What the speed metric does not time | AI draft | Shortened to the harness lesson. Codex/Opus comparability now only in 2.4 and the FAQ. | Yes |

**Facts to confirm personally** (they describe your own actions or recollection):
- Codex usage was "nearly exhausted" when seed 42 started in Cursor (session log: 1% left).
- You raised the Opus timeout after three consecutive timeouts at attempt 9.
- The private session logs contain access credentials, which is why they stay private.
- Nothing from the camera-ready evidence is on a pushed branch yet. Local git refs show none of `results/camera_ready_2026`, the E1 analysis script or the Pair-A controller on `origin/*`. Please confirm before release, since the remote refs may be stale.

## Provisional Part 2 text

# 2 System Design and Meta-Analysis


## 2.1 Agent and Harness

> **[P2-02 r6]**

My role was to design the research framework, define the objectives, compute budgets and evaluation rules, and maintain the experimental infrastructure. The AI agents independently proposed hypotheses, modified Transformer architectures and training settings, and selected which candidates to explore. Three parts of the system should be kept apart. The *model* proposes each change. The *coding application* is the interface through which it reads files and runs commands. The *protected runner* trains and evaluates the code, and the agent cannot change it.

**Historical campaigns.** Most historical AutoPareto and AutoResearch experiments used GPT-5.6 (`gpt-5.6-sol`, high reasoning effort) through the Codex desktop application. For the first five attempts of the seed-42 campaign, the same model and setting were accessed through Cursor because Codex usage was nearly exhausted. In every case a protected experiment runner executed and evaluated the proposed code on the GPU. The agent started the runner over SSH, and isolation relied on the program's instructions plus hash checks: every protected file was hashed before and after each attempt, and any change outside `train.py` rejected the run.

**Pair-A campaigns** (§2.10). Here a controller called the agent and launched every run itself. The Codex arm ran the same GPT-5.6 model through Codex CLI, in a fresh container for every call that held five staged files. The Opus arm ran Claude Opus 5.5 through Claude Code, Anthropic's command-line coding agent, with file tools only and no shell. Both arms used the same prompts and output schemas, made two agent calls per attempt and could write only `train.py`. The Opus arm was isolated by command-line permissions, file modes and hash checks, tested by a canary before the campaign. No agent saw downstream results. Versions, timeouts and per-attempt attribution are in Appendix C. Session logs survive for every campaign and stay private because they contain access credentials.

## 2.2 Tools

> **[P2-03 r6]**

All training and measurement ran in BF16 on NVIDIA A40 GPUs. Historical discovery (July and August) did not log its environment, while the repository's lockfile pinned PyTorch 2.9.1. The matched-token verification (201.3M-token panel, E1, Opus retraining) used Python 3.10.18 with PyTorch 2.9.1. Pair-A and the inference ablations that reuse its benchmark ran in a separate Python 3.12 runtime with PyTorch 2.8.0 (2.9.1 in the September pilot). Table 2 records every era.

## 2.3 Research Loop

> **[P2-04 r6]**

Each attempt follows the eight steps below. The agent acts in steps 2 and 8 and, historically, labels the outcome in step 7. Protected code does everything else (Appendix C names the functions).

1. *Controller:* show the agent this campaign's records, the current frontier and the parent's source.
2. *Agent:* return a hypothesis, the intended change, a change category, the chosen parent and a patch to `train.py`.
3. *Controller:* validate. The parent must be on the current frontier (historical) or parent-eligible in the same arm (Pair-A). Protected files and the source must match their SHA-256 hashes, and only `train.py` may change.
4. *Controller:* train under the contract, 300 s or, in the Pair-A fixed-token arm, N* counted tokens.
5. *Controller:* verify the checkpoint, measure validation BPB, check that cached decoding matches a full recomputation, and measure decode throughput and peak allocated memory.
6. *Controller:* append an immutable record of the hypothesis, patch, hashes, metrics and any failure text. A failure still uses up the attempt.
7. *Controller and agent:* update the archive. Historically a completed candidate was retained unless another completed candidate or the registered same-seed baseline dominated it on raw BPB, decode speed and memory, with no noise margin. The agent labelled each outcome, and the runner rejected any label that disagreed with this rule.
8. *Agent:* choose the next parent among the retained (historical) or parent-eligible (Pair-A) records.

Without a noise margin the archive keeps noise-level trade-offs. In AP-Q's campaign A9 and A10 build the same model, yet A10 was retained. In Pair-A every quality-valid record was parent-eligible even when its speed was uncertain.

## 2.4 Human Interventions

> **[P2-06 r6]**

I chose the objectives, the 300-second contract, the matched-exposure design, the recipes for the panel, E1 and the Opus tests, the comparators and the interpretation. I made or approved each intervention below, and AI coding assistants carried out several of them at my instruction. None of them counts as autonomous agent research.

**(i) Seed-42 rollback.** Attempt 4 of the seed-42 AutoPareto campaign failed a cache-correctness check, was labelled dominated, and attempts 5–8 then ran on 2026-07-28. The check itself was faulty. After it was repaired I discarded attempts 5–8, because they had followed a frontier that wrongly excluded attempt 4 (ids 88–90 are absent). Attempt 4 was re-measured on a recreated checkpoint and became retained, and attempt 5 was replayed unchanged. Attempt 10 was re-adjudicated after an FP32 evaluator amendment, and the program text changed after attempt 4. I started seeds 0 and 1 only after reviewing seed 42. AP-Q comes from seed 0, which none of this touched.

**(ii) AutoResearch reference.** In the 50-attempt AutoResearch campaign, a cache-correctness check later found faulty discarded 27 trained candidates. AR-ref comes from a 5-attempt follow-up under a revised program. This is one reason AR-ref differs from our campaigns in more than the objective (§1.2).

**(iii) Pair-A Codex arm.** Pod and provider failures forced requeues, the inference adapter was amended, the configuration changed before launch, and a later restart adjudicated failures inconsistently. We use this arm only as a second instance of the speed lever, never to rank agents.

**(iv) Pair-A Opus arm.** A defect in the protected FP32 correctness adjudication was fixed mid-campaign and the affected attempt re-run. After three consecutive agent-call timeouts at attempt 9, I raised the call timeout from 600 to 900 s for the rest of the arm (Codex ran at 600 s). Attempt 9 was charged as a candidate failure, and timed-out calls made before any candidate were requeued. The arm was extended from 25 attempts to 25 valid results (27 attempts). Both arms moved to a second A40 once, Opus after A1 and Codex after A9. We also report the frozen first 25 attempts, and neither the timeout nor the extension changes a conclusion (Appendix H).

**(v) Failure accounting.** The same tokenizer failure was counted differently in two campaigns. In seed 0 it consumed attempt A17, while in seed 42 it was relabelled as a setup failure. AP-Q's campaign thus carries the stricter accounting.

**(vi) E1 hardware.** The A40 on which E1's environment check (E0) passed was no longer available, so E1 ran on another A40 with the same driver. The deviation is recorded, and throughput matched E0 (Appendix C).

**(vii) Opus retraining.** Opus had rewritten the training-loop driver for the time contract, so its sources could not run under a token budget unchanged. A scripted driver mapping and a storage export, pre-registered with the plan, made them runnable while keeping model and optimizer code byte-identical (Appendix C).

The AP-Q lineage (seed 0) has no recorded scientific intervention: its program, evaluator and records were not changed. Interventions whose authorship cannot be reconstructed are excluded from claims of autonomous contribution.

## 2.5 Verification

> **[P2-07 r6]**

A number can be wrong without any run failing: a result may come from the wrong script, a run may see the wrong number of tokens, or an evaluation may use another run's checkpoint. A separate audit pipeline therefore re-derives each reported number from the frozen records, without launching experiments, and checks six things.

- *The code behind each model.* For each of the 21 panel runs the hash of the training source must match the training record and the public manifest.
- *Tokens and updates.* Each run must process the exact token total, with updates equal to tokens divided by tokens per update, so recipes compared at equal tokens really saw equal tokens.
- *Checkpoints and evaluations.* Each downstream result must come from its own run's checkpoint, and every metric must point in its stated direction.
- *Reported results.* Opus's frontier was re-measured on a fresh A40, where protected BPB reproduced bit for bit, and an AP-Q checkpoint reproduced all nine August downstream metrics. 118 panel quantities agree with an independent audit to 10^-15.
- *Hypotheses and parents.* Scripts check every quoted agent excerpt verbatim against its attempt record and every drawn lineage edge against a recorded parent link.
- *Statistics.* E1 was analysed by its pre-registered script, the exploratory split reproduces its paired differences exactly, and the Opus tests' pre-registered analysis reproduces byte for byte from the archived raw records.

Every result number in Part 1, every table and every data figure is generated by scripts in `analysis/`. Figure 1, the glossary and the discovery and canary environment rows are hand-authored. Data-integrity checks are in Appendix C.
> **[P2-09 r6]**

**Selection and the winner's curse.** Each discovery score comes from one noisy run. When the best of many noisy scores is picked, the winner tends to be a recipe that was also lucky, so its score overstates its quality. As this winner's curse predicts, every re-run candidate scored 0.0014–0.0087 BPB worse than at discovery. This is why we retrained the selected recipes independently, on new seeds and a held-out shard, instead of trusting their discovery scores. Selection happened in stages (Appendix Table 6). The representative rule was committed on 2026-07-27, before AutoPareto's first attempt, and yields 11 representatives. On 2026-08-10, after discovery, a global four-role selector cut these to four seed-0 recipes. The confirmation campaigns were created on 2026-08-11/12 UTC, so selection preceded the seeds-0–2 confirmation. The 201M panel was assembled around 2026-08-21–25, after those results were known, and kept three of the four roles. No rule was recorded for omitting the lowest-memory role (exp 124). **Not confirmed.** Exp 109, the best seed-42 recipe, keeps the full 50.3M-parameter model at batch 65,536 and reached 1.0940 on 84.1 M tokens (AR-ref: 1.0950 on 88.9 M). It was a pre-registered representative but was never confirmed, and it is the most plausible AutoPareto quality candidate that does not rely on extra exposure. E1 tests AP-Q, not the best possible AutoPareto recipe, and our conclusions do not extend to unconfirmed recipes.
> **[P2-10 r6]**

**Variability.** Within the agent campaigns the best recipe beats the second by at most 0.0003 BPB and the fifth by at most 0.0021, while the pooled fixed-time seed SD is 0.0022. A campaign's "best" recipe is therefore not resolvable from its runners-up. The unchanged baseline differs by 0.024 BPB between seeds 0 and 42, and by 0.0035–0.0070 BPB at the same seed between sessions. Under the fixed-time contract one recipe processed up to 18.7% different token counts across sessions. In E1 the within-recipe seed SD is at most 0.0045 BPB, an order of magnitude below the matched-token gap. Speed-measurement noise in Pair-A and in the ablations is reported in Appendix C.

## 2.6 Reproducibility

> **[P2-17 r6]**

The code and artifacts will be released at https://github.com/ramisworld/AutoPareto. Much of the evidence, including the camera-ready run records, is still local, so every level below needs files that are not yet public. Commands are in Appendix C. *Level 1, artifacts* (no GPU). One script, `paper/build_camera_ready.sh`, regenerates every number, table and figure from the archived records and checks each against the committed audit. The pre-registered E1 and Opus analyses need only their run records, while the full build also needs the database snapshot and the campaign histories. *Level 2, recipes* (one A40). The frozen recipe sources, hashed plans and executors retrain AP-Q, AR-ref and Opus's recipes under the recorded token or time contract. The training-shard revision and one GPU kernel are not pinned in code, so exact losses may differ slightly. *Level 3, agent campaigns.* The Pair-A controller, prompts, output schemas and permission checks run a comparable campaign with Codex CLI or Claude Code. Proprietary model state means that a new campaign will not repeat the same attempts, and pod setup is partly manual. The historical campaigns were operated by hand through desktop agents and have no single entry point.

## 2.7 Significance of result

> **[P2-12 r6]**

Matched-exposure verification shows that the quality ranking depends on the contract, while the efficiency gains hold under every contract we tested. At no pre-registered token budget is AP-Q better per token, so its fixed-time standing reflects its extra exposure, not better learning. Opus pushed the same lever further: its recipes were worse still per token, and most of its decode gain came from graph replay whose recording the metric did not time. A benchmark that ranks agents under a time budget therefore partly ranks how fast their candidates train. Quality gains should be rechecked at controlled exposure, and speed gains with end-to-end timing.

## 2.8 Interpretability

> **[P2-13 r6]**

Every preserved attempt record keeps the agent's hypothesis, intended change, applied patch, parent, source and parent hashes, status, raw metrics and failure text. A headline number can therefore be traced to a source file, to the edit that produced it and to the hypothesis recorded before the run (Figure 2, Appendix G). We keep apart what the agent predicted, what it changed, what the evaluator measured and what our retrospective analysis concludes, and we quote agent motives rather than infer them. The records also expose gaps between intent and execution (Appendix G).

## 2.9 Limitations

> **[P2-14 r6]**

*Addressed since review:* the confounded reversal, the single matched horizon and the fixed-time variability (E1 and above). *Remaining:* one GPU type and small models, so we make no scale-transfer, cross-GPU or energy claim. Matched-token verification compares complete recipes, so architecture and optimizer changes stay bundled, and in E1 tokens and updates change together. E1 resolves the gap with three seeds but not its slope with tokens, and its exposure–architecture split is exploratory. A pre-registered full-size candidate (exp 109) was never confirmed. Each Pair-A agent ran one seed under slightly different conditions (§2.4), so we do not rank agents. The Opus retraining covers one budget and replaces time-fraction learning-rate progress with the token fraction. The ablations measure one workload on one host, and graph replay and fusion were not applied to the baseline or AR-ref. Historical comparators differ in search space, budget, seed and failure history, so they support no search-policy ranking.

## 2.10 Pair-A: controlled agent campaigns

> **[P2-15 r6]**

**The question.** Does the training contract change what an agent pursues? We piloted a prospective test of this hypothesis (H2). Two arms started from identical code at seed 3, with programs that differ only in the paragraph that states the contract. Arm T trains each candidate for 300 s. Arm N trains each candidate on N* counted tokens, set from baseline calibration runs (Appendix C). N* was 88,604,672 in the first launches and 85,458,944 after recalibration on a new A40 from 2026-09-09. Five Codex launches ran between 2026-09-05 and 2026-09-17, with protocol changes between them. Part 1 uses the fixed-time arm of the 2026-09-14 restart (22 attempts per arm).

**What the arms explored** (Appendix Table 9). In each of the three launches in which the agent changed the architecture, the fixed-token arm's best model was larger: 85.9 M against 50.3 M parameters, 35.7 M against 16.3 M and 47.2 M against 24.6 M, or 1.7 to 2.2 times as large. Only fixed-token attempts grew the model, to 10 and 12 layers, expecting a better BPB "by increasing model capacity". Under fixed time the agent instead cut depth and width and trained on up to 175M tokens. Both arms reduced the batch, in different launches, so batch size does not separate them. Each launch ran each arm once with a shared seed and starting code, the harness changed between launches, and six fixed-time attempts were lost to the defect below. The pilot therefore shows an observed tendency, not an estimated effect of the contract.

**A harness defect.** In the lineage reported at submission, 2 of the first 4 fixed-time attempts are invalid. When a candidate's training loop ended before the trusted stop, the harness re-ran the training body and saved a freshly initialized model (step 0 reported against 1,264 observed updates, BPB 3.2159). The controller still marked these runs quality-valid and parent-eligible, and the agent's next hypothesis blamed its learning rates. The Opus harness treats such an early exit as a candidate failure.
> **[P2-16 r6]**

**What the speed metric does not time.** Opus recorded its decode CUDA graph, and later compiled regions, inside the untimed prefill (§1.5), which its program calls "diagnostic". The rules allowed this, so we report both speeds and recommend scoring prefill and checking outputs inside the timed windows.


# Internal audit: not an anonymized submission asset

# Part 2 human-author dossier

**HUMAN AUTHOR MUST REWRITE/VERIFY THIS SECTION BEFORE SUBMISSION.** The PDF contains explicitly provisional agent prose. The workshop expectation comes from https://automlr.com/ and the official Overleaf template.

## Facts and attribution

| Event / time | Evidence | Attribution / verification |
| --- | --- | --- |
| Human-defined original multi-objective goal; exact first request not recovered | DB campaigns.program_instructions; historical program snapshots | Human design described in user brief; original authorship/time must be verified |
| Historical source generation and decisions, July--August 2026 | DB experiments.hypothesis, code_diff, parent_experiment_id; agent_actions | Recorded agent contribution; not an independently observed human action |
| Recipe134: output-head LR edit from parent132 | DB experiment134; exact recipe SHA in panel | Agent-stated hypothesis; prior architecture lineage retained |
| Recipe292: confirmation, 2026-08-12 00:17:52 to 00:25:24 (DB clock timezone unspecified) | DB experiment292 parent134; training seed1 | No new architecture search at ID292 |
| Equal-token retraining August26; downstream August27 | Per-run run_metadata.json; downstream run timestamps | Post-search protocol and evaluation requested by humans; verify original request |
| Budget comparison design and calibration; multiple revisions | study/contracts_2026/protocol.md; provenance/n_star*; runs/pair-a/provenance | Human selected the scientific question; repair authorship needs original messages |
| Latest frozen manifest 2026-09-06T02:00:00Z | runs/pair-a-official-20260906-fixed-warmup-v2/provenance/final_pre_study_manifest_pair_a_20260906_fixed_warmup_v2.json | Machine-recorded freeze timestamp; reconcile with launch times |
| Latest T1--T4/N1--N4 | runs/pair-a-official-20260906-fixed-warmup-v2 | Hypotheses and patches recorded as scientist outputs; exact completion UTC missing from history |
| Paper audit and retrospective analysis freeze 2026-09-06T12:38:02.850468+00:00 | research_question_freeze.md; generated validation report | This assistant; not experimental preregistration |
| Human scientific verification | No signed checklist available | PENDING; never infer from task approval |

## Executed setup

```json
{
  "contracts": {
    "N": {
      "active_n_star": 88604672,
      "token_quantum": 524288,
      "training_seed": 3
    },
    "T": {
      "protected_training_seconds": 300,
      "training_seed": 3
    }
  },
  "scientist": {
    "provider": "headless_codex_cli",
    "model": "gpt-5.6-sol",
    "reasoning": "high",
    "campaign_histories_shared": false,
    "fresh_ephemeral_invocations": true,
    "configuration": {
      "model": "gpt-5.6-sol",
      "reasoning": "high",
      "scientist_bin": "/usr/local/bin/codex",
      "codex_version": "codex-cli 0.153.0",
      "execution_backend": "docker",
      "container_engine": "docker",
      "container_image": "autopareto-scientist-codex:0.153.0",
      "container_image_digest": "sha256:877dae34229cfab2888c13e500ca9e8cc2b7f25cc3789df0664ed3fc0bef2ae2",
      "sandbox_mode": "danger-full-access",
      "approval_policy": "never",
      "timeout_seconds": 600.0
    }
  },
  "lineage": {
    "policy": "same-arm parent branching",
    "cross_arm_parents": false,
    "parent_source_staging_assertion": "parent ID -> stored source SHA -> staged source SHA",
    "quality_valid_speed_uncertain_source_parent_eligible": true,
    "speed_uncertain_pareto_eligible": false
  },
  "runtime": {
    "python": "/usr/local/bin/python",
    "python_version": "3.12",
    "torch": "2.9.1+cu128",
    "cuda": "12.8",
    "pyarrow": "23.0.1",
    "candidate_uid": 2001
  },
  "registered_artifacts": {
    "baseline_registration": {
      "path": "runs/pair-a-seed3-baseline-registration.json",
      "source_sha256": "b2fb8788f02517af8bc2fcfa059fb799a6974c9050d00e322168ec414fac4fab",
      "record_scope": "same_seed_baseline"
    },
    "registered_cache": "/var/cache/autopareto/pair_a_registered_cache",
    "data_manifest_path": "study/contracts_2026/provenance/data_manifest.json",
    "data_manifest_sha256": "6a6d9c60b99b80884553939932e91e93364e548e102cf71b545589317f55eb88",
    "tokenizer_sha256": "888c9fa865c72c25f9c5e9467136a9cc8655253db459fb97481deb326681c629",
    "flash_attn3_revision": "e29f138fc363b396e5d2706c8a5f6fa7d36f41e0"
  }
}
```

## Exact operator invocation (private)

The operator reported the following launch command. Verify it against controller diagnostics before quoting it as executed provenance. It is retained here rather than in the anonymous PDF because it contains a private host and filesystem paths.

```sh
cd /Users/Rami/Documents/dev/autopareto && python3 -m study.contracts_2026.campaign_controller run --execute --state-root /Users/Rami/Documents/dev/autopareto/runs/pair-a-official-20260906-fixed-warmup-v2 --baseline-record /Users/Rami/Documents/dev/autopareto/runs/pair-a-seed3-baseline-registration.json --manifest /Users/Rami/Documents/dev/autopareto/runs/pair-a-official-20260906-fixed-warmup-v2/provenance/final_pre_study_manifest_pair_a_20260906_fixed_warmup_v2.json --manifest-sha256 35ea95fad1f3e4f345852d03eb1daadb1e348e352c57fa9a2fd11dff8ed9d12c --scientist-bin /usr/local/bin/codex --model gpt-5.6-sol --reasoning high --scientist-timeout-seconds 600 --a40-host <pod-id>-<suffix>@ssh.runpod.io --a40-identity /Users/Rami/.ssh/id_ed25519 --a40-repository /opt/autopareto_trusted --a40-cache /var/cache/autopareto/pair_a_registered_cache --a40-state-root /var/lib/autopareto/pair_a_official_20260906_fixed_warmup_v2
```

## Required corrections to the narrative

- Current prompts specify disposable warm-up in a separate process; all 8 completed histories record same_worker_process=true with reconstructed initial state. Do not call this separate-process execution.
- Runtime validation provenance identifies historical_protected_in_process_evaluate_bpb; do not substitute the current model-host description.
- Speed-uncertain candidates are parent-eligible under the executed manifest. Do not say every parent is a measured Pareto member.
- Common same-seed calibration baseline remains from an earlier measurement path. Treat improvements relative to its BPB as reported observations, not a rigorously remeasured paired baseline.
- Do not pool old progress-interface failures, the prior fresh launch, and the fixed-warmup-v2 launch. Restart decisions may affect the research trajectory.
- Hidden provider internals are unavailable, although staged files and recorded model/reasoning match.
- The new study has no sealed downstream results and no completed architecture-by-budget crossover.

## Exact prompts and audit links

Recorded initial requests are exported under audit/prompts/T and audit/prompts/N. Exact instruction diff: audit/executed_program_md.diff. Additional diffs include campaign_view.json, train.py, preparation and runtime. Per-attempt source paths, hashes, constants and hypotheses are in evidence_inventory.md and generated/budget_attempts.json. Original unabridged records remain nested as raw_record. No manuscript or audit is sent back to the scientist.

## Human rewrite checklist

Rewrite the entirety of Part 2 in your own words; confirm agent identity, autonomy boundaries, launch/restart decisions, repairs, any cross-arm information leakage, source selection, and verification procedures you actually performed. Explain which steps you personally verified. Check all automated edit categories against code. Resolve disclosure boxes individually. Remove the provisional banner only after completion. Never submit this internal dossier: it contains local paths and identifiers.

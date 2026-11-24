# Release artifact index

Status: `INTERNAL AUDIT BUNDLE; PUBLIC RELEASE NOT READY`
Audit date: 2026-09-02

## Public-facing artifacts already present

| Artifact | Path | Status |
|---|---|---|
| README and project state | `README.md`, `PROJECT_STATE.md` | Present; summaries bounded by ledger |
| Reconstructed ledger | `docs/RESEARCH_LEDGER.md` | Present; internal audit version |
| Equal-token results | `results/phase1/phase1_equal_token_results.json`, `.csv`, `.md` | Present; aggregate/per-seed results |
| Phase 1 protocol | `evaluation/phase1_public_manifest.json`, `evaluation/phase1_equal_token_preflight.json` | Present |
| Downstream aggregate data | `results/phase1/public_downstream_metrics.json` | Present where available |
| Figures | `plots/` | Present and regenerable from aggregates |
| License | `LICENSE` | Present; SHA `bbb839815589a73c55432f4a672ea8f34358dce45fe049cb608ab687812160e9` |

## Internal evidence required for audit/reproduction

- Recovered populated registry:
  `results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db`.
- Candidate source snapshots under `results/candidates/`.
- Checkpoints under `results/checkpoints/` and recovered snapshot paths.
- Raw event/log/sample directories under `results/events/`, `results/logs/`, and
  `results/samples/`.
- Per-run Phase 1 JSON, timing, telemetry, and milestone files under the recovered
  snapshot.

These artifacts are private/untracked and include platform-specific absolute paths.
They must not be published without privacy, licensing, and secret-scanning review.

## Blocking release gaps

1. Restore `results/phase1/token_stream_verification.json` or mark the runtime stream
   verification permanently unavailable in the release statement.
2. Package a portable populated database or documented export; the canonical local
   `results/phase1/downstream_200m.db` is zero bytes.
3. Decide whether data shards, tokenizer files, checkpoints, raw logs, and caches may
   be redistributed under their respective licenses.
4. Remove private absolute paths and verify no credentials/secrets are included.
5. Add final anonymized paper, supplement, checklist, citation, and artifact DOI or
   equivalent persistent host when submitting.
6. If claiming autonomous research, disclose human interventions and the limits of
   the preserved agent record.

## Not included in a public release by default

Credentials, SSH material, private prompt contexts, downstream raw logits/choices,
private database/WAL files, unlicensed data/weights, and full internal agent
transcripts. Absence of these artifacts is a release policy choice, not evidence that
the corresponding claims were independently reproducible.

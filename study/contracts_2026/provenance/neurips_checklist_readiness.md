# NeurIPS/AutoMLR readiness matrix

Checked: 2026-09-02 against the official NeurIPS 2026 Call for Papers and Main Track
Handbook and the official AutoMLR 2026 workshop page. This is a preparation matrix,
not a submission declaration.

## Requirements mapped to this project

| Requirement | Status | Evidence/action needed |
|---|---|---|
| State claims accurately and include limitations | READY FOR DRAFT | Ledger claim map; do not use invalidated/unidentified claims |
| Disclose sufficient experimental details | PARTIAL | Protocol/manifests are present; runtime token hash and portable raw registry remain open |
| Complete paper checklist | NOT YET | Fill the current venue template at submission time |
| Anonymize main-track submission and supplement | NOT YET | Remove repository identity/author metadata and private paths |
| Include code/dependency specification | PARTIAL | `pyproject.toml`, `uv.lock`, environment and harness registries present; package a reviewer-safe supplement |
| Provide reproduction avenue | PARTIAL | Aggregate results reproducible; full private harness/data/weights are not public |
| Data licensing/privacy/consent review | OPEN | Review dataset/tokenizer licenses and document restrictions |
| Model/checkpoint release decision | OPEN | Decide release, access controls, and license; weights are currently private |
| Agent/LLM use disclosure | READY FOR DRAFT | Human/agent attribution and missing transcript are explicit |
| Human responsibility for paper claims | REQUIRED | Human authors must verify claims, figures, references, and tool outputs |
| Negative results and failures | READY FOR DRAFT | Failure ledger retains OOM, cache, stability, timing, and invalid-calibration events |
| Environmental/resource accounting | PARTIAL | Nominal slot ledger exists; energy/carbon/billing/human time missing |
| AutoMLR qualifying ML result | CONDITIONAL | Historical AP292 trade-off can qualify only if final paper states the bounded result and the agent’s decisive contribution accurately |
| AutoMLR autonomous-system account | PARTIAL | Harness/trajectory/intervention evidence present; full transcript unavailable |
| AutoMLR 4/4/1 structure | NOT YET | Apply to final workshop manuscript |
| AutoMLR author/discussant/in-person requirements | HUMAN ACTION | Confirm authorship, review commitment, and attendance separately |

## Venue facts recorded

- NeurIPS 2026 main-track CFP lists abstract deadline May 4, 2026 AOE and full paper
  deadline May 6, 2026 AOE; those dates are historical as of this audit.
- NeurIPS 2026 materials require the paper checklist and sufficient information for
  reproducing the reported results, with anonymized submission material and code/data
  handling requirements described in the handbook.
- AutoMLR 2026 is described as a non-archival NeurIPS workshop. Its page requires a
  qualifying ML result plus a detailed autonomous-system account, labelled 4-page,
  4-page, and 1-page parts, human authorship/responsibility, disclosure of agent
  involvement, and in-person participation.

## Decision

`PREPARATION DOCUMENTED; PAPER/RELEASE NOT READY.` The ledger is ready for internal
pre-campaign freeze, but this matrix must be revisited immediately before any venue
submission because policies, deadlines, and submission portals are time-sensitive.

# Building the camera-ready draft

## One command

```sh
cd /Users/Rami/Documents/dev/autopareto-camera-ready
AUTOPARETO_PAPER_PYTHON=/path/to/python \
AUTOPARETO_EVIDENCE_ROOT=/Users/Rami/Documents/dev/autopareto \
sh paper/build_camera_ready.sh
```

- **Python.** It needs `matplotlib` 3.10.9, `numpy` 2.2.6 and `pypdf` 6.17.0 (`paper/requirements.txt`). If `AUTOPARETO_PAPER_PYTHON` is unset, the script uses `paper/.venv/bin/python`.
  - This draft was built with a scratch virtualenv: CPython 3.10.20, with site-packages copied from the original repository's `paper/.venv`.
- **Evidence tree.** It is read-only, and defaults to `/Users/Rami/Documents/dev/autopareto`. The build reads:
  - the snapshot DB, opened with `mode=ro&immutable=1`;
  - `phase1_equal_token_results.json`;
  - the per-run panel files;
  - the frozen sources under `results/candidates/`;
  - the run logs under `results/logs/`;
  - the selection manifests, through `git show autopareto/research-infrastructure:…` (read-only).
- **Tectonic** 0.17.0 must be on `PATH`. On a machine whose Tectonic cache is cold, Tectonic downloads TeX support files from its bundle server. That happened once for this draft.

The build launches no experiment, uses no GPU, and calls no paid API.

## What each step writes

| Step | Script | Output |
|---|---|---|
| 1 | `analysis/cr_evidence.py` | `paper/camera_ready/data/cr_evidence.json` (extracted records plus provenance hashes) and `cr_stats.json`. Every derived number is checked against `audit/camera_ready_2026/supporting/{statistical_audit,claims_recomputation}.json`: 118 quantities, maximum \|Δ\| 1.1e-16. The build aborts on any mismatch above 1e-9. |
| 2 | `analysis/lineage_labels.py` | `paper/generated/lineage_edit_labels.json` and a copy in `paper/camera_ready/data/`. Figure 1 edit labels come from the frozen sources via each source's own `build_model_config`, and are checked against the run logs. A17's source is rebuilt from `experiments.code_diff`; that replay reproduces 24 of 24 frozen sources byte-for-byte. |
| 3 | `analysis/cr_tables.py` | `paper/generated/camera_ready/{cr_macros,table_main,table_reversal,table_funnel,table_env}.tex` |
| 4 | `analysis/cr_figures.py` | `paper/figures/cr_{pareto,reversal,decision_graph}.{pdf,png}` and the poster versions in `paper/camera_ready/poster_figures/` |
| 4b | `analysis/cr_v2.py` | E1, Pair-A agent-arm and Phase 5 numbers (5A Opus retraining, 5B inference ablations, from `results/camera_ready_2026/phase5/analysis/phase5_analysis.json`; Figure 2c and the `cr_ablation` poster figure): `paper/camera_ready/data/cr_v2.json`, `generated/camera_ready/{cr_macros_v2,table_e1,table_agents,table_env_v2}.tex`, `paper/figures/cr_{main,e1,agents}.*` and poster versions. Reads the committed E1 analyses and Phase 2 JSON in this worktree, and the two Pair-A `history.jsonl` files under the evidence root (read-only). |
| 5 | `analysis/generate_figures.py` | `paper/figures/autopareto_lineage.*` (derived labels) and `research_loop.*`. It also rewrites the legacy, unused figures. |
| 6 | `tectonic main.tex` | `paper/main.pdf` |
| 7 | `analysis/cr_page_check.py --write` | `paper/camera_ready/page_check.json`, the per-part page counts from `main.aux` labels |
| 8 | copy | `paper/camera_ready/autopareto_camera_ready_draft.pdf`, plus the lineage and loop figures in `poster_figures/` |

## Editing

- **Prose.** Edit `paper/main.tex` directly. The abstract and the Primary Result are now inline.
- **Numbers.** Never type a number into the paper by hand. Every number in prose comes from a `\CR…` macro in `generated/camera_ready/cr_macros.tex`. To change one, fix the computation in `analysis/cr_evidence.py` or `analysis/cr_tables.py` and rebuild.
- **Part 2.** Every Part 2 passage drafted for the camera-ready is wrapped in `\humanauthor[P2-xx]{…}`. Captions use `\humancaption{P2-xx}`.
  - The workshop requires Part 2 to be human-written. Rewrite each passage, then delete its wrapper. `PART2_HUMAN_AUTHORSHIP.md` lists the facts and evidence for each ID.
  - To preview the final layout without the boxes, set `\showhumanmarkersfalse` near the top of `main.tex`. Both layouts were checked: 4 / 4 / 1 pages.
- **Release marker.** `\releasepending` marks the artifact-release sentence in Appendix A. Remove it once the release exists.
- **Do not run the legacy pipeline.** `paper/refresh_paper.sh`, `analysis/refresh.py` and the `generate_reports.py` and `generate_tables.py` part of it belong to the submitted-era pipeline.
  - `refresh_paper.sh` and `refresh.py` are guarded, and they also need `results/snapshots/…` inside the worktree, which is absent.
  - `generate_reports.py` and `generate_tables.py` still write `generated/abstract.tex`, `primary_result.tex`, `results_macros.tex` and the legacy tables. `main.tex` no longer reads any of them.
  - `paper/EDITING_GUIDE.md` describes that legacy flow. For the camera-ready, this file supersedes it.

## Project convention not applied

The project memory says to copy the built paper to `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf`. This task forbade writing outside the worktree, so the draft exists only at `paper/camera_ready/autopareto_camera_ready_draft.pdf` and `paper/main.pdf`. Copy it once the human author approves the draft.

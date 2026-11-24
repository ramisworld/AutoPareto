# AutoPareto working paper

> **Camera-ready (2026-11):** the camera-ready draft is built with `paper/build_camera_ready.sh`; see `paper/camera_ready/BUILD.md`, `CHANGELOG_CAMERA_READY.md`, `SOURCE_RECONCILIATION.md`, `REVIEWER_RESPONSE_MAP.md` and `PART2_HUMAN_AUTHORSHIP.md`. The legacy `refresh_paper.sh` flow described below is disabled for the camera-ready.

Start with **main.pdf**, **FINDINGS_FOR_HUMAN.md**, and **part2_human_author_dossier.md**. This is a technically complete working manuscript. Part 2 is explicitly provisional AI-written prose; a human must rewrite it and verify the disclosures before submission.

The initial audited snapshot contains 21 equal-token verification runs and eight current budget-study attempts. The historical recipe-level efficiency–quality trade-off is supported; the general budget-effect hypothesis remains unestablished. Earlier launches are separately preserved. Read `generated/validation_report.json` for exclusions and corrupt input reports.

Run from any directory:

```sh
/path/to/autopareto/paper/refresh_paper.sh --cutoff 2026-09-06T23:59:00+12:00
```

From the repository root, use `./paper/refresh_paper.sh`. The wrapper uses an isolated Python environment with pinned dependencies. Install Tectonic or latexmk for compilation. `--inventory` rehashes all evidence including checkpoints; `--no-compile` performs analysis only. This pipeline never launches experiments, executes candidate programs, or changes study records.

`headline_changes.md` compares generated quantities. Timestamped numerical/provenance snapshots live under `audit/refreshes/`. The hypothesis file remains unchanged across refreshes. Budget histories without authoritative completion timestamps are eligible only when first observed before cutoff; observation bounds persist by record hash. A late first observation alone cannot prove pre-cutoff completion. Namespaces are never pooled. Changing the selected official launch requires explicit scientific reconciliation in `analysis/common.py` and the paper, not silently selecting whichever run looks best.

The public-facing working source bundle includes only the LaTeX inputs, official style, bibliography, figure PDFs, and manuscript PDF. Internal dossiers, raw data, generated database dumps, environment details, and local paths are excluded. Do not upload the entire `paper/` directory. No publication or submission has been performed.

The exact style was recovered from the official workshop's linked Overleaf template; provenance is in `template/provenance.json`. Automated layout, citation, text, and metadata checks are recorded in `generated/pdf_qa.json`. Rendered-page inspection is tied to the exact PDF hash in `generated/visual_qa.json`; every new PDF requires renewed visual review. Current layout is 4/3/1 content pages plus one reference page, within the separate 4/4/1 limits.

Required evidence and reviewer files are accompanied by `related_work_verification.md`, `historical_baselines.md`, exported exact T/N prompt diffs, raw-field provenance, and the 15-test regression suite under `analysis/tests/`. `analysis/verify_evidence.py` independently recomputes headline quantities from raw step timing, training, inference, and downstream artifacts.

The official workshop website lists September 5 without a timezone. The separately retrieved public OpenReview group lists September 6, 11:59 UTC, matching September 6, 23:59 Pacific/Auckland. The analysis cutoff does not itself confer permission to submit late, and current portal availability must be checked by the human author.

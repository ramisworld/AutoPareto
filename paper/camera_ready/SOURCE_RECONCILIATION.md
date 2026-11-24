# Source reconciliation (discrepancies D18 and D22)

**Decision.** The camera-ready was started from `paper_backups/automlr_overleaf_reconstructed/`. It is the only source that reproduces the local copy of the submitted PDF. It differs from the pre-camera-ready `paper/main.tex` in three ways only:
- the abstract;
- two blank lines;
- the official workshop `.sty`.

**Open item for the human author.** The local file `/Users/Rami/Documents/dev/automlr_autopareto.pdf` is believed to be the submitted PDF, but this is unconfirmed (D22). Before submitting the camera-ready, download the PDF that OpenReview holds and compare it with that file (instructions below). This agent had no network access to OpenReview.

## Candidates examined

| Candidate | Key files (SHA-256, first 16 hex) | Abstract it produces | Reproduces the submitted PDF? |
|---|---|---|---|
| Local submitted PDF `/Users/Rami/Documents/dev/automlr_autopareto.pdf` | `60b419b1e8a22fa7`. 10 pages. CreationDate 2026-09-06T22:59:19Z. | "When an autonomous researcher is evaluated under a fixed wall-clock budget…" | (reference) |
| `paper_backups/automlr_overleaf_reconstructed/` | `main.tex` `3ede53d2de0b6906`; `generated/abstract.tex` `5cda295bfc85ebcc`; `neurips_2026.sty` `c3fc2894e83d2517` (= official template `paper/template/neurips_2026.sty`) | "When an autonomous researcher is evaluated…" | **Yes.** See below. |
| `paper_backups/submitted-automlr-20260907-020829/` | `main.tex` `2dd5d998034d5a4c`; `main.pdf` `88aaef390b318ed4` | "A resource contract can act as a hidden research incentive…" | **No.** The folder name says "submitted", but the abstract differs and the "Affiliation/Address/email" block is missing. |
| `paper/` before the camera-ready edits | `main.tex` `2dd5d998034d5a4c` (identical to the folder above); `generated/abstract.tex` `b518a6aa93893876`; modified `.sty` `c9f6f88e…` | "A resource contract can act…" | **No.** The extracted text differs from the submitted PDF on 696 lines: the abstract, which shifts every following line number. |

## Evidence that the reconstructed source reproduces the submitted PDF

These checks were run in a scratch directory.

1. The reconstructed folder was copied to the scratchpad and compiled with `tectonic main.tex` (Tectonic 0.17.0). Output PDF: `7926d79d7e097788`. The two PDFs differ only in CreationDate.
2. Text was extracted from all 10 pages with `pypdf` 6.17.0 and compared line by line. It is **identical** to the submitted PDF, including the line numbers, the "Affiliation / Address / email" block, and the page breaks.
3. Embedded figure objects are **byte-identical**. Pages 3, 4 and 6 contain one Form XObject each, with these SHA-256 values in both PDFs:
   - `46c92b0da8dcbe52` (lineage)
   - `287aab32e2b85be0` (Pareto)
   - `f8b8816554156f82` (research loop)
4. `generated/*` and `figures/*` are identical across all three source folders (`diff -rq`), except `generated/abstract.tex`.

This agrees with REPRO M1 and `01_PROJECT_INVENTORY.md` §G (V3).

## What was done

- **Preserved the submitted-era sources** in `paper/camera_ready/submitted_source/`, so they can be diffed:
  - `main.tex` (reconstructed source)
  - `abstract.tex` (submitted abstract)
  - `main_paper_dir_before_camera_ready.tex`
  - `abstract_paper_dir_unsubmitted.tex`
  - `primary_result.tex`
- **Replaced `paper/neurips_2026.sty`** with the official template (`c3fc2894…`, provenance in `paper/template/provenance.json`). With the `final` option, the class prints the author block and the workshop footer, and drops line numbers.
- **Inlined the abstract and the Primary Result in `paper/main.tex`.** It no longer reads `generated/abstract.tex` or `generated/primary_result.tex`; the legacy `analysis/generate_reports.py` still rewrites those files, but `main.tex` no longer reads them.
- **Restricted `main.tex` to camera-ready inputs.** It now reads only `generated/camera_ready/*.tex`, which only `analysis/cr_tables.py` writes. It no longer reads `generated/results_macros.tex`, `main_results_table.tex` or `controls_table.tex`.
- **Guarded the legacy pipeline.** `paper/refresh_paper.sh` (exit 2) and `analysis/refresh.py` (SystemExit) refuse to run unless `AUTOPARETO_ALLOW_LEGACY_REFRESH=1` is set. Both refusals were tested.

The camera-ready build is `paper/build_camera_ready.sh`; see `BUILD.md`.

## Remaining step for the human author (D22)

1. Download the PDF that OpenReview holds for the AutoMLR submission.
2. Compute its SHA-256 and compare it with `60b419b1e8a22fa7…`.
3. If the hashes differ, extract its text and diff it against `paper/camera_ready/submitted_source/main.tex` + `abstract.tex`. Use the reconstructed build at `paper_backups/automlr_overleaf_reconstructed/`.
4. Record the outcome here.

The camera-ready content does not depend on which abstract was submitted. The submitted-version text matters only for the changelog's "before" column.

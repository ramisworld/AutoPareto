#!/bin/sh
# Build the AutoPareto camera-ready draft (see camera_ready/BUILD.md).
#
#   AUTOPARETO_PAPER_PYTHON  Python with matplotlib 3.10.9, numpy 2.2.6, pypdf 6.17.0
#                            (default: paper/.venv/bin/python)
#   AUTOPARETO_EVIDENCE_ROOT read-only evidence tree (default: /Users/Rami/Documents/dev/autopareto)
#
# Steps: extract + recompute + cross-check every number (read-only DB access), write macros and
# tables, derive Figure 1 labels, draw figures, compile with Tectonic, check per-part page limits,
# copy the PDF to camera_ready/autopareto_camera_ready_draft.pdf.  Launches no experiment.
set -eu
PAPER_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$PAPER_DIR/.." && pwd)
PY=${AUTOPARETO_PAPER_PYTHON:-$PAPER_DIR/.venv/bin/python}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$PAPER_DIR/camera_ready/.mplcache}
cd "$REPO_DIR/analysis"
"$PY" -B cr_evidence.py          # -> paper/camera_ready/data/cr_evidence.json, cr_stats.json (+ audit cross-check)
"$PY" -B lineage_labels.py >/dev/null   # -> paper/generated/lineage_edit_labels.json (+ camera_ready/data copy)
"$PY" -B cr_tables.py            # -> paper/generated/camera_ready/*.tex
"$PY" -B cr_figures.py           # -> paper/figures/cr_*.{pdf,png}, camera_ready/poster_figures/
"$PY" -B cr_v2.py                # -> E1 + agent-arm macros, tables, figures (cr_v2.json, cr_macros_v2.tex, cr_e1.*, cr_agents.*)
"$PY" -B cr_v3.py                # -> revisions 4-5: Opus milestones and lineage, loop, E1, inference figures; results/funnel/AP-Q/pilot tables; cr_macros_v3.tex
"$PY" -B cr_r8.py                # -> revision 8: results overview, three-agent research tracks, Table 1 (matched tokens + tasks), cr_macros_r8.tex
"$PY" -B cr_p6.py                # -> revision 10: Phase 6 unified 13-recipe dataset and paired statistics (results/camera_ready_2026/phase6/analysis/)   (runs before cr_r9: Figure 4c reads its dataset)
"$PY" -B cr_r9.py                # -> revision 9: candidate comparison, three research histories, downstream tables, inference path, appendix scatter/histories/E1, cr_macros_r9.tex
"$PY" -B cr_p6_paper.py          # -> Figure 1 cr_compare13; Figure 3 cr_unified (revision 11, with stability_check); appendix: table_unified, lm-eval and paired tables; cr_macros_p6.tex
AUTOPARETO_PNG_DPI=300 "$PY" -B generate_figures.py     # -> paper/figures/autopareto_lineage.*, research_loop.* (labels derived)
cd "$PAPER_DIR"
tectonic --keep-logs --keep-intermediates main.tex
"$PY" -B "$REPO_DIR/analysis/cr_page_check.py" --write
cp main.pdf camera_ready/autopareto_camera_ready_draft.pdf
for f in autopareto_lineage research_loop; do
  cp "figures/$f.pdf" "figures/$f.png" camera_ready/poster_figures/
done
echo "Built paper/camera_ready/autopareto_camera_ready_draft.pdf"

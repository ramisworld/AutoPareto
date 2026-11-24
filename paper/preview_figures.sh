#!/bin/sh
set -eu

PAPER_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$PAPER_DIR/.." && pwd)
PYTHON_BIN=${AUTOPARETO_PAPER_PYTHON:-$PAPER_DIR/.venv/bin/python}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Paper Python environment is missing. Run ./paper/refresh_paper.sh once first." >&2
  exit 1
fi

cd "$REPO_DIR"
"$PYTHON_BIN" analysis/generate_figures.py

cd "$PAPER_DIR"
if command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --keep-intermediates main.tex
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode main.tex
else
  echo "Install Tectonic or latexmk to compile the PDF." >&2
  exit 1
fi

echo "Updated figures/*.pdf, figures/*.png, and paper/main.pdf"

#!/bin/sh
set -eu
# CAMERA-READY GUARD (2026-11): this legacy pipeline regenerates generated/abstract.tex,
# generated/primary_result.tex and the legacy tables, and recompiles main.tex.  The camera-ready
# main.tex no longer reads those files, but a refresh would still overwrite the legacy outputs and
# the submitted-version dossiers.  Use paper/build_camera_ready.sh instead (see
# paper/camera_ready/BUILD.md).  Set AUTOPARETO_ALLOW_LEGACY_REFRESH=1 to run it deliberately.
if [ "${AUTOPARETO_ALLOW_LEGACY_REFRESH:-0}" != "1" ]; then
  echo "refresh_paper.sh is disabled for the camera-ready; run paper/build_camera_ready.sh (see paper/camera_ready/BUILD.md)." >&2
  exit 2
fi
PAPER_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${AUTOPARETO_PAPER_PYTHON:-}" ]; then
  PAPER_PYTHON="$AUTOPARETO_PAPER_PYTHON"
elif [ -x "$PAPER_DIR/.venv/bin/python" ]; then
  PAPER_PYTHON="$PAPER_DIR/.venv/bin/python"
else
  python3 -m venv "$PAPER_DIR/.venv"
  PAPER_PYTHON="$PAPER_DIR/.venv/bin/python"
  "$PAPER_PYTHON" -m pip install -r "$PAPER_DIR/requirements.txt"
fi
exec "$PAPER_PYTHON" "$PAPER_DIR/../analysis/refresh.py" "$@"

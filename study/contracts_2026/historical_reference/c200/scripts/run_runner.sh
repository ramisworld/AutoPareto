#!/usr/bin/env bash
# Run the protected runner with the one required SQLite adaptation.
#
# argparse represents --program as pathlib.PosixPath, while SQLite only binds
# built-in scalar types by default.  This adapter changes metadata serialization
# only: it does not alter train.py, the protected evaluator, timing, or results.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${1:-}" == "--metadata-preflight" ]]; then
  shift
  # This check imports no CUDA libraries, so system Python keeps it runnable on
  # both the local Mac and the Linux A40 host.
  exec python3 scripts/check_runner_metadata.py "$@"
fi

PYTHON_COMMAND=(uv run python)
case "${1:-}" in
  init-db|protection-check|state|frontiers|campaign-view|register-campaign|record-action|record-decision|archive-dashboard-campaigns)
    # These commands use only the Python standard library. Keeping them outside the
    # CUDA-only lock makes registration, inspection, and decisions portable to the
    # local Mac without changing any GPU execution path.
    PYTHON_COMMAND=(python3)
    ;;
esac

exec "${PYTHON_COMMAND[@]}" -c '
import pathlib
import runpy
import sqlite3
import sys

sqlite3.register_adapter(pathlib.PosixPath, str)
if hasattr(pathlib, "WindowsPath"):
    sqlite3.register_adapter(pathlib.WindowsPath, str)
sys.argv = ["runner.py", *sys.argv[1:]]
runpy.run_path("runner.py", run_name="__main__")
' "$@"

"""Shared helpers for the camera-ready (2026-11) build.

Read-only by construction:
  * the original evidence tree is only ever opened for reading;
  * SQLite is opened with ``mode=ro&immutable=1``;
  * ``git show`` is the only git command used (it reads objects, never writes).

Outputs go to ``paper/generated/camera_ready/``, ``paper/figures/cr_*``,
``paper/camera_ready/data/`` and ``paper/camera_ready/poster_figures/`` only.
The legacy pipeline (``refresh.py``) never writes to any of these paths.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Evidence root: the original repository (read-only).  Override with
# AUTOPARETO_EVIDENCE_ROOT if the evidence lives elsewhere.
EVIDENCE = Path(os.environ.get('AUTOPARETO_EVIDENCE_ROOT', '/Users/Rami/Documents/dev/autopareto'))
SNAP_PHASE1 = EVIDENCE / 'results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1'
SNAPDB = SNAP_PHASE1 / 'downstream_200m.db'
P1JSON = SNAP_PHASE1 / 'phase1_equal_token_results.json'
TOKEN_STREAM = SNAP_PHASE1 / 'token_stream_verification.json'
CANDIDATES = EVIDENCE / 'results/candidates'
LOGS = EVIDENCE / 'results/logs'
ORIGINAL_TRAIN = ROOT / 'train.py'  # campaign "original/train.py"; validated below by hash replay
AUDIT = ROOT / 'audit/camera_ready_2026'
AUDIT_STATS = AUDIT / 'supporting/statistical_audit.json'
AUDIT_CLAIMS = AUDIT / 'supporting/claims_recomputation.json'
PAPER = ROOT / 'paper'
CR = PAPER / 'camera_ready'
CR_DATA = CR / 'data'
GEN_CR = PAPER / 'generated/camera_ready'
FIG = PAPER / 'figures'
POSTER = CR / 'poster_figures'
SELECTION_REF = 'autopareto/research-infrastructure:results/autopareto_confirmation_selection.json'
PLAN_REF = 'autopareto/research-infrastructure:results/autopareto_primary_confirmation_plan.json'

# One name per object (09 section 0.1).  Old identifiers are kept once in the glossary.
PANEL_ORDER = ['baseline', 'autoresearch_compatible', 'tpe_quality', 'motpe_quality',
               'autopareto_292', 'autopareto_299', 'autopareto_294']
NAMES = {'baseline': 'Baseline', 'autoresearch_compatible': 'AR-ref', 'tpe_quality': 'TPE',
         'motpe_quality': 'MOTPE', 'autopareto_292': 'AP-Q', 'autopareto_299': 'AP-F',
         'autopareto_294': 'AP-S'}
OLD_IDS = {'autopareto_292': 'A22 / exp 134 / AP292', 'autopareto_299': 'A25 / exp 137 / AP299',
           'autopareto_294': 'A24 / exp 136 / AP294', 'autoresearch_compatible': 'exp 68',
           'tpe_quality': 'exp 183', 'motpe_quality': 'exp 234', 'baseline': 'registered baseline'}
DISCOVERY_EXP = {'autopareto_292': 134, 'autopareto_299': 137, 'autopareto_294': 136,
                 'autoresearch_compatible': 68, 'tpe_quality': 183, 'motpe_quality': 234}


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False) + '\n')


def sha256_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def connect_ro():
    con = sqlite3.connect(f'file:{SNAPDB}?mode=ro&immutable=1', uri=True)
    con.row_factory = sqlite3.Row
    return con


def git_show(ref_path):
    """Read one blob from the shared object store.  Read-only."""
    return subprocess.check_output(['git', 'show', ref_path], cwd=ROOT)


def git_commit_of(ref, path):
    out = subprocess.check_output(['git', 'log', '-1', '--format=%H %aI', ref, '--', path], cwd=ROOT, text=True)
    return out.strip()


T975_DF2 = math.sqrt(2 * 0.95 ** 2 / (1 - 0.95 ** 2))  # exact closed form for Student t, df=2


def paired(diffs):
    """Mean, SD and 95% t interval of n paired differences (closed-form t for df=2)."""
    n = len(diffs)
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if n != 3:
        raise ValueError('closed-form critical value implemented for n=3 only')
    half = T975_DF2 * sd / math.sqrt(n)
    return {'n': n, 'diffs': list(diffs), 'mean': mean, 'sd': sd, 'ci95': [mean - half, mean + half],
            't': mean / (sd / math.sqrt(n)), 'all_same_sign': all(d > 0 for d in diffs) or all(d < 0 for d in diffs)}


def fmt_signed(x, nd):
    s = f'{x:+.{nd}f}'
    return s.replace('-', '$-$') if s.startswith('-') else s


# Revision 9: one name per candidate everywhere in the paper (glossary maps the old names).
DISPLAY = {'AP-Q': 'GPT A22', 'AP-F': 'GPT A25', 'AP-S': 'GPT A24'}


def display_names(s):
    """Replace the revision-8 candidate names in generated LaTeX with the revision-9 names."""
    import re as _re
    return _re.sub(r'\bAP-([QFS])\b', lambda m: DISPLAY['AP-' + m.group(1)], s)

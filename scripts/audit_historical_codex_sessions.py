#!/usr/bin/env python3
"""Attribute every historical agent attempt to the Codex session, model and reasoning effort that launched it.

Reads (read-only):
  * the experiment registry  <evidence-root>/results/results.db  (opened with mode=ro&immutable=1);
  * the author's local Codex session logs  ~/.codex/sessions/2026/0[78]/*/*.jsonl.zst  (decompressed with zstdcat);
  * optionally, Cursor's local store  ~/Library/Application Support/Cursor/User/globalStorage/state.vscdb  (read-only):
    conversation settings (composerData) and executed terminal tool calls (bubbleId:*, toolFormerData) that launch a
    seed-42 attempt or record its decision. Seed-42 attempts 1-5 were proposed in Cursor while Codex usage was exhausted.

Writes a metadata-only summary (no prompts, no tool arguments, no outputs; session logs contain access credentials and
stay private):  audit/camera_ready_2026/supporting/historical_codex_attribution.json

Method. For each session, every tool call whose arguments name a historical campaign id is recorded with the session's
model and effort at that moment (its latest `turn_context`). An attempt is attributed to the latest such call within
20 minutes before its `started_at`, preferring calls that launch work (tmux / runner run); attempts with none in 20
minutes are retried with a 90-minute window and reported as "probable". The seed-0 A16->A17 gap is summarised from the
activity timestamps of the session that ran seed 0.

    python3 scripts/audit_historical_codex_sessions.py [--evidence-root DIR] [--sessions DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import glob
import json
from pathlib import Path
import re
import sqlite3
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ['autoresearch-seed42-20260723', 'autoresearch-diagnostic5-seed42-20260724', 'autopareto-seed42-20260727',
             'autopareto-seed0-20260731', 'autopareto-seed1-20260731']
LAUNCH = re.compile(r'tmux new|run_runner\.sh (run|launch|execute|submit|propose)|nohup')


def sessions(root: Path):
    for f in sorted(glob.glob(str(root / '2026/0[78]/*/*.jsonl.zst'))):
        lines = subprocess.run(['zstdcat', f], capture_output=True, text=True, check=True).stdout.splitlines()
        meta, cur, calls, stamps = {}, (None, None), [], []
        for line in lines:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            p, ts = d.get('payload') or {}, (d.get('timestamp') or '')[:19]
            if ts:
                stamps.append(ts)
            if d.get('type') == 'session_meta':
                meta = {k: p.get(k) for k in ('cwd', 'cli_version', 'originator')}
            elif d.get('type') == 'turn_context':
                cur = (p.get('model'), str(p.get('effort')))
            elif p.get('type') in ('function_call', 'custom_tool_call'):
                args = str(p.get('arguments') or p.get('input') or '')
                named = [c for c in CAMPAIGNS if c in args]
                calls.append({'ts': ts, 'model': cur[0], 'effort': cur[1], 'campaigns': named,
                              'launch': bool(LAUNCH.search(args))})
        yield Path(f).name, meta, calls, stamps


def cursor_seed42(path: Path) -> dict:
    """Executed seed-42 run / record-decision tool calls per Cursor conversation, with that conversation's settings."""
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    pat = re.compile(r'run_runner\.sh run --candidate queue/autopareto-seed42-20260727/(\d+)-train\.py'
                     r'|record-decision --experiment-id (\d+)')
    conv = {}
    for k, v in db.execute("select key, value from cursorDiskKV where key like 'composerData:%'"):
        try:
            d = json.loads(v)
        except json.JSONDecodeError:
            continue
        created = d.get('createdAt')
        if not isinstance(created, (int, float)):
            continue
        ts = datetime.utcfromtimestamp(created / 1000).isoformat()[:19]
        if not '2026-07-27' <= ts <= '2026-07-31':
            continue
        sel = (d.get('modelConfig') or {}).get('selectedModels') or [{}]
        conv[k.split(':', 1)[1]] = {'created_utc': ts, 'name': d.get('name'),
                                    'model': sel[0].get('modelId'),
                                    'parameters': {p.get('id'): p.get('value') for p in sel[0].get('parameters', [])},
                                    'executed': []}
    for cid, c in conv.items():
        seen = set()
        for k, v in db.execute("select key, value from cursorDiskKV where key like ?", (f'bubbleId:{cid}%',)):
            try:
                d = json.loads(v)
            except json.JSONDecodeError:
                continue
            tf = d.get('toolFormerData') or {}
            for m in pat.finditer(json.dumps({x: tf.get(x) for x in ('rawArgs', 'params')})):
                what = f'run attempt {m.group(1)}' if m.group(1) else f'record-decision exp {m.group(2)}'
                seen.add((str(d.get('createdAt'))[:19], what))
        c['executed'] = sorted(seen)
    return {k: v for k, v in conv.items() if v['executed']}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-root', type=Path, default=Path('/Users/Rami/Documents/dev/autopareto'))
    ap.add_argument('--sessions', type=Path, default=Path.home() / '.codex/sessions')
    ap.add_argument('--cursor-db', type=Path,
                    default=Path.home() / 'Library/Application Support/Cursor/User/globalStorage/state.vscdb')
    ap.add_argument('--out', type=Path, default=ROOT / 'audit/camera_ready_2026/supporting/historical_codex_attribution.json')
    a = ap.parse_args()
    db = sqlite3.connect(f"file:{a.evidence_root / 'results/results.db'}?mode=ro&immutable=1", uri=True)
    sess = list(sessions(a.sessions))
    calls = sorted((dict(c, session=name) for name, _, cs, _ in sess for c in cs), key=lambda c: c['ts'])
    meta = {name: m for name, m, _, _ in sess}
    out = {'format': 'autopareto_historical_codex_attribution_v1', 'sessions_scanned': len(sess),
           'session_metadata': {n: m for n, m in meta.items() if m.get('cwd', '').endswith('/autopareto')},
           'campaigns': {}}
    for cid in CAMPAIGNS:
        rows = list(db.execute('select id, started_at from experiments where campaign_id=? order by id', (cid,)))
        per = []
        for eid, st in rows:
            st = st.replace(' ', 'T')
            pick, quality = None, 'attributed'
            for minutes in (20, 90):
                t0 = (datetime.fromisoformat(st) - timedelta(minutes=minutes)).isoformat()
                prior = [c for c in calls if cid in c['campaigns'] and t0 <= c['ts'] <= st]
                launch = [c for c in prior if c['launch']] or prior
                if launch:
                    pick = launch[-1]
                    break
                quality = 'probable'
            per.append({'experiment_id': eid, 'started_at': st, 'attribution': quality if pick else 'none',
                        'session': pick and pick['session'], 'model': pick and pick['model'],
                        'effort': pick and pick['effort']})
        summary = {}
        for r in per:
            k = f"{r['model']}/{r['effort']} ({r['attribution']})"
            summary.setdefault(k, []).append(r['experiment_id'])
        out['campaigns'][cid] = {'attempts': len(per), 'summary': summary, 'per_attempt': per}
    # seed-0 A16 -> A17 gap: activity of the session that launched seed 0
    s0 = out['campaigns']['autopareto-seed0-20260731']['per_attempt']
    a16, a17 = next(r for r in s0 if r['experiment_id'] == 128), next(r for r in s0 if r['experiment_id'] == 129)
    fin16 = db.execute('select finished_at from experiments where id=128').fetchone()[0].replace(' ', 'T')
    stamps = next(st for name, _, _, st in sess if name == a17['session'])
    window = sorted(t for t in stamps if fin16 <= t <= a17['started_at'])
    gaps = sorted(((datetime.fromisoformat(y) - datetime.fromisoformat(x)).total_seconds() / 3600, x, y)
                  for x, y in zip(window, window[1:]))[-3:]
    out['seed0_gap'] = {'a16_finished_at': fin16, 'a17_started_at': a17['started_at'],
                        'session': a17['session'], 'events_in_gap': len(window),
                        'largest_silences_hours': [{'hours': round(h, 2), 'from': x, 'to': y} for h, x, y in gaps[::-1]]}
    out['cursor_seed42'] = cursor_seed42(a.cursor_db) if a.cursor_db.exists() else 'cursor store not found'
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    for cid, c in out['campaigns'].items():
        print(cid, c['summary'])
    print('seed-0 gap:', out['seed0_gap']['largest_silences_hours'])
    if isinstance(out['cursor_seed42'], dict):
        for cid, c in out['cursor_seed42'].items():
            print('cursor', c['created_utc'], c['name'], c['model'], c['parameters'], len(c['executed']), 'executed calls')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

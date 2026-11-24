#!/usr/bin/env python3
"""Derive the Figure 1 edit labels from the frozen recipe sources (fixes D9 / REPRO M2).

The submitted Figure 1 used a hand-typed label dictionary that ignored the width
rounding in each source's ``build_model_config``:

    model_dim = ((depth * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM

This module never imports or executes a candidate program.  It
  1. reads each seed-0 attempt's frozen ``train.py`` (read-only);
  2. for A17, whose frozen file was never written, rebuilds the source by applying the
     recorded ``experiments.code_diff`` to the campaign's original ``train.py``.  The same
     replay is validated against every attempt whose frozen file exists (byte-identical
     SHA-256 required);
  3. evaluates literal module constants and the body of ``build_model_config`` with a small
     AST interpreter (arithmetic, ``max``/``min``/``round`` and simple generator filters only);
  4. cross-checks depth, width, heads and KV heads against the ``Model config`` line of
     each run log;
  5. labels each attempt by the change relative to its recorded parent.

Output: ``paper/generated/lineage_edit_labels.json`` (cached; used when the evidence tree is
not available) and the same file under ``paper/camera_ready/data/``.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_common import (CANDIDATES, CR_DATA, LOGS, ORIGINAL_TRAIN, PAPER, connect_ro, read_json,
                       sha256_bytes, sha256_file, write_json)

CAMPAIGN = 'autopareto-seed0-20260731'
BASELINE_ID = 8
BASELINE_SOURCE = CANDIDATES / 'baseline-a40-cached-fresh-20260722/0001-train.py'
CACHE = PAPER / 'generated/lineage_edit_labels.json'

# ---------------------------------------------------------------- unified diff replay

def apply_unified_diff(original: str, diff: str) -> str:
    src = original.split('\n')
    out, pos = [], 0
    lines = diff.split('\n')
    i = 0
    while i < len(lines):
        m = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', lines[i])
        if not m:
            i += 1
            continue
        start = int(m.group(1)) - 1
        out.extend(src[pos:start])
        pos = start
        i += 1
        while i < len(lines) and not lines[i].startswith('@@'):
            ln = lines[i]
            if ln.startswith('\\'):
                i += 1
                continue
            if ln == '' and i == len(lines) - 1:
                break
            tag, body = (ln[:1], ln[1:]) if ln else (' ', '')
            if tag == ' ':
                if src[pos] != body:
                    raise ValueError(f'context mismatch at line {pos + 1}')
                out.append(body)
                pos += 1
            elif tag == '-':
                if src[pos] != body:
                    raise ValueError(f'removal mismatch at line {pos + 1}')
                pos += 1
            elif tag == '+':
                out.append(body)
            else:
                raise ValueError(f'unexpected diff line {ln[:40]!r}')
            i += 1
    out.extend(src[pos:])
    return '\n'.join(out)


# ---------------------------------------------------------------- safe evaluation

_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b if abs(b) < 64 else (_ for _ in ()).throw(ValueError('pow'))}
_CMP = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}
_CALLS = {'max': max, 'min': min, 'round': round, 'int': int, 'range': range, 'abs': abs}


def _ev(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise KeyError(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ev(node.operand, env)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_ev(node.left, env), _ev(node.right, env))
    if isinstance(node, ast.Tuple):
        return tuple(_ev(x, env) for x in node.elts)
    if isinstance(node, ast.Compare):
        left = _ev(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _ev(comp, env)
            if not _CMP[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        vals = (_ev(v, env) for v in node.values)
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALLS and not node.keywords:
        if len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            return _CALLS[node.func.id](_gen(node.args[0], env))
        return _CALLS[node.func.id](*[_ev(a, env) for a in node.args])
    raise ValueError(f'unsupported node {ast.dump(node)[:80]}')


def _gen(g, env):
    if len(g.generators) != 1:
        raise ValueError('nested generator')
    comp = g.generators[0]
    if not isinstance(comp.target, ast.Name):
        raise ValueError('generator target')
    out = []
    for v in _ev(comp.iter, env):
        local = dict(env)
        local[comp.target.id] = v
        if all(_ev(c, local) for c in comp.ifs):
            out.append(_ev(g.elt, local))
    return out


def module_constants(tree):
    env = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                env[n.targets[0].id] = _ev(n.value, env)
            except (KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError):
                pass
    return {k: v for k, v in env.items() if k.isupper()}


def effective_config(source: str):
    """Evaluate build_model_config(DEPTH) from the source's own function body."""
    tree = ast.parse(source)
    const = module_constants(tree)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_model_config')
    env = dict(const)
    env[fn.args.args[0].arg] = const['DEPTH']
    env.update({'MAX_SEQ_LEN': const.get('MAX_SEQ_LEN', 2048), 'vocab_size': 8192})
    for st in fn.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            env[st.targets[0].id] = _ev(st.value, env)
        elif isinstance(st, ast.Return):
            call = st.value
            kw = {k.arg: _ev(k.value, env) for k in call.keywords}
            return const, {'n_layer': kw['n_layer'], 'n_embd': kw['n_embd'], 'n_head': kw['n_head'],
                           'n_kv_head': kw['n_kv_head'], 'window_pattern': kw['window_pattern']}
        elif isinstance(st, ast.Expr):
            continue
        else:
            raise ValueError('unsupported statement in build_model_config')
    raise ValueError('no return')


def log_config(attempt: int):
    p = LOGS / f'{CAMPAIGN}-{attempt:04d}.log'
    if not p.exists():
        return None
    for line in p.read_text(errors='replace').splitlines():
        if line.startswith('Model config:'):
            d = ast.literal_eval(line.split(':', 1)[1].strip())
            return {k: d[k] for k in ('n_layer', 'n_embd', 'n_head', 'n_kv_head', 'window_pattern')}
    return None


# ---------------------------------------------------------------- labels

LR_NAMES = {'MATRIX_LR': 'matrix LR', 'EMBEDDING_LR': 'embed LR', 'UNEMBEDDING_LR': 'head LR',
            'SCALAR_LR': 'scalar LR', 'WEIGHT_DECAY': 'weight decay'}
IGNORED = {'DEPTH', 'ASPECT_RATIO', 'HEAD_DIM', 'DEVICE_BATCH_SIZE'}


def _num(x):
    if isinstance(x, float):
        s = f'{x:g}'
        return s[1:] if s.startswith('0.') else s
    return str(x)


def _pow2(a, b):
    def e(x):
        return f'2^{{{x.bit_length() - 1}}}' if isinstance(x, int) and x > 0 and x & (x - 1) == 0 else str(x)
    return f'${e(a)}\\to{e(b)}$'


def edit_label(pc, pcfg, cc, ccfg):
    parts = []
    width_changed = pcfg['n_embd'] != ccfg['n_embd']
    if pc.get('DEPTH') != cc.get('DEPTH'):
        parts.append(f"depth {pc['DEPTH']}$\\to${cc['DEPTH']}")
    if pc.get('ASPECT_RATIO') != cc.get('ASPECT_RATIO'):
        parts.append(f"aspect {pc['ASPECT_RATIO']}$\\to${cc['ASPECT_RATIO']}")
    if width_changed:
        parts.append(f"width {pcfg['n_embd']}$\\to${ccfg['n_embd']}")
    elif pc.get('DEPTH') != cc.get('DEPTH') or pc.get('ASPECT_RATIO') != cc.get('ASPECT_RATIO'):
        parts.append(f"width stays {ccfg['n_embd']}")
    # KV heads are listed only when the query/KV grouping changes; when n_kv_head simply
    # follows n_head (because width changed) the width entry already describes the edit.
    if pcfg['n_kv_head'] * ccfg['n_head'] != ccfg['n_kv_head'] * pcfg['n_head']:
        parts.append(f"KV heads {pcfg['n_kv_head']}$\\to${ccfg['n_kv_head']}")
    if pcfg['window_pattern'] != ccfg['window_pattern']:
        parts.append(f"window {pcfg['window_pattern']}$\\to${ccfg['window_pattern']}")
    if pc.get('TOTAL_BATCH_SIZE') != cc.get('TOTAL_BATCH_SIZE'):
        parts.append(f"batch {_pow2(pc.get('TOTAL_BATCH_SIZE'), cc.get('TOTAL_BATCH_SIZE'))}")
    for k in sorted(set(pc) | set(cc)):
        if k in IGNORED or k == 'TOTAL_BATCH_SIZE' or pc.get(k) == cc.get(k):
            continue
        if k == 'KV_HEAD_RATIO':
            continue  # represented through the effective KV-head count
        name = LR_NAMES.get(k, k)
        parts.append(f"{name} {_num(pc.get(k))}$\\to${_num(cc.get(k))}")
    effective = (pcfg != ccfg) or any(pc.get(k) != cc.get(k) for k in set(pc) | set(cc)
                                       if k not in {'ASPECT_RATIO', 'DEVICE_BATCH_SIZE'})
    if not parts:
        return 'code-only edit (see recorded patch)', effective
    # Collapse the two-part aspect/width wording when nothing effective changed.
    if not effective and pc.get('ASPECT_RATIO') != cc.get('ASPECT_RATIO'):
        return f"aspect {pc['ASPECT_RATIO']}$\\to${cc['ASPECT_RATIO']} (no effective change)", False
    label = '; '.join(p for p in parts if not p.startswith('width stays'))
    stays = [p for p in parts if p.startswith('width stays')]
    if stays and pc.get('ASPECT_RATIO') == cc.get('ASPECT_RATIO'):
        label += f" ({stays[0]})"
    return label, effective


def derive():
    con = connect_ro()
    rows = [dict(r) for r in con.execute(
        "select id, experiment_number, parent_experiment_id, status, code_diff from experiments "
        "where campaign_id=? order by experiment_number", (CAMPAIGN,))]
    original = ORIGINAL_TRAIN.read_text()
    sources, provenance, replay = {}, {}, []
    sources[BASELINE_ID] = BASELINE_SOURCE.read_text()
    provenance[BASELINE_ID] = {'kind': 'frozen', 'path': str(BASELINE_SOURCE), 'sha256': sha256_file(BASELINE_SOURCE)}
    for r in rows:
        frozen = CANDIDATES / CAMPAIGN / f"{r['experiment_number']:04d}-train.py"
        rebuilt = apply_unified_diff(original, r['code_diff']) if r['code_diff'] else None
        if frozen.exists():
            text = frozen.read_text()
            ok = rebuilt is not None and sha256_bytes(rebuilt.encode()) == sha256_file(frozen)
            replay.append({'attempt': r['experiment_number'], 'exp': r['id'], 'replay_matches_frozen': ok})
            sources[r['id']] = text
            provenance[r['id']] = {'kind': 'frozen', 'path': str(frozen), 'sha256': sha256_file(frozen)}
        else:
            sources[r['id']] = rebuilt
            provenance[r['id']] = {'kind': 'replayed_code_diff', 'path': None,
                                   'sha256': sha256_bytes(rebuilt.encode()),
                                   'note': 'frozen file not preserved; rebuilt from experiments.code_diff '
                                           f'applied to {ORIGINAL_TRAIN.name} (sha256 {sha256_file(ORIGINAL_TRAIN)[:16]})'}
    if not all(x['replay_matches_frozen'] for x in replay):
        raise SystemExit('code_diff replay does not reproduce every frozen source; refusing to label')
    cfg = {}
    for i, s in sources.items():
        cfg[i] = effective_config(s)
    out = []
    for r in rows:
        pid = r['parent_experiment_id']
        pc, pcfg = cfg[pid]
        cc, ccfg = cfg[r['id']]
        label, effective = edit_label(pc, pcfg, cc, ccfg)
        logged = log_config(r['experiment_number'])
        out.append({'attempt': r['experiment_number'], 'exp': r['id'], 'parent_exp': pid, 'status': r['status'],
                    'label': label, 'effective_change': effective, 'config': ccfg,
                    'log_config': logged, 'log_matches_source': (logged == ccfg) if logged else None,
                    'source': provenance[r['id']]})
    bad = [x for x in out if x['log_matches_source'] is False]
    if bad:
        raise SystemExit(f'source/log config mismatch: {[x["attempt"] for x in bad]}')
    result = {'campaign': CAMPAIGN, 'baseline_exp': BASELINE_ID,
              'baseline_config': cfg[BASELINE_ID][1], 'baseline_source': provenance[BASELINE_ID],
              'rule': 'width = ceil(DEPTH*ASPECT_RATIO/HEAD_DIM)*HEAD_DIM, evaluated from each source\'s own build_model_config',
              'code_diff_replay_check': replay, 'labels': out}
    write_json(CACHE, result)
    write_json(CR_DATA / 'lineage_edit_labels.json', result)
    return result


def load_labels():
    """Return {attempt_number: label}.  Derive from sources when available, else use the cache."""
    try:
        res = derive()
    except (FileNotFoundError, OSError) as e:  # evidence tree unavailable
        print(f'lineage_labels: evidence unavailable ({e}); using cached {CACHE}', file=sys.stderr)
        res = read_json(CACHE)
    return {x['attempt']: x['label'] for x in res['labels']}


if __name__ == '__main__':
    r = derive()
    for x in r['labels']:
        print(f"A{x['attempt']:02d} exp{x['exp']} <- exp{x['parent_exp']}: {x['label']}  "
              f"[width {x['config']['n_embd']}, log ok={x['log_matches_source']}, {x['source']['kind']}]")
    print('replay checks:', sum(x['replay_matches_frozen'] for x in r['code_diff_replay_check']), 'of',
          len(r['code_diff_replay_check']))

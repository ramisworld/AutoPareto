#!/usr/bin/env python3
"""Map an Opus 5.5 Contract-T (Pair-A) candidate to the Phase-1 fixed-token recipe format.

Phase 5A (2026-11-08) of the camera-ready study. The frozen Pair-A adapter
(``scripts/pair_a_to_phase1_source.py``, unchanged) refuses the three Opus
frontier sources (attempts 23, 24, 27). It looks for the panel baseline's
training-loop anchors, but Opus rewrote the loop *driver* for Contract T:
- a ``contract_progress`` call whose return value is used;
- a perf_counter clock with no warm-up exclusion;
- a fallback time stop;
- a NaN ``val_bpb`` placeholder.
This module restores exactly those driver lines to the panel baseline's text,
which the unchanged protected ``phase1_source_transform`` then rewrites to
token-fraction progress and an exact-token stop, as it does for AR-ref and AP292.

Nothing scientific is edited:
- model code, initialisation, optimizer, hyperparameters, batch sizes, LR /
  momentum / weight-decay schedule functions and the loop's training statements
  are byte-identical;
- the recipe's own post-training storage conversion lines
  (``set_matrix_dtype(bfloat16)``; for attempt 27 also ``set_table_dtype(int8)``)
  are moved verbatim from after the summary prints to before the restored final
  evaluation. In Pair-A, too, the scored model is the converted one, because
  the protected evaluator scores the checkpoint saved after the script ends.

Every replacement must match exactly once, or the source is refused. The
gates are listed in ``self_check``.

Usage:
    python3 scripts/opus_contract_t_to_phase1_source.py --source <opus train.py> --output <file>
    python3 scripts/opus_contract_t_to_phase1_source.py --self-check --sources <dir with opus_attempt*.train.py>
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pair_a_to_phase1_source as frozen_adapter  # noqa: E402  (unchanged frozen adapter; only its gate helper is used)

SETUP_MARKER = "# Setup: tokenizer, model, optimizer, dataloader"
STORAGE_START = "# Inference storage precision: after the last completed update and outside the\n"
STORAGE_END = "# Protected launch-interface exports (val_bpb is diagnostic only).\n"

PREPARE_IMPORT = (
    "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader\n",
    "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb\n",
)

# (label, Contract-T driver text, panel-baseline driver text). Each old text must occur exactly once.
DRIVER_RESTORATIONS = (
    (
        "contract_t_constants",
        'from study_runtime import contract_progress\n\nCONTRACT_ID = "T"\n'
        "CONTRACT_TIME_BUDGET_S = 300.0  # protected Contract-T measured training budget\n"
        "# Margin on the candidate-side fallback stop. The candidate clock can run\n"
        "# slightly ahead of the parent's observer clock, so a fallback break exactly at\n"
        "# 300 s can return before the protected boundary stop (attempt-02's\n"
        "# CandidateExitedBeforeStop). With this margin the protected stop always ends\n"
        "# training under protected execution; the fallback only matters standalone.\n"
        "STOP_GUARD_S = 10.0\n\n",
        "",
    ),
    (
        "budget_print",
        'print(f"Time budget: {CONTRACT_TIME_BUDGET_S}s (Contract {CONTRACT_ID}; prepare.TIME_BUDGET={TIME_BUDGET})")\n',
        'print(f"Time budget: {TIME_BUDGET}s")\n',
    ),
    (
        "loop_clock_start",
        "t_start_training = time.time()\nsmooth_train_loss = 0\nstep = 0\n\n"
        "# Contract-T measured training clock: synchronized perf_counter started\n"
        "# immediately before the first scientific forward. Every step counts (no\n"
        "# warm-up exclusion); the trusted parent's clock remains authoritative.\n"
        "torch.cuda.synchronize()\nt_train_start = time.perf_counter()\n"
        "measured_training_time = 0.0  # synchronized elapsed time at the last completed update\n"
        "total_training_time = 0.0\n\nwhile True:\n    t0 = time.perf_counter()\n",
        "t_start_training = time.time()\nsmooth_train_loss = 0\ntotal_training_time = 0\nstep = 0\n\n"
        "while True:\n    torch.cuda.synchronize()\n    t0 = time.time()\n",
    ),
    (
        "progress",
        "    # Protected contract boundary: declare the next optimizer update and use\n"
        "    # the returned progress for all progress-based schedules.\n"
        "    tokens_so_far = (step + 1) * TOTAL_BATCH_SIZE\n"
        '    progress = contract_progress("T", measured_training_time, tokens_so_far)\n',
        "    # Progress and schedules\n    progress = min(total_training_time / TIME_BUDGET, 1.0)\n",
    ),
    (
        "step_clock_end",
        "    t1 = time.perf_counter()\n    dt = t1 - t0\n"
        "    measured_training_time = t1 - t_train_start\n    total_training_time = measured_training_time\n",
        "    t1 = time.time()\n    dt = t1 - t0\n\n    if step > 10:\n        total_training_time += dt\n",
    ),
    (
        "remaining",
        "    remaining = max(0, CONTRACT_TIME_BUDGET_S - measured_training_time)\n",
        "    remaining = max(0, TIME_BUDGET - total_training_time)\n",
    ),
    (
        "stop",
        "    # Fallback candidate-side stop after a completed update, unreachable under\n"
        "    # protected execution (protected stop fires at 300 s on the parent clock).\n"
        "    # The protected boundary stop is intentionally never caught here.\n"
        "    if measured_training_time >= CONTRACT_TIME_BUDGET_S + STOP_GUARD_S:\n",
        "    # Time's up \u2014 but only stop after warmup steps so we don't count compilation\n"
        "    if step > 10 and total_training_time >= TIME_BUDGET:\n",
    ),
    (
        "val_bpb_print",
        'print("---")\nprint(f"training_seconds:',
        'print("---")\nprint(f"val_bpb:          {val_bpb:.6f}")\nprint(f"training_seconds:',
    ),
)

POST_LOOP_OLD = (
    "total_tokens = step * TOTAL_BATCH_SIZE\noptimizer_updates = step\nmicrobatches = step * grad_accum_steps\n"
    "val_bpb = float('nan')  # diagnostic only; trusted parent owns validation\n\n# Final summary\n"
)
POST_LOOP_HEAD = "total_tokens = step * TOTAL_BATCH_SIZE\noptimizer_updates = step\nmicrobatches = step * grad_accum_steps\n\n"
FINAL_EVAL = (
    "# Final eval\nmodel.eval()\nwith autocast_ctx:\n    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)\n\n# Final summary\n"
)
FORBIDDEN_NAMES = {
    "contract_progress", "study_runtime", "CONTRACT_ID", "CONTRACT_TIME_BUDGET_S", "STOP_GUARD_S",
    "measured_training_time", "t_train_start", "tokens_so_far",
}
ALLOWED_STORAGE_CALLS = {"set_matrix_dtype", "set_table_dtype"}


class MappingRefusal(ValueError):
    """The source is not an Opus Contract-T recipe this mapping may restore."""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise MappingRefusal(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def storage_block(source: str) -> str:
    """The recipe's own post-training storage conversion block (comments + set_*_dtype calls only)."""
    if source.count(STORAGE_START) != 1 or source.count(STORAGE_END) != 1:
        raise MappingRefusal("post-training storage block anchors not found exactly once")
    start, end = source.index(STORAGE_START), source.index(STORAGE_END)
    if end <= start:
        raise MappingRefusal("storage block precedes its start anchor")
    block = source[start:end]
    statements = ast.parse(block).body
    if not statements:
        raise MappingRefusal("empty storage block")
    for node in statements:
        call = node.value if isinstance(node, ast.Expr) else None
        ok = (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in ALLOWED_STORAGE_CALLS
            and ast.unparse(call.func.value) == "getattr(model, '_orig_mod', model)"
        )
        if not ok:
            raise MappingRefusal(f"storage block holds a non-storage statement: {ast.unparse(node)!r}")
    return block


def storage_statements(source: str) -> list[str]:
    return [ast.unparse(node) for node in ast.parse(storage_block(source)).body]


def adapt(source: str) -> str:
    block = storage_block(source)
    mapped = source.replace(block, "", 1)
    mapped = _replace_once(mapped, *PREPARE_IMPORT, "prepare_import")
    for label, old, new in DRIVER_RESTORATIONS:
        mapped = _replace_once(mapped, old, new, label)
    mapped = _replace_once(mapped, POST_LOOP_OLD, POST_LOOP_HEAD + block + FINAL_EVAL, "post_loop")
    tree = ast.parse(mapped)
    for node in ast.walk(tree):
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if name in FORBIDDEN_NAMES:
            raise MappingRefusal(f"source still uses {name} after the driver restoration")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            if "study_runtime" in modules:
                raise MappingRefusal("study_runtime import remains")
    return mapped


def definitions_identical(original: str, mapped: str) -> bool:
    """Everything before the setup marker is byte-identical except the prepare import line."""
    before_o = original.split(SETUP_MARKER, 1)[0].replace(*PREPARE_IMPORT, 1)
    before_m = mapped.split(SETUP_MARKER, 1)[0]
    return before_o == before_m


def training_statements(source: str) -> list[str]:
    """Unparsed scientific statements of the setup/training section (driver statements excluded)."""
    section = source.split(SETUP_MARKER, 1)[1]
    tree = ast.parse("\n".join(section.splitlines()[1:]))
    driver_names = FORBIDDEN_NAMES | {"total_training_time", "t0", "t1", "dt", "remaining", "progress", "val_bpb"}
    keep: list[str] = []

    def visit(nodes, depth=0):
        for node in nodes:
            if isinstance(node, ast.While):
                keep.append("while True:")
                visit(node.body, depth + 1)
                continue
            text = ast.unparse(node)
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if names & driver_names and not isinstance(node, (ast.FunctionDef, ast.For)):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "print":
                continue
            if isinstance(node, ast.If) and "step > 10" in text:
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "study_runtime":
                continue
            keep.append("  " * depth + text)

    visit(tree.body)
    return keep


def statement_changes(original: str, mapped: str) -> dict:
    """Removed and added scientific statements (with loop depth), as sorted lists."""
    removed, added = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=training_statements(original), b=training_statements(mapped), autojunk=False
    ).get_opcodes():
        if tag in ("delete", "replace"):
            removed.extend(training_statements(original)[i1:i2])
        if tag in ("insert", "replace"):
            added.extend(training_statements(mapped)[j1:j2])
    return {"removed": sorted(removed), "added": sorted(added)}


def expected_statement_changes(original: str) -> dict:
    """The only permitted statement-level changes:
    - the pre-loop synchronize becomes the loop-top synchronize (panel-baseline step timing);
    - the recipe's storage conversion statements move before the final evaluation;
    - the restored final evaluation adds model.eval().
    """
    storage = storage_statements(original)
    return {
        "removed": sorted(["torch.cuda.synchronize()"] + storage),
        "added": sorted(["  torch.cuda.synchronize()", "model.eval()"] + storage),
    }


def unified_diff(original: str, mapped: str, name: str) -> str:
    return "".join(difflib.unified_diff(
        original.splitlines(keepends=True), mapped.splitlines(keepends=True),
        fromfile=f"{name} (Pair-A Contract-T source)", tofile=f"{name} (Phase-1 recipe format)", n=1,
    ))


def self_check(sources: pathlib.Path) -> dict:
    report: dict = {"frozen_adapter_gates": {}, "sources": {}}
    report["frozen_adapter_gates"] = {
        key: value for key, value in frozen_adapter.self_check(None).items() if key.startswith("gate_")
    }
    hunks_by_source = {}
    for path in sorted(sources.glob("opus_attempt*.train.py")):
        original = path.read_text()
        row: dict = {"source_sha256": hashlib.sha256(original.encode()).hexdigest()}
        try:
            frozen_adapter.adapt(original)
            row["frozen_adapter"] = "accepted"
        except (frozen_adapter.AdapterRefusal, SyntaxError) as exc:
            row["frozen_adapter"] = f"refused: {exc}"
        try:
            mapped = adapt(original)
        except MappingRefusal as exc:
            row["mapping"] = f"refused: {exc}"
            report["sources"][path.name] = row
            continue
        row["mapped_sha256"] = hashlib.sha256(mapped.encode()).hexdigest()
        row["gate_definitions_byte_identical_except_prepare_import"] = definitions_identical(original, mapped)
        row["training_statement_changes"] = statement_changes(original, mapped)
        row["gate_training_statements_change_only_as_expected"] = (
            row["training_statement_changes"] == expected_statement_changes(original)
        )
        row["storage_statements_moved_verbatim"] = storage_statements(original)
        row["gate_storage_block_moved_verbatim"] = storage_block(original) in mapped
        transform = {}
        for seed in (6, 7, 8):
            reason = _transform_accepts(mapped, 201_326_592, seed)
            transform[str(seed)] = reason or "accepted"
        row["protected_transform_201326592"] = transform
        row["gate_protected_transform_accepts_all_seeds"] = all(v == "accepted" for v in transform.values())
        diff = unified_diff(original, mapped, path.name)
        row["changed_lines"] = sum(
            1 for line in diff.splitlines() if line[:1] in "+-" and not line.startswith(("+++", "---"))
        )
        hunks_by_source[path.name] = [
            line for line in diff.splitlines()
            if line[:1] in "+-" and not line.startswith(("+++", "---"))
        ]
        report["sources"][path.name] = row
    # The driver restoration is the same edit for every source; only the moved storage block may differ.
    def driver_only(lines):
        return [line for line in lines if "set_table_dtype" not in line and "Gather-only tables" not in line
                and "int8 storage + per-row" not in line and "the dtype a fresh model host declares; same Parameter" not in line]
    variants = {json.dumps(driver_only(lines)) for lines in hunks_by_source.values()}
    report["gate_same_driver_edit_for_every_source"] = len(variants) == 1
    gates = [v for k, v in report["frozen_adapter_gates"].items()]
    gates.append(report["gate_same_driver_edit_for_every_source"])
    for row in report["sources"].values():
        gates.extend(v for k, v in row.items() if k.startswith("gate_"))
        if "mapping" in row:
            gates.append(False)
    report["all_gates_pass"] = all(gates)
    return report


def _transform_accepts(source: str, token_budget: int, seed: int) -> str | None:
    """None if the unchanged protected Phase-1 transform accepts the source at this budget and seed."""
    import tempfile
    sys.path.insert(0, str(ROOT))
    from protected import phase1_source_transform

    work = pathlib.Path(tempfile.mkdtemp(prefix="opus-mapping-check-"))
    try:
        derived = phase1_source_transform.transform_source(
            source,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            token_budget=token_budget,
            seed=seed,
            milestone_dir=work / "milestones",
            final_checkpoint=work / "final.pt",
            step_timing_path=work / "step_timing.jsonl",
            timing_summary_path=work / "timing_summary.json",
            telemetry_path=work / "telemetry.json",
            telemetry_interval_seconds=2.0,
        )
    except ValueError as exc:
        return str(exc)
    if f"PHASE1_TARGET_STEPS = {token_budget // 131072}\n" not in derived:
        return "derived source lacks the expected step target"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--diff-output", type=pathlib.Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--sources", type=pathlib.Path)
    args = parser.parse_args()
    if args.self_check:
        if args.sources is None:
            parser.error("--self-check needs --sources")
        report = self_check(args.sources)
        print(json.dumps(report, indent=2))
        return 0 if report["all_gates_pass"] else 1
    if args.source is None or args.output is None:
        parser.error("--source and --output are required unless --self-check is given")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    original = args.source.read_text()
    mapped = adapt(original)
    for seed in (6, 7, 8):
        reason = _transform_accepts(mapped, 201_326_592, seed)
        if reason is not None:
            raise SystemExit(f"protected Phase-1 transform refuses the mapped source (seed {seed}): {reason}")
    args.output.write_text(mapped)
    if args.diff_output is not None:
        if args.diff_output.exists():
            raise SystemExit(f"refusing to overwrite {args.diff_output}")
        args.diff_output.write_text(unified_diff(original, mapped, args.source.name))
    print(json.dumps({
        "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "mapped_sha256": hashlib.sha256(mapped.encode()).hexdigest(),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Camera-ready E0/E1 executor.  Validation is the default; GPU work is explicit.

    python scripts/run_camera_ready_e0_e1.py                      # == --validate-only
    python scripts/run_camera_ready_e0_e1.py --stage E0 --execute  # on the A40 pod
    python scripts/run_camera_ready_e0_e1.py --stage E1 --execute  # only after E0 PASS

The pre-registration is ``evaluation/camera_ready_e0_e1_plan_amendment1.json``
(amendment 1 of ``evaluation/camera_ready_e0_e1_plan.json``).  This file is
deliberately thin:

* Fixed-token cells call the Phase-1 ``run_one`` / ``evaluate_one`` functions
  from ``scripts/run_phase1_equal_tokens.py`` unchanged.  Only the module-level
  ``PHASE1_ROOT`` is pointed at ``<plan output_root>/<stage>`` for the
  duration of one call, and a per-run plan view supplies the budget and seed.
  The launcher (``protected.phase1_train_launcher``), the source transform, the
  runtime recorders and the checkpoint evaluator are therefore byte-for-byte the
  Phase-1 code paths.
* The 300 s fixed-time cell runs the historical protected fixed-time launcher
  (``protected/train_launcher.py`` from commit c200ccb, sha256 in the plan) the way
  the historical ``runner.py`` ran it: from an isolate directory that holds the
  frozen candidate as ``train.py`` plus the protected modules it imports, with
  ``PYTORCH_CUDA_ALLOC_CONF`` set to the historical allocator.  The final
  checkpoint is then scored by the same Phase-1 checkpoint evaluator
  (``scripts.evaluate_phase1_checkpoints --heldout``).  Nothing under
  ``protected/`` is edited.

Nothing is ever written to ``results/phase1``; run directories are never reused.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Amendment 1 (2026-11-04) is the active plan. The original plan
# evaluation/camera_ready_e0_e1_plan.json (sha256 806a6397...) is kept unchanged;
# its E0 halted, and its results stay under results/camera_ready_2026/E0.
PLAN_PATH = ROOT / "evaluation/camera_ready_e0_e1_plan_amendment1.json"
PLAN_FORMAT = "autopareto_camera_ready_e0_e1_preregistration_v1"
RECORD_FORMAT = "autopareto_camera_ready_run_record_v1"
RECORD_NAME = "camera_ready_run_record.json"
PHASE1_RESULTS = ROOT / "results/phase1"

OK, FAIL, POD_ONLY, INFO = "ok", "FAIL", "POD_ONLY", "info"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(name: str, path: Path) -> types.ModuleType:
    """Import a repository script by path (no reliance on a ``scripts`` package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase1 = load_module("camera_ready_phase1_executor", ROOT / "scripts/run_phase1_equal_tokens.py")
from protected.phase1_source_transform import read_total_batch_size, transform_source  # noqa: E402


def write_json_new(path: Path, payload: dict) -> None:
    """Write a JSON artifact that must not already exist (evidence is never overwritten)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def module_constants(path: Path, names: set[str]) -> dict:
    """Read literal module-level constants without importing (no torch needed)."""
    found: dict = {}
    for statement in ast.parse(path.read_text()).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    try:
                        found[target.id] = ast.literal_eval(statement.value)
                    except ValueError:
                        try:
                            found[target.id] = phase1_integer(statement.value)
                        except ValueError:
                            found[target.id] = ast.unparse(statement.value)
    return found


def phase1_integer(node: ast.AST) -> int:
    from protected.phase1_source_transform import _integer_expression

    return _integer_expression(node)


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class Checks:
    def __init__(self, strict: bool):
        self.strict = strict
        self.rows: list[dict] = []

    def add(self, name: str, status: str, detail) -> None:
        self.rows.append({"check": name, "status": status, "detail": detail})

    def pod(self, name: str, detail) -> None:
        """A check that can only pass on the restored pod: FAIL when executing."""
        self.add(name, FAIL if self.strict else POD_ONLY, detail)

    @property
    def failures(self) -> list[dict]:
        return [row for row in self.rows if row["status"] == FAIL]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def load_plan() -> tuple[dict, str]:
    raw = PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    if plan.get("format") != PLAN_FORMAT:
        raise RuntimeError(f"unexpected plan format: {plan.get('format')}")
    return plan, sha256_bytes(raw)


def output_root(plan: dict) -> Path:
    root = (ROOT / plan["output_root"]).resolve()
    phase1_root = PHASE1_RESULTS.resolve()
    if root == phase1_root or phase1_root in root.parents or root in phase1_root.parents:
        raise RuntimeError(f"output root overlaps results/phase1: {root}")
    return root


def stage_runs(plan: dict, stage: str) -> list[dict]:
    return sorted(plan[stage]["runs"], key=lambda row: row["order"])


def cell_of(plan: dict, run: dict) -> dict:
    return plan["cells"][run["cell"]]


def recipe_of(plan: dict, run: dict) -> dict:
    return plan["recipes"][run["recipe"]]


def attempt_dirname(run: dict, attempt: int) -> str:
    return run["run_id"] if attempt == 1 else f"{run['run_id']}-attempt{attempt}"


def validate_plan_structure(plan: dict, checks: Checks) -> None:
    batch = plan["batch"]["TOTAL_BATCH_SIZE"]
    problems: list[str] = []
    for key, recipe in plan["recipes"].items():
        source = ROOT / recipe["source_path"]
        if not source.is_file():
            problems.append(f"{key}: source missing {recipe['source_path']}")
            continue
        actual = sha256(source)
        if actual != recipe["source_sha256"]:
            problems.append(f"{key}: source sha256 {actual} != {recipe['source_sha256']}")
        text = source.read_text()
        if read_total_batch_size(text) != batch:
            problems.append(f"{key}: TOTAL_BATCH_SIZE != {batch}")
        constants = module_constants(source, {"DEVICE_BATCH_SIZE"})
        if constants.get("DEVICE_BATCH_SIZE") != plan["batch"]["DEVICE_BATCH_SIZE"]:
            problems.append(f"{key}: DEVICE_BATCH_SIZE {constants.get('DEVICE_BATCH_SIZE')}")
    for name, cell in plan["cells"].items():
        if cell["contract"] == "fixed_tokens":
            budget = cell["token_budget"]
            if budget % batch or budget % (4 * batch):
                problems.append(f"{name}: budget not divisible by batch and 4 x batch")
            if budget // batch != cell["optimizer_updates"]:
                problems.append(f"{name}: optimizer_updates mismatch")
            expected_milestones = [budget // 4, budget // 2, (budget * 3) // 4, budget]
            if cell["milestone_tokens"] != expected_milestones:
                problems.append(f"{name}: milestone tokens mismatch")
        elif cell["contract"] == "fixed_time":
            if cell["time_budget_seconds"] != 300:
                problems.append(f"{name}: time budget must be 300 s")
        else:
            problems.append(f"{name}: unknown contract {cell['contract']}")

    all_ids: set[str] = set()
    excluded = set()
    for values in plan["seeds"]["excluded_from_E1"].values():
        excluded.update(values)
    for stage in ("E0", "E1"):
        runs = stage_runs(plan, stage)
        orders = [row["order"] for row in runs]
        if orders != list(range(1, len(runs) + 1)):
            problems.append(f"{stage}: order is not 1..{len(runs)}")
        for row in runs:
            if row["stage"] != stage:
                problems.append(f"{row['run_id']}: stage field {row['stage']} != {stage}")
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", row["run_id"]) or row["run_id"] in all_ids:
                problems.append(f"invalid or duplicate run_id {row['run_id']}")
            all_ids.add(row["run_id"])
            recipe, cell = recipe_of(plan, row), cell_of(plan, row)
            if row["source_path"] != recipe["source_path"] or row["source_sha256"] != recipe["source_sha256"]:
                problems.append(f"{row['run_id']}: source fields disagree with recipe")
            if row.get("token_budget") != cell.get("token_budget") or row.get("time_budget_seconds") != cell.get("time_budget_seconds"):
                problems.append(f"{row['run_id']}: budget fields disagree with cell")
    e1 = stage_runs(plan, "E1")
    design = {(r, c, s) for r in plan["recipes"] for c in plan["cells"] for s in plan["seeds"]["E1"]}
    observed = [(row["recipe"], row["cell"], row["seed"]) for row in e1]
    if set(observed) != design or len(observed) != len(design) or len(design) != 24:
        problems.append("E1 run list is not the full 2 recipes x 4 cells x 3 seeds design")
    if excluded & set(plan["seeds"]["E1"]):
        problems.append("E1 seeds overlap a previously used seed set")
    e0 = stage_runs(plan, "E0")
    if len(e0) != 2:
        problems.append("E0 must contain exactly 2 runs")
    checks.add(
        "plan_structure_and_sources",
        FAIL if problems else OK,
        problems or f"{len(e0)} E0 + {len(e1)} E1 runs; both frozen sources hash-verified; batch {batch}",
    )


def validate_implementation_hashes(plan: dict, checks: Checks) -> None:
    mismatched = {}
    for relative, expected in plan["implementation_sha256"].items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatched[relative] = {"expected": expected, "actual": actual}
    checks.add(
        "implementation_sha256",
        FAIL if mismatched else OK,
        mismatched or f"{len(plan['implementation_sha256'])} files match the pre-registered hashes",
    )


def validate_phase1_identity(plan: dict, checks: Checks) -> None:
    problems = []
    try:
        panel = phase1.load_plan()  # verifies all 7 frozen panel sources and the order
    except Exception as exc:  # pragma: no cover - reported, not raised
        checks.add("phase1_plan_integrity", FAIL, f"{type(exc).__name__}: {exc}")
        return
    by_id = {model["model_id"]: model for model in panel["models"]}
    for key, recipe in plan["recipes"].items():
        model = by_id.get(recipe["phase1_model_id"])
        if model is None:
            problems.append(f"{key}: Phase-1 model {recipe['phase1_model_id']} missing")
            continue
        if (model["source_path"], model["source_sha256"]) != (recipe["source_path"], recipe["source_sha256"]):
            problems.append(f"{key}: differs from Phase-1 {model['model_id']} source")
        if model["tokens_per_optimizer_step"] != plan["batch"]["TOTAL_BATCH_SIZE"]:
            problems.append(f"{key}: Phase-1 tokens_per_optimizer_step differs")
    checks.add(
        "phase1_source_identity",
        FAIL if problems else OK,
        problems or "AP-Q == Phase-1 autopareto_292 source and AR-ref == Phase-1 autoresearch_compatible source (same path, same sha256)",
    )

    # The protected transform must still reproduce the panel's executed derived sources.
    reproduced = {}
    for row in plan["execution"]["token_cells"]["panel_derived_source_reproduction"]:
        recipe = plan["recipes"][row["recipe"]]
        run_dir = Path(row["panel_run_dir"])
        derived = transform_source(
            (ROOT / recipe["source_path"]).read_text(),
            source_sha256=recipe["source_sha256"],
            token_budget=row["token_budget"],
            seed=row["seed"],
            milestone_dir=run_dir / "milestones",
            final_checkpoint=run_dir / "final.pt",
            step_timing_path=run_dir / "step_timing.jsonl",
            timing_summary_path=run_dir / "timing_summary.json",
            telemetry_path=run_dir / "telemetry.json",
            telemetry_interval_seconds=plan["execution"]["token_cells"]["telemetry_sampling_interval_seconds"],
            telemetry_mode="enabled",
        )
        reproduced[row["panel_run_id"]] = sha256_bytes(derived.encode()) == row["derived_source_sha256"]
    checks.add(
        "phase1_transform_reproduces_panel_derived_sources",
        OK if all(reproduced.values()) else FAIL,
        reproduced,
    )


def validate_token_transforms(plan: dict, checks: Checks, stage_root_by_stage: dict) -> None:
    problems = []
    count = 0
    for stage in ("E0", "E1"):
        for run in stage_runs(plan, stage):
            cell = cell_of(plan, run)
            if cell["contract"] != "fixed_tokens":
                continue
            recipe = recipe_of(plan, run)
            run_dir = stage_root_by_stage[stage] / "runs" / run["run_id"]
            try:
                derived = transform_source(
                    (ROOT / recipe["source_path"]).read_text(),
                    source_sha256=recipe["source_sha256"],
                    token_budget=cell["token_budget"],
                    seed=run["seed"],
                    milestone_dir=run_dir / "milestones",
                    final_checkpoint=run_dir / "final.pt",
                    step_timing_path=run_dir / "step_timing.jsonl",
                    timing_summary_path=run_dir / "timing_summary.json",
                    telemetry_path=run_dir / "telemetry.json",
                    telemetry_interval_seconds=plan["execution"]["token_cells"]["telemetry_sampling_interval_seconds"],
                )
            except ValueError as exc:
                problems.append(f"{run['run_id']}: {exc}")
                continue
            target = f"PHASE1_TARGET_STEPS = {cell['optimizer_updates']}\n"
            if target not in derived or f"seed={run['seed']}," not in derived:
                problems.append(f"{run['run_id']}: derived source lacks the expected horizon/seed")
            batch = plan["batch"]["TOTAL_BATCH_SIZE"]
            if any(tokens % batch for tokens in cell["milestone_tokens"]):
                problems.append(f"{run['run_id']}: a milestone is not reachable at batch {batch}")
            count += 1
    checks.add(
        "phase1_transform_accepts_every_budget_and_seed",
        FAIL if problems else OK,
        problems or f"{count} fixed-token runs: transform accepted; 4 milestones reachable per budget",
    )


# ---------------------------------------------------------------------------
# Fixed-time (historical launcher) helpers
# ---------------------------------------------------------------------------

def resolve_fixed_time_launcher(plan: dict, override: Path | None) -> tuple[bytes | None, str]:
    spec = plan["execution"]["fixed_time_cell"]
    expected = spec["launcher_sha256"]
    candidates: list[tuple[str, callable]] = []
    if override is not None:
        candidates.append((str(override), lambda: override.read_bytes()))
    for relative in spec["launcher_local_copies"]:
        path = ROOT / relative
        candidates.append((relative, lambda path=path: path.read_bytes()))
    git = shutil.which("git")
    if git and (ROOT / ".git").exists():
        blob = spec["launcher_git_blob"]
        candidates.append((
            f"git cat-file blob {blob}",
            lambda: subprocess.run([git, "cat-file", "blob", blob], cwd=ROOT, capture_output=True, check=True).stdout,
        ))
    tried = []
    for label, reader in candidates:
        try:
            payload = reader()
        except (OSError, subprocess.CalledProcessError):
            tried.append(f"{label}: unavailable")
            continue
        if sha256_bytes(payload) == expected:
            return payload, label
        tried.append(f"{label}: sha256 {sha256_bytes(payload)}")
    return None, "; ".join(tried) or "no candidate locations"


def validate_fixed_time(plan: dict, checks: Checks, override: Path | None) -> None:
    spec = plan["execution"]["fixed_time_cell"]
    payload, label = resolve_fixed_time_launcher(plan, override)
    checks.add(
        "historical_fixed_time_launcher",
        OK if payload is not None else FAIL,
        f"resolved from {label}" if payload is not None else label,
    )
    config = module_constants(ROOT / "benchmark_config.py", {
        "TRAINING_BUDGET_SECONDS", "TRAINING_TIMEOUT_SECONDS", "TRAINING_MAX_OVERRUN_SECONDS", "CUDA_ALLOCATOR",
    })
    prepare = module_constants(ROOT / "prepare.py", {"TIME_BUDGET", "MAX_SEQ_LEN", "EVAL_TOKENS"})
    expected = spec["protected_constants"]
    observed = {**{f"benchmark_config.{k}": v for k, v in config.items()}, **{f"prepare.{k}": v for k, v in prepare.items()}}
    mismatch = {k: {"expected": v, "actual": observed.get(k)} for k, v in expected.items() if observed.get(k) != v}
    checks.add("fixed_time_protected_constants", FAIL if mismatch else OK, mismatch or observed)
    missing = [rel for rel in spec["isolate_files"] if not (ROOT / rel).is_file()]
    checks.add("fixed_time_isolate_inputs", FAIL if missing else OK, missing or spec["isolate_files"])


def fixed_time_paths(stage_root: Path, dirname: str) -> dict[str, Path]:
    run_dir = stage_root / "runs" / dirname
    return {
        "run_dir": run_dir,
        "isolate": run_dir / "isolate",
        "checkpoint": run_dir / "final.pt",
        "metrics": run_dir / "fixed_time_training_metrics.json",
        "event_log": run_dir / "events.jsonl",
        "training_log": run_dir / "training.log",
        "manifest": run_dir / "fixed_time_checkpoint_manifest.json",
        "evaluations": run_dir / "checkpoint_evaluations.json",
        "evaluation_log": run_dir / "evaluation.log",
        "compile_cache": run_dir / "compile_cache",
    }


def fixed_time_commands(plan: dict, stage_root: Path, run: dict, dirname: str) -> dict:
    """Single source of truth for the two fixed-time subprocess invocations."""
    spec = plan["execution"]["fixed_time_cell"]
    paths = fixed_time_paths(stage_root, dirname)
    env_delta = {
        "PYTORCH_CUDA_ALLOC_CONF": spec["environment"]["PYTORCH_CUDA_ALLOC_CONF"],
        "TORCHINDUCTOR_CACHE_DIR": str(paths["compile_cache"] / "torchinductor"),
        "TRITON_CACHE_DIR": str(paths["compile_cache"] / "triton"),
        "CUDA_CACHE_PATH": str(paths["compile_cache"] / "cuda"),
    }
    training = phase1.command_prefix() + [
        "protected/train_launcher.py",
        "--train", "train.py",
        "--checkpoint", str(paths["checkpoint"]),
        "--metrics", str(paths["metrics"]),
        "--seed", str(run["seed"]),
        "--event-log", str(paths["event_log"]),
        "--campaign-id", f"camera-ready-2026-{dirname}",
    ]
    evaluation = phase1.command_prefix() + [
        "-m", "scripts.evaluate_phase1_checkpoints",
        "--candidate", str(paths["isolate"] / "train.py"),
        "--milestone-manifest", str(paths["manifest"]),
        "--output", str(paths["evaluations"]),
        "--heldout",
    ]
    return {
        "training": {"argv": training, "cwd": str(paths["isolate"]), "env": env_delta, "timeout_seconds": spec["timeout_seconds"]},
        "evaluation": {"argv": evaluation, "cwd": str(ROOT), "env": {}},
    }


# ---------------------------------------------------------------------------
# Phase-1 reuse (fixed-token cells)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def phase1_redirected(stage_root: Path, recorder=None):
    """Point Phase-1 run_one/evaluate_one at the camera-ready stage root."""
    saved_root, saved_subprocess = phase1.PHASE1_ROOT, phase1.subprocess
    if PHASE1_RESULTS.resolve() in (stage_root.resolve(), *stage_root.resolve().parents):
        raise RuntimeError("refusing to point the Phase-1 executor at results/phase1")
    phase1.PHASE1_ROOT = stage_root
    if recorder is not None:
        phase1.subprocess = types.SimpleNamespace(run=recorder, STDOUT=subprocess.STDOUT)
    try:
        yield
    finally:
        phase1.PHASE1_ROOT, phase1.subprocess = saved_root, saved_subprocess


def token_views(plan: dict, run: dict, dirname: str, attempt: int) -> tuple[dict, dict]:
    recipe, cell = recipe_of(plan, run), cell_of(plan, run)
    plan_view = {
        "models": [{
            "model_id": recipe["phase1_model_id"],
            "source_path": recipe["source_path"],
            "source_sha256": recipe["source_sha256"],
        }],
        "token_budget": cell["token_budget"],
        "telemetry_sampling_interval_seconds": plan["execution"]["token_cells"]["telemetry_sampling_interval_seconds"],
        "seeds": [],
        "preregistered_model_order": {},
    }
    run_view = {
        "run_id": dirname,
        "model_id": recipe["phase1_model_id"],
        "seed": run["seed"],
        "telemetry_mode": "enabled",
        "model_order_position": run["order"],
        "retried": attempt > 1,
        "scientific": True,
    }
    return plan_view, run_view


def captured_token_commands(plan: dict, stage_root: Path, run: dict, dirname: str, attempt: int) -> dict:
    """Exact Phase-1 commands, captured by intercepting subprocess.run in a temp root."""
    calls: list[dict] = []

    def recorder(argv, **kwargs):
        calls.append({"argv": list(argv), "cwd": kwargs.get("cwd"), "env": kwargs.get("env")})
        return subprocess.CompletedProcess(argv, 0)

    plan_view, run_view = token_views(plan, run, dirname, attempt)
    with tempfile.TemporaryDirectory() as temporary:
        fake_root = Path(temporary) / stage_root.name
        with phase1_redirected(fake_root, recorder):
            phase1.run_one(plan_view, run_view)
            phase1.evaluate_one(plan_view, run_view)
        fake, real = str(fake_root), str(stage_root)
        result = {}
        for name, call in zip(("training", "evaluation"), calls):
            env = call["env"] or {}
            delta = {k: v.replace(fake, real) for k, v in env.items() if os.environ.get(k) != v}
            result[name] = {
                "argv": [item.replace(fake, real) for item in call["argv"]],
                "cwd": str(call["cwd"] or ROOT),
                "env": delta,
            }
    if len(calls) != 2:
        raise RuntimeError(f"expected 2 Phase-1 subprocess calls, captured {len(calls)}")
    return result


def render(command: dict) -> str:
    env = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(command["env"].items()))
    prefix = f"cd {shlex.quote(command['cwd'])} && "
    return prefix + (f"env {env} " if env else "") + shlex.join(command["argv"])


def run_commands(plan: dict, stage_root: Path, run: dict, attempt: int) -> dict:
    dirname = attempt_dirname(run, attempt)
    if cell_of(plan, run)["contract"] == "fixed_tokens":
        return captured_token_commands(plan, stage_root, run, dirname, attempt)
    return fixed_time_commands(plan, stage_root, run, dirname)


# ---------------------------------------------------------------------------
# Environment / pod-only checks
# ---------------------------------------------------------------------------

def git_state() -> dict:
    git = shutil.which("git")
    if not git or not (ROOT / ".git").exists():
        return {"available": False}
    head = subprocess.run([git, "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run([git, "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return {"available": True, "head": head.stdout.strip(), "dirty_entries": len(status.stdout.splitlines())}


def nvidia_smi(query: list[str]) -> list[str] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    completed = subprocess.run([exe, *query, "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    if completed.returncode:
        return None
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def kernel_cache_revisions() -> dict:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    found = {}
    for repo in hub.glob("models--kernels-community--flash-attn3*"):
        refs = repo / "refs"
        found[repo.name] = {ref.name: ref.read_text().strip() for ref in refs.glob("*")} if refs.is_dir() else {}
    return {"hub": str(hub), "repos": found}


def uv_lock_problem(plan: dict) -> str | None:
    """`uv run --locked` (Phase-1 command_prefix) cannot start without the panel lockfile."""
    if phase1.command_prefix()[0] != "uv":
        return None
    lock = ROOT / "uv.lock"
    expected = plan["environment_expectations"]["uv_lock_sha256"]
    if not lock.is_file():
        return (
            "uv is on PATH, so Phase-1 command_prefix() runs `uv run --locked python`, which needs "
            f"{lock} (sha256 {expected}; private branch autopareto/research-infrastructure:uv.lock)"
        )
    if sha256(lock) != expected:
        return f"uv.lock sha256 {sha256(lock)} != {expected}"
    return None


PROBE = (
    "import json, platform, sys, torch; "
    "cuda = bool(torch.cuda.is_available()); "
    "print(json.dumps({'python': platform.python_version(), 'executable': sys.executable, "
    "'torch': torch.__version__, 'cuda_available': cuda, "
    "'gpu_name': torch.cuda.get_device_name(0) if cuda else None}))"
)


def probe_training_interpreter(plan: dict) -> dict:
    """Ask the interpreter the training subprocesses will use (never CUDA in this process)."""
    prefix = phase1.command_prefix()
    attempts = []
    if uv_lock_problem(plan) is None:
        attempts.append(("command_prefix", prefix))
    attempts.append(("this_interpreter", [sys.executable]))
    errors = []
    for label, argv in attempts:
        try:
            completed = subprocess.run(argv + ["-c", PROBE], cwd=ROOT, capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{label}: {type(exc).__name__}")
            continue
        if completed.returncode == 0 and completed.stdout.strip():
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            return {"probed_with": label, "argv": argv, **payload, "errors": errors}
        errors.append(f"{label}: exit {completed.returncode}: {completed.stderr.strip()[-300:]}")
    return {"probed_with": None, "errors": errors}


def nearest_existing(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path.resolve()


def validate_environment(plan: dict, checks: Checks, root_out: Path) -> dict:
    expect = plan["environment_expectations"]
    info: dict = {
        "executor_python": platform.python_version(),
        "executor_executable": sys.executable,
        "platform": platform.platform(),
        "root": str(ROOT),
        "git": git_state(),
        "command_prefix": phase1.command_prefix(),
    }
    problem = uv_lock_problem(plan)
    if problem:
        checks.pod("uv_lockfile", problem)
    elif info["command_prefix"][0] == "uv":
        checks.add("uv_lockfile", OK, "uv.lock matches the panel environment lock (262a4d55...)")
    else:
        checks.add("uv_lockfile", INFO, f"uv not on PATH; subprocesses use {sys.executable}")

    probe = probe_training_interpreter(plan)
    info["training_interpreter"] = probe
    trusted = probe.get("probed_with") == "command_prefix" or info["command_prefix"][0] != "uv"
    python = probe.get("python") or "unknown"
    if trusted and python.rsplit(".", 1)[0] == expect["python"].rsplit(".", 1)[0]:
        checks.add("python_version", OK if python == expect["python"] else INFO, f"{python} (panel: {expect['python']})")
    else:
        checks.pod("python_version", f"{python} via {probe.get('probed_with')} (panel: {expect['python']}); {probe.get('errors')}")
    if trusted and probe.get("torch") == expect["torch"]:
        checks.add("torch_version", OK, probe["torch"])
    else:
        checks.pod("torch_version", f"{probe.get('torch')} via {probe.get('probed_with')} (panel: {expect['torch']})")
    if trusted and probe.get("cuda_available") and expect["gpu_name_contains"] in (probe.get("gpu_name") or ""):
        checks.add("cuda_gpu", OK, probe["gpu_name"])
    else:
        checks.pod("cuda_gpu", f"CUDA available={probe.get('cuda_available')} gpu={probe.get('gpu_name')}; an NVIDIA A40 is required")

    gpus = nvidia_smi(["--query-gpu=name,driver_version,uuid,memory.total"])
    info["nvidia_smi_gpus"] = gpus
    apps = nvidia_smi(["--query-compute-apps=pid,process_name,used_memory"])
    info["gpu_processes"] = apps
    if gpus is None:
        checks.pod("gpu_idle", "nvidia-smi unavailable")
    elif apps:
        checks.add("gpu_idle", FAIL if checks.strict else POD_ONLY, f"other GPU processes: {apps}")
    else:
        checks.add("gpu_idle", OK, f"{gpus}; no compute processes")

    for name in plan["execution"]["token_cells"]["parent_environment_must_not_set"]:
        if os.environ.get(name):
            checks.add(f"env_{name}", FAIL, (
                f"{name}={os.environ[name]!r} is set; Phase-1 token runs inherited the default allocator. "
                "Unset it (the fixed-time cell sets the historical allocator itself)."
            ))
    info["kernel_cache"] = kernel_cache_revisions()

    # Persistent storage on RunPod.
    workspace = Path("/workspace")
    if workspace.is_dir():
        anchor = nearest_existing(root_out)
        inside = workspace.resolve() in (anchor, *anchor.parents)
        checks.add("persistent_output_root", OK if inside else FAIL, f"{root_out} resolves under /workspace: {inside}")
        usage = shutil.disk_usage(workspace)
        free_gb = usage.free / 1e9
        need = plan["execution"]["minimum_free_disk_gb"]
        checks.add("free_disk", OK if free_gb >= need else FAIL, f"{free_gb:.1f} GB free on /workspace (need {need})")
    else:
        checks.pod("persistent_output_root", "/workspace is absent (not a RunPod pod)")

    # Training/validation data cache and tokenizer (read-only here).
    gate = plan["data_identity_gate"]
    cache = Path(os.path.expanduser(gate["runtime_cache"]))
    if not cache.exists():
        checks.pod("data_cache", f"{cache} absent (tokenizer, 10 training shards, validation shard_06542 live on the pod)")
    else:
        bad = []
        for row in gate["files"]:
            path = cache / row["path"]
            if not path.is_file():
                bad.append(f"missing {row['path']}")
            elif path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
                bad.append(f"content differs {row['path']}")
        checks.add("data_cache", FAIL if bad else OK, bad or f"{len(gate['files'])} files match the data manifest")
    heldout = Path(os.path.expanduser(gate["heldout_cache_path"]))
    if not heldout.is_file():
        checks.pod("heldout_shard_cache", f"{heldout} absent (downstream_evaluator downloads it from the pinned revision if missing)")
    elif sha256(heldout) != gate["heldout_sha256"]:
        checks.add("heldout_shard_cache", FAIL, f"{heldout} sha256 differs")
    else:
        checks.add("heldout_shard_cache", OK, str(heldout))
    return info


def validate_output_root(plan: dict, checks: Checks, root_out: Path) -> dict:
    state = {}
    for stage in ("E0", "E1"):
        runs_dir = root_out / stage / "runs"
        existing = sorted(p.name for p in runs_dir.iterdir()) if runs_dir.is_dir() else []
        state[stage] = existing
    checks.add("output_root", OK, {
        "root": str(root_out),
        "exists": root_out.exists(),
        "existing_run_dirs": state,
        "results_phase1_untouched": "output root is disjoint from results/phase1",
    })
    verdict = read_json(root_out / "E0" / "e0_verdict.json")
    checks.add("e0_verdict", INFO, verdict.get("status") if verdict else "absent: E1 --execute is blocked until E0 PASS")
    return state


def validate(plan: dict, plan_sha: str, *, strict: bool, launcher_override: Path | None) -> tuple[Checks, dict]:
    checks = Checks(strict)
    root_out = output_root(plan)
    stage_roots = {stage: root_out / stage for stage in ("E0", "E1")}
    validate_plan_structure(plan, checks)
    validate_implementation_hashes(plan, checks)
    validate_phase1_identity(plan, checks)
    validate_token_transforms(plan, checks, stage_roots)
    validate_fixed_time(plan, checks, launcher_override)
    validate_output_root(plan, checks, root_out)
    info = validate_environment(plan, checks, root_out)
    info["plan_sha256"] = plan_sha
    return checks, info


# ---------------------------------------------------------------------------
# Integrity gates
# ---------------------------------------------------------------------------

def heldout_ok(evals: dict, plan: dict, failures: list[str]) -> dict:
    gate = plan["data_identity_gate"]
    heldout = evals.get("heldout_bpb")
    if not isinstance(heldout, dict) or not finite(heldout.get("metric_value")):
        failures.append("held-out BPB missing")
        return {}
    if heldout.get("dataset_revision") != gate["heldout_dataset_revision"] or heldout.get("dataset_config") != gate["heldout_filename"]:
        failures.append("held-out BPB used an unregistered dataset revision or shard")
    raw = heldout.get("raw") or {}
    if raw.get("shard_sha256") != gate["heldout_sha256"] or raw.get("evaluated_tokens") != gate["heldout_evaluated_tokens"]:
        failures.append("held-out shard sha256 or evaluated token count differs from the registered shard")
    return heldout


def sanity(value, name: str, plan: dict, failures: list[str]) -> None:
    bound = plan["integrity_gates"]["bpb_sanity_upper_bound"]
    if not finite(value) or not 0.5 < float(value) < bound:
        failures.append(f"{name}={value} outside the sanity range (0.5, {bound})")


def implementation_failures(plan: dict) -> list[str]:
    return [
        f"implementation hash changed: {relative}"
        for relative, expected in plan["implementation_sha256"].items()
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != expected
    ]


def gate_token_run(plan: dict, run: dict, run_dir: Path) -> tuple[list[str], dict]:
    failures: list[str] = []
    cell, recipe = cell_of(plan, run), recipe_of(plan, run)
    batch = plan["batch"]["TOTAL_BATCH_SIZE"]
    budget = cell["token_budget"]
    steps = budget // batch
    metrics = read_json(run_dir / "training_metrics.json")
    evals = read_json(run_dir / "checkpoint_evaluations.json")
    manifest_path = run_dir / "milestones" / "manifest.json"
    manifest = read_json(manifest_path)
    if not metrics or "failure_reason" in metrics:
        return [f"training failed: {(metrics or {}).get('failure_reason', 'no training_metrics.json')}"], {}
    if not manifest or not evals:
        return ["milestone manifest or checkpoint evaluations missing"], {}
    expect = {
        "format": "autopareto_phase1_training_metrics_v3",
        "source_sha256": recipe["source_sha256"],
        "training_seed": run["seed"],
        "total_training_tokens": budget,
        "optimizer_steps": steps,
        "tokens_per_optimizer_step": batch,
        "steady_state_tokens": max(0, steps - 11) * batch,
        "primary_training_efficiency_metric": "steady_state_equivalent_seconds",
        "cuda_event_profile": "disabled_scientific",
    }
    for key, value in expect.items():
        if metrics.get(key) != value:
            failures.append(f"training_metrics.{key}={metrics.get(key)!r} expected {value!r}")
    for key in ("step_timing_artifact", "timing_summary_artifact", "telemetry_artifact", "run_metadata"):
        if not Path(str(metrics.get(key))).is_file():
            failures.append(f"artifact missing: {key}")
    telemetry = read_json(Path(str(metrics.get("telemetry_artifact")))) or {}
    if telemetry.get("sampling_interval_seconds") != plan["execution"]["token_cells"]["telemetry_sampling_interval_seconds"] or not telemetry.get("samples"):
        failures.append("telemetry interval/samples invalid")
    timing_path = Path(str(metrics.get("step_timing_artifact")))
    if timing_path.is_file():
        with timing_path.open() as stream:
            lines = sum(1 for line in stream if line.strip())
        if lines != steps:
            failures.append(f"step_timing.jsonl has {lines} rows, expected {steps}")
    final = run_dir / "final.pt"
    if not final.is_file() or sha256(final) != metrics.get("checkpoint_sha256"):
        failures.append("final checkpoint missing or hash differs from training_metrics")
    # P4: milestone/checkpoint step counts equal the observed optimizer updates.
    rows = manifest.get("checkpoints", [])
    if [row.get("token_count") for row in rows] != cell["milestone_tokens"]:
        failures.append("milestone token counts differ from the registered schedule")
    if any(row.get("optimizer_step") != row.get("token_count", -1) // batch for row in rows):
        failures.append("milestone optimizer_step != token_count / batch")
    if not rows or rows[-1].get("optimizer_step") != metrics.get("optimizer_steps") or Path(rows[-1].get("checkpoint_path", "")) != final.resolve():
        failures.append("final milestone is not the observed final optimizer step / final.pt")
    if manifest.get("seed") != run["seed"] or manifest.get("source_sha256") != recipe["source_sha256"] or manifest.get("token_budget") != budget:
        failures.append("milestone manifest seed/source/budget mismatch")
    # Evaluator output belongs to this run and this exact derived source.
    if evals.get("candidate_sha256") != metrics.get("derived_source_sha256"):
        failures.append("evaluator candidate hash != derived source hash")
    if evals.get("milestone_manifest_sha256") != sha256(manifest_path):
        failures.append("evaluator milestone manifest hash differs")
    eval_rows = evals.get("checkpoints", [])
    if [row.get("token_count") for row in eval_rows] != cell["milestone_tokens"]:
        failures.append("evaluated milestones differ from the registered schedule")
    heldout = heldout_ok(evals, plan, failures)
    validation = eval_rows[-1].get("validation_bpb") if eval_rows else None
    in_process = metrics.get("val_bpb")
    sanity(validation, "validation_bpb", plan, failures)
    sanity(heldout.get("metric_value"), "heldout_bpb", plan, failures)
    tolerance = plan["integrity_gates"]["in_process_vs_checkpoint_validation_tolerance"]
    if finite(validation) and finite(in_process) and abs(float(validation) - float(in_process)) > tolerance:
        failures.append(f"checkpoint validation {validation} disagrees with in-process {in_process} (> {tolerance})")
    failures.extend(implementation_failures(plan))
    values = {
        "validation_bpb": validation,
        "validation_bpb_in_process": in_process,
        "heldout_bpb": heldout.get("metric_value"),
        "heldout_ci95": [heldout.get("confidence_low"), heldout.get("confidence_high")],
        "tokens": metrics.get("total_training_tokens"),
        "updates": metrics.get("optimizer_steps"),
        "steady_state_seconds": metrics.get("steady_state_training_seconds"),
        "steady_state_tokens_per_second": metrics.get("steady_state_training_tokens_per_second"),
        "total_scientific_step_wall_seconds": metrics.get("total_scientific_step_wall_seconds"),
        "peak_training_memory_mb": metrics.get("peak_training_cuda_memory_mb"),
        "parameter_count": metrics.get("parameter_count"),
        "end_to_end_training_seconds": metrics.get("end_to_end_runtime_seconds"),
        "milestone_validation_bpb": {str(row.get("token_count")): row.get("validation_bpb") for row in eval_rows},
        "checkpoint_sha256": metrics.get("checkpoint_sha256"),
        "derived_source_sha256": metrics.get("derived_source_sha256"),
    }
    return failures, values


def logged_num_steps(log: Path) -> int | None:
    if not log.is_file():
        return None
    matches = re.findall(r"^num_steps:\s+(\d+)\s*$", log.read_text(errors="replace"), flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


def gate_fixed_time_run(plan: dict, run: dict, paths: dict[str, Path]) -> tuple[list[str], dict]:
    failures: list[str] = []
    spec = plan["execution"]["fixed_time_cell"]
    recipe = recipe_of(plan, run)
    batch = plan["batch"]["TOTAL_BATCH_SIZE"]
    metrics = read_json(paths["metrics"])
    evals = read_json(paths["evaluations"])
    if not metrics or "failure_reason" in metrics:
        return [f"training failed: {(metrics or {}).get('failure_reason', 'no fixed-time metrics')}"], {}
    if not evals:
        return ["checkpoint evaluations missing"], {}
    if sha256(paths["isolate"] / "train.py") != recipe["source_sha256"]:
        failures.append("isolate train.py hash differs from the frozen recipe")
    if sha256(paths["isolate"] / "protected/train_launcher.py") != spec["launcher_sha256"]:
        failures.append("isolate launcher hash differs from the historical launcher")
    seconds = metrics.get("training_seconds")
    low, high = spec["training_seconds_window"]
    if not finite(seconds) or not low <= float(seconds) <= high:
        failures.append(f"training_seconds {seconds} outside [{low}, {high}]")
    if metrics.get("training_seed") != run["seed"]:
        failures.append("training seed mismatch")
    tokens = metrics.get("total_training_tokens")
    if not isinstance(tokens, int) or tokens <= 0 or tokens % batch:
        failures.append(f"total_training_tokens {tokens} not a positive multiple of {batch}")
        updates = None
    else:
        updates = tokens // batch
        if metrics.get("measured_training_tokens") != max(0, updates - 11) * batch:
            failures.append("measured_training_tokens != (updates - 11) x batch")
    logged = logged_num_steps(paths["training_log"])
    if logged != updates:
        failures.append(f"logged num_steps {logged} != tokens / batch {updates}")
    checkpoint_hash = sha256(paths["checkpoint"]) if paths["checkpoint"].is_file() else None
    manifest = read_json(paths["manifest"]) or {}
    rows = manifest.get("checkpoints", [])
    if checkpoint_hash is None or len(rows) != 1 or rows[0].get("checkpoint_sha256") != checkpoint_hash:
        failures.append("fixed-time checkpoint missing or manifest hash differs")
    if rows and (rows[0].get("token_count") != tokens or rows[0].get("optimizer_step") != updates):
        failures.append("manifest token/update count differs from launcher metrics")
    if evals.get("candidate_sha256") != recipe["source_sha256"]:
        failures.append("evaluator candidate is not the frozen recipe")
    if evals.get("milestone_manifest_sha256") != (sha256(paths["manifest"]) if paths["manifest"].is_file() else None):
        failures.append("evaluator manifest hash differs")
    eval_rows = evals.get("checkpoints", [])
    if len(eval_rows) != 1 or eval_rows[0].get("checkpoint_sha256") != checkpoint_hash:
        failures.append("evaluator did not score exactly the fixed-time checkpoint")
    heldout = heldout_ok(evals, plan, failures)
    validation = eval_rows[0].get("validation_bpb") if eval_rows else None
    in_process = metrics.get("val_bpb")
    sanity(validation, "validation_bpb", plan, failures)
    sanity(in_process, "validation_bpb_in_process", plan, failures)
    sanity(heldout.get("metric_value"), "heldout_bpb", plan, failures)
    tolerance = plan["integrity_gates"]["in_process_vs_checkpoint_validation_tolerance"]
    if finite(validation) and finite(in_process) and abs(float(validation) - float(in_process)) > tolerance:
        failures.append(f"checkpoint validation {validation} disagrees with in-process {in_process} (> {tolerance})")
    failures.extend(implementation_failures(plan))
    values = {
        "validation_bpb": validation,
        "validation_bpb_in_process": in_process,
        "heldout_bpb": heldout.get("metric_value"),
        "heldout_ci95": [heldout.get("confidence_low"), heldout.get("confidence_high")],
        "tokens": tokens,
        "updates": updates,
        "steady_state_seconds": seconds,
        "steady_state_tokens_per_second": metrics.get("training_tokens_per_second"),
        "peak_training_memory_mb": metrics.get("peak_training_cuda_memory_mb"),
        "parameter_count": metrics.get("parameter_count"),
        "end_to_end_training_seconds": metrics.get("total_runtime_seconds"),
        "checkpoint_sha256": checkpoint_hash,
    }
    return failures, values


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def assert_gpu_idle() -> None:
    apps = nvidia_smi(["--query-compute-apps=pid,process_name,used_memory"])
    if apps is None:
        raise RuntimeError("nvidia-smi unavailable; refusing to run without a GPU idle check")
    if apps:
        raise RuntimeError(f"GPU is not idle: {apps}")


def execute_token_attempt(plan: dict, stage_root: Path, run: dict, attempt: int) -> tuple[list[str], dict]:
    dirname = attempt_dirname(run, attempt)
    run_dir = stage_root / "runs" / dirname
    if run_dir.exists():
        raise RuntimeError(f"refusing to reuse run directory {run_dir}")
    plan_view, run_view = token_views(plan, run, dirname, attempt)
    failures: list[str] = []
    with phase1_redirected(stage_root):
        try:
            phase1.run_one(plan_view, run_view)
            phase1.evaluate_one(plan_view, run_view)
        except SystemExit as exc:
            failures.append(f"Phase-1 launcher exited with {exc.code} (see training.log)")
        except subprocess.CalledProcessError as exc:
            failures.append(f"Phase-1 evaluator exited with {exc.returncode}")
    if failures:
        gate_failures, values = gate_token_run(plan, run, run_dir)
        return failures + gate_failures, values
    return gate_token_run(plan, run, run_dir)


def execute_fixed_time_attempt(plan: dict, stage_root: Path, run: dict, attempt: int, launcher: bytes) -> tuple[list[str], dict]:
    spec = plan["execution"]["fixed_time_cell"]
    recipe = recipe_of(plan, run)
    dirname = attempt_dirname(run, attempt)
    paths = fixed_time_paths(stage_root, dirname)
    if paths["run_dir"].exists():
        raise RuntimeError(f"refusing to reuse run directory {paths['run_dir']}")
    paths["isolate"].mkdir(parents=True)
    (paths["isolate"] / "protected").mkdir()
    for relative in spec["isolate_files"]:
        shutil.copy2(ROOT / relative, paths["isolate"] / relative)
    (paths["isolate"] / "protected/train_launcher.py").write_bytes(launcher)
    shutil.copy2(ROOT / recipe["source_path"], paths["isolate"] / "train.py")
    for relative in spec["isolate_files"]:
        if sha256(paths["isolate"] / relative) != plan["implementation_sha256"][relative]:
            raise RuntimeError(f"isolate copy of {relative} does not match the plan")
    commands = fixed_time_commands(plan, stage_root, run, dirname)
    training = commands["training"]
    env = dict(os.environ)
    env.update(training["env"])
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    with paths["training_log"].open("w") as log:
        # Historical runner semantics (new session, 900 s timeout); on timeout the whole
        # process group is killed so no orphaned `uv` -> python child keeps the GPU.
        process = subprocess.Popen(
            training["argv"], cwd=training["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=training["timeout_seconds"])
            if returncode:
                failures.append(f"fixed-time launcher exited with {returncode} (see training.log)")
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            failures.append(f"fixed-time launcher timed out after {training['timeout_seconds']} s")
    metrics = read_json(paths["metrics"]) or {}
    if not failures and paths["checkpoint"].is_file() and finite(metrics.get("total_training_tokens")):
        tokens = int(metrics["total_training_tokens"])
        write_json_new(paths["manifest"], {
            "format": "autopareto_camera_ready_fixed_time_checkpoint_manifest_v1",
            "note": "single final checkpoint of a 300 s fixed-time run; consumed by scripts/evaluate_phase1_checkpoints.py",
            "device_batch_size": plan["batch"]["DEVICE_BATCH_SIZE"],
            "sequence_length": plan["batch"]["sequence_length"],
            "seed": run["seed"],
            "source_sha256": recipe["source_sha256"],
            "checkpoints": [{
                "token_count": tokens,
                "optimizer_step": tokens // plan["batch"]["TOTAL_BATCH_SIZE"],
                "historical_steady_state_training_seconds": metrics.get("training_seconds"),
                "checkpoint_path": str(paths["checkpoint"].resolve()),
                "checkpoint_bytes": paths["checkpoint"].stat().st_size,
                "checkpoint_sha256": sha256(paths["checkpoint"]),
            }],
        })
        evaluation = commands["evaluation"]
        with paths["evaluation_log"].open("w") as log:
            completed = subprocess.run(evaluation["argv"], cwd=evaluation["cwd"], stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            failures.append(f"checkpoint evaluator exited with {completed.returncode} (see evaluation.log)")
    elif not failures:
        failures.append("fixed-time launcher returned success without checkpoint/metrics")
    gate_failures, values = gate_fixed_time_run(plan, run, paths)
    return failures + gate_failures, values


def write_record(plan: dict, plan_sha: str, stage_root: Path, run: dict, attempt: int, failures: list[str], values: dict, started: float) -> dict:
    cell, recipe = cell_of(plan, run), recipe_of(plan, run)
    dirname = attempt_dirname(run, attempt)
    record = {
        "format": RECORD_FORMAT,
        "synthetic": False,
        "plan_sha256": plan_sha,
        "stage": run["stage"],
        "run_id": run["run_id"],
        "attempt": attempt,
        "run_dir": str(stage_root / "runs" / dirname),
        "order": run["order"],
        "recipe": run["recipe"],
        "phase1_model_id": recipe["phase1_model_id"],
        "source_sha256": recipe["source_sha256"],
        "cell": run["cell"],
        "contract": cell["contract"],
        "token_budget": cell.get("token_budget"),
        "time_budget_seconds": cell.get("time_budget_seconds"),
        "seed": run["seed"],
        "status": "completed" if not failures else "failed",
        "integrity": {"passed": not failures, "failures": failures},
        "metrics": values,
        "attempt_wall_seconds": time.time() - started,
        "finished_utc": utc_now(),
    }
    write_json_new(stage_root / "runs" / dirname / RECORD_NAME, record)
    return record


def existing_attempts(stage_root: Path, run: dict) -> list[tuple[int, dict | None]]:
    found = []
    for attempt in (1, 2):
        run_dir = stage_root / "runs" / attempt_dirname(run, attempt)
        if run_dir.exists():
            found.append((attempt, read_json(run_dir / RECORD_NAME)))
    return found


def run_with_retry(plan: dict, plan_sha: str, stage_root: Path, run: dict, launcher: bytes | None, events: Path) -> dict | None:
    """Run once; on any failure re-run once on the same seed in a new directory."""
    previous = existing_attempts(stage_root, run)
    for attempt, record in previous:
        if record and record.get("integrity", {}).get("passed"):
            append_jsonl(events, {"utc": utc_now(), "event": "resume_skip_completed", "run_id": run["run_id"], "attempt": attempt})
            return record
    used = {attempt for attempt, _ in previous}
    for attempt in (1, 2):
        if attempt in used:
            continue  # an interrupted or failed attempt counts toward the single retry
        assert_gpu_idle()
        started = time.time()
        append_jsonl(events, {"utc": utc_now(), "event": "attempt_started", "run_id": run["run_id"], "attempt": attempt})
        if cell_of(plan, run)["contract"] == "fixed_tokens":
            failures, values = execute_token_attempt(plan, stage_root, run, attempt)
        else:
            failures, values = execute_fixed_time_attempt(plan, stage_root, run, attempt, launcher)
        record = write_record(plan, plan_sha, stage_root, run, attempt, failures, values, started)
        append_jsonl(events, {"utc": utc_now(), "event": "attempt_finished", "run_id": run["run_id"], "attempt": attempt, "status": record["status"], "failures": failures})
        print(json.dumps({"run_id": run["run_id"], "attempt": attempt, "status": record["status"], "failures": failures[:5]}, sort_keys=True), flush=True)
        if record["integrity"]["passed"]:
            return record
    append_jsonl(events, {"utc": utc_now(), "event": "run_reported_failed", "run_id": run["run_id"]})
    return None


def data_identity_gate(plan: dict, stage_root: Path) -> dict:
    output = stage_root / "data_identity.json"
    prior = read_json(output)
    if prior is not None:
        return prior
    gate = plan["data_identity_gate"]
    cache = Path(os.path.expanduser(gate["runtime_cache"]))
    failures = []
    files = []
    for row in gate["files"]:
        path = cache / row["path"]
        actual = sha256(path) if path.is_file() else None
        files.append({"path": row["path"], "sha256": actual, "expected": row["sha256"]})
        if actual != row["sha256"]:
            failures.append(f"data file differs or missing: {row['path']}")
    registered = {row["path"] for row in gate["files"]}
    extra = sorted(
        str(path.relative_to(cache)) for path in (cache / "data").glob("shard_*.parquet")
        if str(path.relative_to(cache)) not in registered
    ) if (cache / "data").is_dir() else []
    stream = None
    if not failures:
        # CPU-only replay of the exact training loader (scripts/verify_phase1_token_stream.py,
        # unchanged); its result is written only here, never to results/phase1 or evaluation/.
        try:
            verifier = load_module("camera_ready_token_stream", ROOT / "scripts/verify_phase1_token_stream.py")
            panel_plan = verifier.load_plan(verifier.PLAN)
            audit = verifier.static_audit(panel_plan)
            result = verifier.runtime_hash({**panel_plan, "seeds": gate["stream_check_seeds"]}, audit)
        except Exception as exc:
            failures.append(f"token-stream replay failed: {type(exc).__name__}: {exc}")
        else:
            hashes = {
                str(row["seed"]): row["regimes"]["single_microbatch"]["stream_sha256"]
                for row in result["runtime_verification"]["seeds"]
            }
            stream = {"microbatches": audit["total_microbatches"], "stream_sha256_by_seed": hashes}
            if set(hashes.values()) != {gate["stream_sha256"]}:
                failures.append(f"token stream differs from the panel: {hashes}")
            tokenizer = {Path(row["path"]).name: row["sha256"] for row in result["runtime_verification"]["tokenizer_files"]}
            if tokenizer != gate["tokenizer_sha256"]:
                failures.append(f"tokenizer differs: {tokenizer}")
    payload = {
        "format": "autopareto_camera_ready_data_identity_v1",
        "status": "PASS" if not failures else "HALT",
        "failures": failures,
        "files": files,
        "extra_training_shards_not_in_manifest": extra,
        "stream": stream,
        "recorded_utc": utc_now(),
    }
    write_json_new(output, payload)
    return payload


def e0_verdict(plan: dict, plan_sha: str, stage_root: Path, records: dict[str, dict | None], data: dict) -> dict:
    accept = plan["E0"]["acceptance"]
    reasons, comparisons = [], {}
    if data.get("status") != "PASS":
        reasons.append(f"data identity gate failed before any GPU run: {data.get('failures')}")
    for run in stage_runs(plan, "E0") if data.get("status") == "PASS" else []:
        ref = accept[run["run_id"]]
        record = records.get(run["run_id"])
        if record is None:
            reasons.append(f"{run['run_id']}: failed twice (integrity gates)")
            continue
        values = record["metrics"]
        row = {}
        for metric, spec in ref["bpb"].items():
            observed = values.get(metric)
            delta = float(observed) - spec["reference"] if finite(observed) else None
            row[metric] = {"observed": observed, "reference": spec["reference"], "delta": delta, "tolerance": spec["tolerance"]}
            if delta is None or abs(delta) > spec["tolerance"]:
                reasons.append(f"{run['run_id']}: {metric} differs by {delta} (> {spec['tolerance']})")
        if "exact_tokens" in ref:
            row["tokens"] = {"observed": values.get("tokens"), "required": ref["exact_tokens"]}
            row["updates"] = {"observed": values.get("updates"), "required": ref["exact_updates"]}
            if values.get("tokens") != ref["exact_tokens"] or values.get("updates") != ref["exact_updates"]:
                reasons.append(f"{run['run_id']}: token/update count differs")
        if "updates_range" in ref:
            low, high = ref["updates_range"]
            row["updates"] = {"observed": values.get("updates"), "allowed": [low, high], "historical": ref["historical_updates"]}
            if not isinstance(values.get("updates"), int) or not low <= values["updates"] <= high:
                reasons.append(f"{run['run_id']}: updates {values.get('updates')} outside [{low}, {high}]")
        diagnostics = {}
        for metric, reference in ref.get("diagnostics", {}).items():
            observed = values.get(metric)
            diagnostics[metric] = {
                "observed": observed,
                "reference": reference,
                "relative_difference": (float(observed) / reference - 1.0) if finite(observed) and reference else None,
            }
        row["diagnostics_not_halting"] = diagnostics
        comparisons[run["run_id"]] = row
    verdict = {
        "format": "autopareto_camera_ready_e0_verdict_v1",
        "plan_sha256": plan_sha,
        "status": "PASS" if not reasons else "HALT",
        "halt_reasons": reasons,
        "comparisons": comparisons,
        "data_identity_status": data.get("status"),
        "recorded_utc": utc_now(),
        "rule": plan["E0"]["halt_rule"],
    }
    write_json_new(stage_root / "e0_verdict.json", verdict)
    return verdict


def execute_stage(plan: dict, plan_sha: str, stage: str, *, resume: bool, launcher_override: Path | None) -> int:
    problem = uv_lock_problem(plan)
    if problem:
        print(json.dumps({"status": "refused", "reason": problem}, indent=2, sort_keys=True))
        return 2
    # Identical to Phase 1: bind the persistent data cache and checkpoint storage first.
    phase1.validate_persistent_runtime()
    checks, info = validate(plan, plan_sha, strict=True, launcher_override=launcher_override)
    if checks.failures:
        print(json.dumps({"status": "refused", "failures": checks.failures}, indent=2, sort_keys=True))
        return 2
    root_out = output_root(plan)
    stage_root = root_out / stage
    runs = stage_runs(plan, stage)
    if stage == "E1":
        verdict = read_json(root_out / "E0" / "e0_verdict.json")
        if not verdict or verdict.get("status") != "PASS" or verdict.get("plan_sha256") != plan_sha:
            print(json.dumps({"status": "refused", "reason": "E1 requires e0_verdict.json status PASS under this exact plan"}, sort_keys=True))
            return 2
    else:
        if (stage_root / "e0_verdict.json").exists():
            print(json.dumps({"status": "refused", "reason": "E0 verdict already recorded; E0 is not re-run"}, sort_keys=True))
            return 2
    existing = [name for run in runs for name in (attempt_dirname(run, 1), attempt_dirname(run, 2)) if (stage_root / "runs" / name).exists()]
    if existing and not resume:
        print(json.dumps({"status": "refused", "reason": "run directories exist; pass --resume to continue an interrupted stage (directories are never reused)", "existing": existing}, sort_keys=True))
        return 2
    launcher = None
    if any(cell_of(plan, run)["contract"] == "fixed_time" for run in runs):
        launcher, _label = resolve_fixed_time_launcher(plan, launcher_override)
        if launcher is None:
            raise RuntimeError("historical fixed-time launcher unavailable")
    stage_root.mkdir(parents=True, exist_ok=True)
    events = stage_root / "stage_events.jsonl"
    lock_path = root_out / "gpu.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "refused", "reason": f"GPU lock held: {lock_path}"}, sort_keys=True))
            return 2
        write_json_new(stage_root / f"stage_manifest-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", {
            "format": "autopareto_camera_ready_stage_manifest_v1",
            "stage": stage,
            "plan_sha256": plan_sha,
            "resume": resume,
            "environment": info,
            "checks": checks.rows,
            "started_utc": utc_now(),
        })
        append_jsonl(events, {"utc": utc_now(), "event": "stage_started", "stage": stage, "plan_sha256": plan_sha, "resume": resume})
        data = {}
        if stage == "E0":
            data = data_identity_gate(plan, stage_root)
            if data.get("status") != "PASS":
                verdict = e0_verdict(plan, plan_sha, stage_root, {}, data)
                print(json.dumps({"status": verdict["status"], "halt_reasons": verdict["halt_reasons"]}, indent=2, sort_keys=True))
                return 3
        records: dict[str, dict | None] = {}
        for run in runs:
            records[run["run_id"]] = run_with_retry(plan, plan_sha, stage_root, run, launcher, events)
        append_jsonl(events, {"utc": utc_now(), "event": "stage_finished", "stage": stage})
    if stage == "E0":
        verdict = e0_verdict(plan, plan_sha, stage_root, records, data)
        print(json.dumps({"status": verdict["status"], "halt_reasons": verdict["halt_reasons"], "comparisons": verdict["comparisons"]}, indent=2, sort_keys=True))
        return 0 if verdict["status"] == "PASS" else 3
    summary = {
        "format": "autopareto_camera_ready_e1_execution_summary_v1",
        "plan_sha256": plan_sha,
        "completed": sorted(k for k, v in records.items() if v),
        "reported_failed": sorted(k for k, v in records.items() if not v),
        "recorded_utc": utc_now(),
        "next": f"python scripts/analyze_camera_ready_e1.py --runs-root {stage_root}",
    }
    write_json_new(stage_root / f"e1_execution_summary-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate-only", action="store_true", help="default: verify and print every command; run nothing")
    parser.add_argument("--stage", choices=("E0", "E1"))
    parser.add_argument("--execute", action="store_true", help="required for GPU work (pod only)")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted stage in new attempt directories")
    parser.add_argument("--expect-plan-sha256", help="refuse unless the plan file has exactly this sha256")
    parser.add_argument("--fixed-time-launcher", type=Path, help="explicit path to the historical launcher (sha256-verified)")
    parser.add_argument("--json", action="store_true", help="print the validation report as JSON")
    args = parser.parse_args()

    plan, plan_sha = load_plan()
    if args.expect_plan_sha256 and args.expect_plan_sha256 != plan_sha:
        print(json.dumps({"status": "refused", "reason": "plan sha256 mismatch", "plan_sha256": plan_sha}, sort_keys=True))
        return 2
    if args.execute and not args.validate_only:
        if args.stage is None:
            parser.error("--execute requires --stage E0 or --stage E1")
        return execute_stage(plan, plan_sha, args.stage, resume=args.resume, launcher_override=args.fixed_time_launcher)

    checks, info = validate(plan, plan_sha, strict=False, launcher_override=args.fixed_time_launcher)
    root_out = output_root(plan)
    stages = [args.stage] if args.stage else ["E0", "E1"]
    commands = {}
    for stage in stages:
        for run in stage_runs(plan, stage):
            commands[run["run_id"]] = {
                "stage": stage,
                "order": run["order"],
                "recipe": run["recipe"],
                "cell": run["cell"],
                "seed": run["seed"],
                "commands": {k: render(v) for k, v in run_commands(plan, root_out / stage, run, 1).items()},
                "retry_directory": attempt_dirname(run, 2),
            }
    report = {
        "status": "validated" if not checks.failures else "FAILED",
        "plan": str(PLAN_PATH),
        "plan_sha256": plan_sha,
        "checks": checks.rows,
        "environment": info,
        "commands": commands,
        "message": "No GPU work launched. On the pod: --stage E0 --execute, then --stage E1 --execute only after E0 PASS.",
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"plan: {PLAN_PATH.relative_to(ROOT)}  sha256: {plan_sha}")
        width = max(len(row["check"]) for row in checks.rows)
        for row in checks.rows:
            detail = row["detail"] if isinstance(row["detail"], str) else json.dumps(row["detail"], sort_keys=True)
            print(f"[{row['status']:>8}] {row['check']:<{width}}  {detail}")
        probe = info["training_interpreter"]
        print(f"\ncommand_prefix(): {' '.join(info['command_prefix'])}   training interpreter probed via "
              f"{probe.get('probed_with')}: python {probe.get('python')}, torch {probe.get('torch')}")
        for run_id, row in commands.items():
            print(f"\n# {row['stage']} #{row['order']:02d} {run_id}  ({row['recipe']}, {row['cell']}, seed {row['seed']}; retry dir {row['retry_directory']})")
            for name, line in row["commands"].items():
                print(f"  {name}: {line}")
        counts = {status: sum(1 for row in checks.rows if row["status"] == status) for status in (OK, FAIL, POD_ONLY, INFO)}
        print(f"\nsummary: {counts}  -> {report['status']}")
    return 0 if not checks.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

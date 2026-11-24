#!/usr/bin/env python3
"""Authoritative, resumable experiment runner for the shared A40 benchmark."""

from __future__ import annotations

import argparse
import contextlib
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from typing import Any

from autopareto_db import (
    CACHED_MODE,
    confirmed_frontier,
    migrate as migrate_scientific_schema,
    quality_budget_report,
    raw_frontier,
    record_event,
)
from benchmark_config import CUDA_ALLOCATOR, PROTECTED_FILES, TRAINING_TIMEOUT_SECONDS
from event_log import read_events

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DB_PATH = RESULTS / "results.db"
LATEST_PATH = RESULTS / "latest_result.json"
PARETO_PATH = RESULTS / "pareto_frontier.json"
CONFIRMED_PARETO_PATH = RESULTS / "confirmed_pareto_frontier.json"
QUALITY_BUDGET_PATH = RESULTS / "quality_budgets.json"
LOCK_PATH = RESULTS / "gpu.lock"
WORKTREES = ROOT / "worktrees"
ORIGINAL_AUTORESEARCH_COMMIT = "228791fb499afffb54b46200aca536f79142f117"
SHARED_A40_BASELINE_COMMIT = "5ea452c9d86a43ad4bebd507c5872f18249627ce"
TERMINAL_STATUSES = {"completed", "crash", "oom", "timeout", "protection_failure", "interrupted"}
SEARCH_METHODS = {"random_search", "tpe", "cmaes", "motpe", "nsgaii"}
CONTROLLED_METHODS = {"autoresearch", "autopareto", *SEARCH_METHODS}


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT NOT NULL,
    experiment_number INTEGER NOT NULL,
    method TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'campaign',
    training_seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    valid_result INTEGER NOT NULL DEFAULT 0,
    exclude_from_analysis INTEGER NOT NULL DEFAULT 1,
    description TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    code_diff TEXT NOT NULL DEFAULT '',
    val_bpb REAL,
    inference_tokens_per_second REAL,
    peak_inference_cuda_memory_mb REAL,
    training_tokens_per_second REAL,
    total_training_tokens INTEGER,
    peak_training_cuda_memory_mb REAL,
    parameter_count INTEGER,
    training_seconds REAL,
    inference_seconds REAL,
    total_runtime_seconds REAL,
    checkpoint_path TEXT,
    sample_path TEXT,
    log_path TEXT,
    crash_or_failure_reason TEXT,
    protected_manifest_sha256 TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    UNIQUE(campaign_id, experiment_number)
);
CREATE INDEX IF NOT EXISTS idx_experiments_campaign ON experiments(campaign_id, experiment_number);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

CREATE TABLE IF NOT EXISTS campaign_state (
    campaign_id TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'campaign',
    attempt_limit INTEGER NOT NULL,
    training_seed INTEGER,
    algorithm_seed INTEGER,
    coding_model_name TEXT,
    reasoning_level TEXT,
    program_sha256 TEXT,
    search_space_sha256 TEXT,
    baseline_experiment_id INTEGER REFERENCES experiments(id),
    next_experiment_number INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'ready',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect() -> sqlite3.Connection:
    RESULTS.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript(SCHEMA)
    columns = {row["name"] for row in db.execute("PRAGMA table_info(experiments)")}
    migrations = {
        "phase": "ALTER TABLE experiments ADD COLUMN phase TEXT NOT NULL DEFAULT 'campaign'",
        "valid_result": "ALTER TABLE experiments ADD COLUMN valid_result INTEGER NOT NULL DEFAULT 0",
        "exclude_from_analysis": "ALTER TABLE experiments ADD COLUMN exclude_from_analysis INTEGER NOT NULL DEFAULT 1",
    }
    for name, statement in migrations.items():
        if name not in columns:
            db.execute(statement)
    campaign_state_columns = {row["name"] for row in db.execute("PRAGMA table_info(campaign_state)")}
    campaign_state_migrations = {
        "phase": "ALTER TABLE campaign_state ADD COLUMN phase TEXT NOT NULL DEFAULT 'campaign'",
        "training_seed": "ALTER TABLE campaign_state ADD COLUMN training_seed INTEGER",
        "algorithm_seed": "ALTER TABLE campaign_state ADD COLUMN algorithm_seed INTEGER",
        "coding_model_name": "ALTER TABLE campaign_state ADD COLUMN coding_model_name TEXT",
        "reasoning_level": "ALTER TABLE campaign_state ADD COLUMN reasoning_level TEXT",
        "program_sha256": "ALTER TABLE campaign_state ADD COLUMN program_sha256 TEXT",
        "search_space_sha256": "ALTER TABLE campaign_state ADD COLUMN search_space_sha256 TEXT",
        "baseline_experiment_id": "ALTER TABLE campaign_state ADD COLUMN baseline_experiment_id INTEGER REFERENCES experiments(id)",
    }
    for name, statement in campaign_state_migrations.items():
        if name not in campaign_state_columns:
            db.execute(statement)
    migrate_scientific_schema(db)
    db.commit()
    return db


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_manifest(root: Path = ROOT) -> dict[str, str]:
    manifest = {}
    for relative in PROTECTED_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"protected file missing: {relative}")
        manifest[relative] = sha256(path)
    return manifest


def manifest_digest(manifest: dict[str, str]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_isolate(candidate_root: Path, canonical_manifest: dict[str, str]) -> None:
    isolated = protected_manifest(candidate_root)
    if isolated != canonical_manifest:
        changed = sorted(key for key in canonical_manifest if isolated.get(key) != canonical_manifest[key])
        raise RuntimeError(f"candidate modified protected files: {', '.join(changed)}")


def immutable_source_manifest(root: Path) -> dict[str, str]:
    """Hash every candidate-tree file except train.py and runtime-generated paths."""
    ignored_parts = {".git", ".venv", "__pycache__", "results", "worktrees", "queue"}
    manifest = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "train.py" or any(part in ignored_parts for part in path.parts):
            continue
        if path.suffix in {".pyc", ".log"}:
            continue
        manifest[str(path.relative_to(root))] = sha256(path)
    return manifest


def current_git_commit() -> str:
    override = os.environ.get("AUTOPARETO_GIT_COMMIT")
    if override:
        return override.strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ORIGINAL_AUTORESEARCH_COMMIT


def candidate_git_commit(method: str) -> str:
    """Return candidate provenance without conflating it with evaluator provenance."""
    override = os.environ.get("AUTOPARETO_CANDIDATE_GIT_COMMIT")
    if override:
        return override.strip()
    if method == "baseline":
        return SHARED_A40_BASELINE_COMMIT
    return current_git_commit()


def code_diff(candidate: Path) -> str:
    baseline = (ROOT / "train.py").read_text().splitlines(keepends=True)
    proposed = candidate.read_text().splitlines(keepends=True)
    return "".join(difflib.unified_diff(baseline, proposed, fromfile="original/train.py", tofile="candidate/train.py"))


def register_campaign(
    db: sqlite3.Connection,
    *,
    campaign_id: str,
    method: str,
    phase: str,
    attempt_limit: int,
    training_seed: int,
    algorithm_seed: int | None = None,
    program_path: Path | None = None,
    search_space_sha256: str | None = None,
    coding_model: str | None = None,
    reasoning_level: str | None = None,
    baseline_experiment_id: int | None = None,
) -> sqlite3.Row:
    """Create or verify one immutable campaign registration before any attempt."""
    if attempt_limit < 1:
        raise ValueError("attempt limit must be positive")
    if method in CONTROLLED_METHODS and attempt_limit != 25:
        raise ValueError(f"{method} campaigns require exactly 25 attempts")
    if method in CONTROLLED_METHODS and training_seed not in (0, 1, 42):
        raise ValueError(f"{method} training seed must be 0, 1, or 42")
    if method in SEARCH_METHODS and algorithm_seed is None:
        raise ValueError("classical campaigns require an algorithm seed")
    if method in SEARCH_METHODS and not search_space_sha256:
        raise ValueError("classical campaigns require the frozen search-space hash")
    if method in {"autoresearch", "autopareto"} and program_path is None:
        raise ValueError(f"{method} campaigns require a program snapshot")
    if method == "autopareto" and baseline_experiment_id is None:
        raise ValueError("autopareto campaigns require one registered baseline experiment")
    if baseline_experiment_id is not None:
        baseline = db.execute(
            """SELECT id, method, training_seed, status, valid_result, exclude_from_analysis,
                      inference_metric_mode
               FROM experiments WHERE id=?""",
            (baseline_experiment_id,),
        ).fetchone()
        if baseline is None:
            raise ValueError(f"registered baseline does not exist: {baseline_experiment_id}")
        if (
            baseline["method"] != "baseline"
            or int(baseline["training_seed"]) != training_seed
            or baseline["status"] != "completed"
            or not baseline["valid_result"]
            or baseline["exclude_from_analysis"]
            or baseline["inference_metric_mode"] != CACHED_MODE
        ):
            raise ValueError(
                "registered baseline must be a valid compatible baseline with the campaign seed"
            )
    registered_training_seed = (
        training_seed if phase == "campaign" and method not in {"confirmation", "ablation"} else None
    )
    program_sha = sha256(program_path) if program_path else None
    program_text = program_path.read_text() if program_path else None
    expected = {
        "method": method,
        "phase": phase,
        "attempt_limit": attempt_limit,
        "training_seed": registered_training_seed,
        "algorithm_seed": algorithm_seed,
        "coding_model_name": coding_model,
        "reasoning_level": reasoning_level,
        "program_sha256": program_sha,
        "search_space_sha256": search_space_sha256,
        "baseline_experiment_id": baseline_experiment_id,
    }
    existing = db.execute(
        "SELECT * FROM campaign_state WHERE campaign_id=?", (campaign_id,)
    ).fetchone()
    if existing is not None:
        mismatches = {
            key: (existing[key], value)
            for key, value in expected.items()
            if existing[key] is not None and existing[key] != value
        }
        if mismatches:
            rendered = ", ".join(
                f"{key}: registered={old!r}, requested={new!r}"
                for key, (old, new) in sorted(mismatches.items())
            )
            raise ValueError(f"campaign registration mismatch for {campaign_id}: {rendered}")
        db.execute(
            """UPDATE campaign_state SET phase=COALESCE(phase, ?),
               training_seed=COALESCE(training_seed, ?),
               algorithm_seed=COALESCE(algorithm_seed, ?),
               coding_model_name=COALESCE(coding_model_name, ?),
               reasoning_level=COALESCE(reasoning_level, ?),
               program_sha256=COALESCE(program_sha256, ?),
               search_space_sha256=COALESCE(search_space_sha256, ?),
               baseline_experiment_id=COALESCE(baseline_experiment_id, ?),
               updated_at=CURRENT_TIMESTAMP WHERE campaign_id=?""",
            (
                phase, registered_training_seed, algorithm_seed, coding_model,
                reasoning_level, program_sha,
                search_space_sha256, baseline_experiment_id, campaign_id,
            ),
        )
    else:
        db.execute(
            """INSERT INTO campaign_state
               (campaign_id, method, phase, attempt_limit, training_seed, algorithm_seed,
                coding_model_name, reasoning_level, program_sha256, search_space_sha256,
                baseline_experiment_id, next_experiment_number, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'ready')""",
            (
                campaign_id, method, phase, attempt_limit, registered_training_seed, algorithm_seed,
                coding_model, reasoning_level, program_sha, search_space_sha256,
                baseline_experiment_id,
            ),
        )
    db.execute(
        """INSERT INTO campaigns
           (campaign_id, method, phase, attempt_limit, training_seed, algorithm_seed,
            search_space_sha256, starting_git_commit, coding_model_name, reasoning_level,
            program_path, program_sha256, program_instructions, baseline_experiment_id, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready')
           ON CONFLICT(campaign_id) DO UPDATE SET
             coding_model_name=COALESCE(campaigns.coding_model_name, excluded.coding_model_name),
             reasoning_level=COALESCE(campaigns.reasoning_level, excluded.reasoning_level),
             program_path=COALESCE(campaigns.program_path, excluded.program_path),
             program_sha256=COALESCE(campaigns.program_sha256, excluded.program_sha256),
             program_instructions=COALESCE(campaigns.program_instructions, excluded.program_instructions),
             baseline_experiment_id=COALESCE(campaigns.baseline_experiment_id, excluded.baseline_experiment_id),
             updated_at=CURRENT_TIMESTAMP""",
        (
            campaign_id, method, phase, attempt_limit, registered_training_seed, algorithm_seed,
            search_space_sha256, candidate_git_commit(method), coding_model,
            reasoning_level, str(program_path) if program_path else None, program_sha,
            program_text, baseline_experiment_id,
        ),
    )
    db.commit()
    return db.execute(
        "SELECT * FROM campaign_state WHERE campaign_id=?", (campaign_id,)
    ).fetchone()


def copy_isolate(destination: Path) -> None:
    ignored = shutil.ignore_patterns(
        ".git", ".venv", "__pycache__", "*.pyc", "results", "worktrees", "queue", "*.log"
    )
    shutil.copytree(ROOT, destination, ignore=ignored)


def tail_failure(log_path: Path, limit: int = 6000) -> str:
    if not log_path.exists():
        return "process failed without a log"
    text = log_path.read_text(errors="replace")
    return text[-limit:].strip()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def refresh_exports(db: sqlite3.Connection) -> None:
    latest = db.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT 1").fetchone()
    write_json_atomic(LATEST_PATH, row_to_dict(latest) if latest else {})
    objectives = {
        "val_bpb": "min",
        "median_cached_decode_tokens_per_second": "max",
        "peak_inference_cuda_memory_mb": "min",
    }
    write_json_atomic(PARETO_PATH, {
        "frontier_kind": "raw_single_seed",
        "inference_metric_mode": CACHED_MODE,
        "objectives": objectives,
        "experiments": raw_frontier(db),
    })
    write_json_atomic(CONFIRMED_PARETO_PATH, {
        "frontier_kind": "confirmed_seeds_0_through_2",
        "inference_metric_mode": CACHED_MODE,
        "objectives": objectives,
        "experiments": confirmed_frontier(db),
    })
    write_json_atomic(QUALITY_BUDGET_PATH, quality_budget_report(db))


@contextlib.contextmanager
def gpu_lock():
    RESULTS.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lock.write(f"pid={os.getpid()} acquired={time.time()}\n")
        lock.flush()
        yield
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def reserve_experiment(
    db: sqlite3.Connection,
    campaign_id: str,
    number: int,
    method: str,
    phase: str,
    seed: int,
    description: str,
    diff: str,
    manifest_sha: str,
    attempt_limit: int,
) -> tuple[int, str | None]:
    registration = db.execute(
        "SELECT * FROM campaign_state WHERE campaign_id=?", (campaign_id,)
    ).fetchone()
    if registration is None:
        raise ValueError(f"campaign is not registered: {campaign_id}")
    if registration["method"] != method or int(registration["attempt_limit"]) != attempt_limit:
        raise ValueError("runner arguments do not match immutable campaign registration")
    if registration["status"] == "paused":
        raise ValueError(
            f"campaign {campaign_id} is paused after an inference or measurement failure; "
            "resolve and explicitly resume it before reserving another experiment"
        )
    existing = db.execute(
        "SELECT id, status FROM experiments WHERE campaign_id=? AND experiment_number=?",
        (campaign_id, number),
    ).fetchone()
    if existing and existing["status"] in TERMINAL_STATUSES:
        db.commit()
        return int(existing["id"]), str(existing["status"])
    if existing:
        db.execute(
            """UPDATE experiments SET status='interrupted', finished_at=CURRENT_TIMESTAMP,
               crash_or_failure_reason='previous runner stopped before completion' WHERE id=?""",
            (existing["id"],),
        )
        db.execute(
            "UPDATE campaign_state SET next_experiment_number=?, updated_at=CURRENT_TIMESTAMP WHERE campaign_id=?",
            (number + 1, campaign_id),
        )
        db.commit()
        return int(existing["id"]), "interrupted"
    expected_number = int(registration["next_experiment_number"])
    if number != expected_number:
        raise ValueError(
            f"campaign {campaign_id} requires experiment {expected_number}, got {number}"
        )
    if number > attempt_limit:
        raise ValueError(f"campaign {campaign_id} is complete at {attempt_limit} attempts")
    db.execute(
        """UPDATE campaign_state SET status='running', updated_at=CURRENT_TIMESTAMP
           WHERE campaign_id=?""",
        (campaign_id,),
    )
    cursor = db.execute(
        """INSERT INTO experiments
           (campaign_id, experiment_number, method, phase, training_seed, status,
            valid_result, exclude_from_analysis, description, git_commit, code_diff,
            protected_manifest_sha256)
           VALUES (?, ?, ?, ?, ?, 'running', 0, 1, ?, ?, ?, ?)""",
        (campaign_id, number, method, phase, seed, description, candidate_git_commit(method), diff, manifest_sha),
    )
    db.commit()
    return int(cursor.lastrowid), None


def attach_experiment_metadata(db: sqlite3.Connection, experiment_id: int, args: argparse.Namespace) -> int:
    """Store visible agent metadata and ensure a normalized training-run record exists."""
    parent_id = getattr(args, "parent_experiment_id", None)
    if parent_id is not None:
        parent = db.execute(
            "SELECT id, campaign_id, method FROM experiments WHERE id=?", (parent_id,)
        ).fetchone()
        if parent is None:
            raise ValueError(f"parent experiment does not exist: {parent_id}")
        if args.method in {"autoresearch", "autopareto"} and (
            parent["campaign_id"] != args.campaign_id and parent["method"] != "baseline"
        ):
            raise ValueError(
                f"{args.method} parent must belong to its own campaign or be a baseline"
            )
    program_path = getattr(args, "program", None)
    program_sha = sha256(Path(program_path)) if program_path else None
    db.execute(
        """UPDATE experiments SET hypothesis=?, intended_change=?, change_category=?,
           parent_experiment_id=?, agent_interpretation=?, coding_model_name=?,
           reasoning_level=?, program_sha256=? WHERE id=?""",
        (
            getattr(args, "hypothesis", None), getattr(args, "intended_change", None),
            getattr(args, "change_category", None), parent_id,
            getattr(args, "agent_interpretation", None), getattr(args, "coding_model", None),
            getattr(args, "reasoning_level", None), program_sha, experiment_id,
        ),
    )
    db.execute(
        """INSERT INTO campaigns
           (campaign_id, method, phase, attempt_limit, training_seed, algorithm_seed,
            search_space_sha256, starting_git_commit, coding_model_name,
            reasoning_level, program_path, program_sha256, program_instructions,
            baseline_experiment_id, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
           ON CONFLICT(campaign_id) DO UPDATE SET coding_model_name=excluded.coding_model_name,
              reasoning_level=excluded.reasoning_level, program_path=excluded.program_path,
              program_sha256=excluded.program_sha256, program_instructions=excluded.program_instructions,
              baseline_experiment_id=excluded.baseline_experiment_id,
              status='running', updated_at=CURRENT_TIMESTAMP""",
        (
            args.campaign_id, args.method, args.phase,
            getattr(args, "attempt_limit", 25),
            args.seed, getattr(args, "algorithm_seed", None),
            getattr(args, "search_space_sha256", None),
            candidate_git_commit(args.method), getattr(args, "coding_model", None),
            getattr(args, "reasoning_level", None), program_path, program_sha,
            Path(program_path).read_text() if program_path else None,
            getattr(args, "baseline_experiment_id", None),
        ),
    )
    run_kind = args.phase if args.phase in {"setup", "baseline", "confirmation", "ablation"} else "discovery"
    db.execute(
        """INSERT INTO training_runs (experiment_id, run_kind, seed, status, started_at)
           VALUES (?, ?, ?, 'running', CURRENT_TIMESTAMP)
           ON CONFLICT(experiment_id, run_kind, seed) DO UPDATE SET status='running'
        """,
        (experiment_id, run_kind, args.seed),
    )
    training_run_id = int(db.execute(
        "SELECT id FROM training_runs WHERE experiment_id=? AND run_kind=? AND seed=?",
        (experiment_id, run_kind, args.seed),
    ).fetchone()["id"])
    if args.phase == "confirmation":
        if parent_id is None or args.seed not in (0, 1, 2):
            raise ValueError("confirmation requires a candidate parent and seed in 0..2")
        db.execute(
            """INSERT OR REPLACE INTO confirmation_runs
               (candidate_experiment_id, training_run_id, seed, selection_role, status)
               VALUES (?, ?, ?, ?, 'running')""",
            (parent_id, training_run_id, args.seed, getattr(args, "selection_role", None)),
        )
    if args.phase == "ablation":
        if parent_id is None or not getattr(args, "ablation_name", None):
            raise ValueError("ablation requires a source parent and --ablation-name")
        db.execute(
            """INSERT OR REPLACE INTO ablation_runs
               (source_experiment_id, training_run_id, ablation_name, isolated_change,
                hypothesis, status) VALUES (?, ?, ?, ?, ?, 'running')""",
            (parent_id, training_run_id, args.ablation_name,
             getattr(args, "intended_change", None) or args.description,
             getattr(args, "hypothesis", None)),
        )
    db.commit()
    return training_run_id


def import_event_log(db: sqlite3.Connection, path: Path) -> None:
    for event in read_events(path):
        record_event(db, **event)


def store_inference_repetitions(
    db: sqlite3.Connection,
    experiment_id: int,
    training_run_id: int,
    metrics: dict[str, Any],
) -> None:
    repetitions = [
        *metrics.get("warmup_repetitions", []),
        *metrics.get("inference_repetitions", []),
    ]
    for repetition in repetitions:
        is_warmup = 1 if repetition.get("is_warmup") else 0
        correctness = repetition.get("correctness_passed")
        db.execute(
            """INSERT OR REPLACE INTO inference_measurements
               (experiment_id, training_run_id, repetition_index, is_warmup, metric_mode,
                input_tokens, generated_tokens, inference_batch_size, workload_name,
                prefill_milliseconds, prefill_tokens_per_second, decode_seconds,
                decode_tokens_per_second, peak_allocated_cuda_memory_mb,
                peak_reserved_cuda_memory_mb, prefill_kv_cache_memory_mb,
                incremental_kv_cache_memory_mb, correctness_passed,
                max_logit_absolute_error, evaluator_git_commit, evaluator_sha256, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id, training_run_id, repetition["repetition_index"], is_warmup,
                repetition["metric_mode"], metrics.get("input_tokens"),
                metrics.get("generated_tokens"), 1, "headline",
                repetition.get("prefill_milliseconds"),
                repetition.get("prefill_tokens_per_second"), repetition.get("decode_seconds"),
                repetition.get("decode_tokens_per_second"), repetition.get("peak_allocated_cuda_memory_mb"),
                repetition.get("peak_reserved_cuda_memory_mb"), repetition.get("prefill_kv_cache_memory_mb"),
                repetition.get("incremental_kv_cache_memory_mb"),
                None if correctness is None else int(bool(correctness)),
                repetition.get("max_logit_absolute_error"), metrics.get("evaluator_git_commit"),
                metrics.get("evaluator_sha256"), json.dumps(repetition, sort_keys=True),
            ),
        )
    db.commit()


def persist_successful_training(
    db: sqlite3.Connection,
    experiment_id: int,
    training_run_id: int,
    metrics: dict[str, Any],
    checkpoint: Path,
    candidate_source: Path,
    candidate_archive: Path,
) -> None:
    """Persist validated training output before post-training inference begins."""
    required = {
        "val_bpb", "training_tokens_per_second", "total_training_tokens",
        "peak_training_cuda_memory_mb", "parameter_count", "training_seconds",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        raise RuntimeError(f"successful training omitted required metrics: {', '.join(missing)}")
    candidate_archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_source, candidate_archive)
    values = {
        "id": experiment_id,
        "val_bpb": metrics["val_bpb"],
        "training_tokens_per_second": metrics["training_tokens_per_second"],
        "total_training_tokens": metrics["total_training_tokens"],
        "peak_training_cuda_memory_mb": metrics["peak_training_cuda_memory_mb"],
        "parameter_count": metrics["parameter_count"],
        "training_seconds": metrics["training_seconds"],
        "checkpoint_path": str(checkpoint),
    }
    db.execute(
        """UPDATE experiments SET val_bpb=:val_bpb, training_status='completed',
           training_tokens_per_second=:training_tokens_per_second,
           total_training_tokens=:total_training_tokens,
           peak_training_cuda_memory_mb=:peak_training_cuda_memory_mb,
           parameter_count=:parameter_count, training_seconds=:training_seconds,
           checkpoint_path=:checkpoint_path, inference_status='pending'
           WHERE id=:id""",
        values,
    )
    db.execute(
        """UPDATE training_runs SET status='completed', val_bpb=:val_bpb,
           training_tokens_per_second=:training_tokens_per_second,
           total_training_tokens=:total_training_tokens,
           peak_training_cuda_memory_mb=:peak_training_cuda_memory_mb,
           parameter_count=:parameter_count, training_seconds=:training_seconds,
           checkpoint_path=:checkpoint_path, failure_reason=NULL,
           finished_at=CURRENT_TIMESTAMP WHERE id=:training_run_id""",
        {**values, "training_run_id": training_run_id},
    )
    db.commit()


def inference_failure_status(returncode: int, reason: str) -> str:
    lowered = reason.lower()
    if returncode == 124:
        return "timeout"
    if "out of memory" in lowered or "outofmemory" in lowered:
        return "oom"
    if any(marker in lowered for marker in ("correctness", "token mismatch", "logit")):
        return "incompatible"
    return "failed"


def autopareto_scope_rows(
    db: sqlite3.Connection,
    campaign_id: str,
) -> list[sqlite3.Row]:
    """Return only one campaign and its explicitly registered same-seed baseline."""
    registration = db.execute(
        """SELECT method, training_seed, baseline_experiment_id
           FROM campaign_state WHERE campaign_id=?""",
        (campaign_id,),
    ).fetchone()
    if registration is None or registration["method"] != "autopareto":
        raise ValueError(f"AutoPareto campaign is not registered: {campaign_id}")
    baseline_id = registration["baseline_experiment_id"]
    if baseline_id is None:
        raise ValueError("AutoPareto campaign has no registered baseline")
    return db.execute(
        """SELECT * FROM experiments
           WHERE (campaign_id=? OR id=?)
             AND training_seed=?
             AND status='completed' AND valid_result=1 AND exclude_from_analysis=0
             AND inference_metric_mode=? AND val_bpb IS NOT NULL
             AND median_cached_decode_tokens_per_second IS NOT NULL
             AND peak_inference_cuda_memory_mb IS NOT NULL""",
        (
            campaign_id, int(baseline_id), int(registration["training_seed"]),
            CACHED_MODE,
        ),
    ).fetchall()


def autopareto_frontier_ids(
    db: sqlite3.Connection,
    campaign_id: str,
) -> set[int]:
    candidates = [
        {
            "id": int(row["id"]),
            "val_bpb": float(row["val_bpb"]),
            "speed": float(row["median_cached_decode_tokens_per_second"]),
            "memory": float(row["peak_inference_cuda_memory_mb"]),
        }
        for row in autopareto_scope_rows(db, campaign_id)
    ]
    return {int(row["id"]) for row in raw_pareto_rows(candidates)}


def validate_autopareto_proposal(
    db: sqlite3.Connection,
    *,
    campaign_id: str,
    experiment_number: int,
    parent_experiment_id: int,
) -> None:
    """Reject wrong-seed, dominated, cross-campaign, or non-sequential proposals."""
    if experiment_number > 1:
        previous = db.execute(
            """SELECT status, decision FROM experiments
               WHERE campaign_id=? AND experiment_number=?""",
            (campaign_id, experiment_number - 1),
        ).fetchone()
        if previous is None or previous["status"] not in TERMINAL_STATUSES:
            raise ValueError("previous AutoPareto attempt must be terminal before the next proposal")
        if previous["decision"] not in {"retained", "dominated"}:
            raise ValueError("previous AutoPareto attempt requires a recorded Pareto decision")
    if parent_experiment_id not in autopareto_frontier_ids(db, campaign_id):
        raise ValueError(
            "AutoPareto parent must be on the current same-seed campaign frontier"
        )


def run_process(command: list[str], cwd: Path, log_path: Path, timeout: int) -> tuple[int, str | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        try:
            environment = os.environ.copy()
            # Fixed for every training and inference subprocess. This changes only the
            # allocator backend, not model code, tensors, precision, or numerics.
            environment["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOCATOR
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                start_new_session=True,
            )
            return completed.returncode, None
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout} seconds"


def execute_experiment(args: argparse.Namespace) -> dict[str, Any]:
    attempt_limit = getattr(args, "attempt_limit", 25)
    if args.method in {"autoresearch", "autopareto"}:
        if args.seed not in (0, 1, 42):
            raise ValueError("controlled discovery seeds must be one of 0, 1, or 42")
        if not 1 <= args.experiment_number <= attempt_limit:
            raise ValueError(
                f"controlled campaign experiment number must be in 1..{attempt_limit}"
            )
        required_metadata = {
            "hypothesis": getattr(args, "hypothesis", None),
            "intended_change": getattr(args, "intended_change", None),
            "change_category": getattr(args, "change_category", None),
            "parent_experiment_id": getattr(args, "parent_experiment_id", None),
            "program": getattr(args, "program", None),
        }
        missing = [key for key, value in required_metadata.items() if value in (None, "")]
        if missing:
            raise ValueError(f"controlled campaign omitted required metadata: {', '.join(missing)}")
    if args.phase == "confirmation" and (
        getattr(args, "parent_experiment_id", None) is None or args.seed not in (0, 1, 2)
    ):
        raise ValueError("confirmation requires --parent-experiment-id and seed 0..2")
    if args.phase == "ablation" and (
        getattr(args, "parent_experiment_id", None) is None or not getattr(args, "ablation_name", None)
    ):
        raise ValueError("ablation requires --parent-experiment-id and --ablation-name")
    if getattr(args, "program", None) and not Path(args.program).is_file():
        raise FileNotFoundError(args.program)
    candidate_source = Path(args.candidate).resolve() if args.candidate else ROOT / "train.py"
    if not candidate_source.is_file():
        raise FileNotFoundError(candidate_source)
    # Resolve and create a dangling checkpoint-symlink target before reserving an
    # experiment. Storage setup faults must never consume a scientific attempt.
    checkpoint_root = RESULTS / "checkpoints"
    checkpoint_target = checkpoint_root.resolve(strict=False)
    checkpoint_target.mkdir(parents=True, exist_ok=True)
    if not checkpoint_target.is_dir():
        raise RuntimeError(f"checkpoint storage is not a directory: {checkpoint_target}")
    canonical_manifest = protected_manifest()
    protected_sha = manifest_digest(canonical_manifest)
    diff = code_diff(candidate_source)
    db = connect()
    program_path = Path(args.program) if getattr(args, "program", None) else None
    registration = register_campaign(
        db,
        campaign_id=args.campaign_id,
        method=args.method,
        phase=args.phase,
        attempt_limit=attempt_limit,
        training_seed=args.seed,
        algorithm_seed=getattr(args, "algorithm_seed", None),
        program_path=program_path,
        search_space_sha256=getattr(args, "search_space_sha256", None),
        coding_model=getattr(args, "coding_model", None),
        reasoning_level=getattr(args, "reasoning_level", None),
        baseline_experiment_id=getattr(args, "baseline_experiment_id", None),
    )
    attempt_limit = int(registration["attempt_limit"])
    if registration["training_seed"] is not None and int(registration["training_seed"]) != args.seed:
        raise ValueError("runner seed does not match immutable campaign registration")
    if args.method == "autopareto":
        validate_autopareto_proposal(
            db,
            campaign_id=args.campaign_id,
            experiment_number=args.experiment_number,
            parent_experiment_id=int(args.parent_experiment_id),
        )
    if getattr(args, "parent_experiment_id", None) is not None and db.execute(
        "SELECT id FROM experiments WHERE id=?", (args.parent_experiment_id,)
    ).fetchone() is None:
        raise ValueError(f"parent experiment does not exist: {args.parent_experiment_id}")
    record_id, prior_status = reserve_experiment(
        db, args.campaign_id, args.experiment_number, args.method, args.phase, args.seed,
        args.description, diff, protected_sha, attempt_limit,
    )
    if prior_status:
        row = db.execute("SELECT * FROM experiments WHERE id=?", (record_id,)).fetchone()
        refresh_exports(db)
        return row_to_dict(row)
    training_run_id = attach_experiment_metadata(db, record_id, args)

    slug = f"{args.campaign_id}-{args.experiment_number:04d}"
    isolate = WORKTREES / slug
    log_path = RESULTS / "logs" / f"{slug}.log"
    inference_log = RESULTS / "logs" / f"{slug}-inference.log"
    checkpoint = RESULTS / "checkpoints" / args.campaign_id / f"{args.experiment_number:04d}.pt"
    train_metrics_path = RESULTS / "tmp" / f"{slug}-train.json"
    inference_metrics_path = RESULTS / "tmp" / f"{slug}-inference.json"
    sample_path = RESULTS / "samples" / args.campaign_id / f"{args.experiment_number:04d}.txt"
    candidate_archive = RESULTS / "candidates" / args.campaign_id / f"{args.experiment_number:04d}-train.py"
    event_path = RESULTS / "events" / f"{slug}.jsonl"
    started = time.time()
    status = "crash"
    failure_reason = None
    training_status = "running"
    training_failure_reason = None
    inference_status = "not_started"
    values: dict[str, Any] = {}
    record_event(
        db, event_uuid=f"{slug}-proposed", event_type="experiment_proposed",
        campaign_id=args.campaign_id, experiment_id=record_id, training_run_id=training_run_id,
        status="pending", payload={
            "hypothesis": getattr(args, "hypothesis", None),
            "parent_experiment_id": getattr(args, "parent_experiment_id", None),
            "intended_change": getattr(args, "intended_change", None),
            "change_category": getattr(args, "change_category", None),
            "description": args.description,
        },
    )

    try:
        if isolate.exists():
            shutil.rmtree(isolate)
        isolate.parent.mkdir(parents=True, exist_ok=True)
        copy_isolate(isolate)
        shutil.copy2(candidate_source, isolate / "train.py")
        validate_isolate(isolate, canonical_manifest)
        immutable_before = immutable_source_manifest(isolate)
        if protected_manifest() != canonical_manifest:
            raise RuntimeError("canonical protected files changed before GPU execution")

        with gpu_lock():
            record_event(
                db, event_uuid=f"{slug}-training-started", event_type="training_started",
                campaign_id=args.campaign_id, experiment_id=record_id, training_run_id=training_run_id,
                status="running", elapsed_seconds=time.time() - started,
            )
            train_command = [
                sys.executable, "protected/train_launcher.py",
                "--train", "train.py", "--checkpoint", str(checkpoint),
                "--metrics", str(train_metrics_path), "--seed", str(args.seed),
                "--event-log", str(event_path), "--campaign-id", args.campaign_id,
                "--experiment-id", str(record_id), "--training-run-id", str(training_run_id),
            ]
            returncode, timeout_reason = run_process(
                train_command, isolate, log_path, int(args.timeout)
            )
            train_metrics = read_json(train_metrics_path)
            if returncode != 0:
                failure_reason = timeout_reason or train_metrics.get("failure_reason") or tail_failure(log_path)
                is_oom = "out of memory" in failure_reason.lower() or "outofmemory" in failure_reason.lower()
                status = "timeout" if returncode == 124 else ("oom" if is_oom else "crash")
                training_status = status
                training_failure_reason = failure_reason
            elif not checkpoint.exists():
                failure_reason = "training returned success without a checkpoint"
                status = "crash"
                training_status = status
                training_failure_reason = failure_reason
            else:
                values.update(train_metrics)
                persist_successful_training(
                    db, record_id, training_run_id, train_metrics, checkpoint,
                    isolate / "train.py", candidate_archive,
                )
                training_status = "completed"
                inference_status = "running"
                record_event(
                    db, event_uuid=f"{slug}-validation-completed", event_type="validation_completed",
                    campaign_id=args.campaign_id, experiment_id=record_id, training_run_id=training_run_id,
                    status="running", elapsed_seconds=time.time() - started,
                    payload={"val_bpb": train_metrics.get("val_bpb")},
                )
                inference_command = [
                    sys.executable, "inference_benchmark.py",
                    "--train", "train.py", "--checkpoint", str(checkpoint),
                    "--output", str(inference_metrics_path), "--sample", str(sample_path),
                    "--event-log", str(event_path), "--campaign-id", args.campaign_id,
                    "--experiment-id", str(record_id), "--training-run-id", str(training_run_id),
                ]
                inference_code, inference_timeout = run_process(
                    inference_command, isolate, inference_log, int(args.inference_timeout)
                )
                if inference_code != 0:
                    failure_reason = inference_timeout or tail_failure(inference_log)
                    is_oom = "out of memory" in failure_reason.lower() or "outofmemory" in failure_reason.lower()
                    status = "timeout" if inference_code == 124 else ("oom" if is_oom else "crash")
                    inference_status = inference_failure_status(inference_code, failure_reason)
                else:
                    inference_metrics = read_json(inference_metrics_path)
                    inference_metrics["evaluator_git_commit"] = (
                        inference_metrics.get("evaluator_git_commit") or current_git_commit()
                    )
                    store_inference_repetitions(db, record_id, training_run_id, inference_metrics)
                    values.update(inference_metrics)
                    required = {
                        "val_bpb", "median_cached_decode_tokens_per_second",
                        "peak_inference_cuda_memory_mb", "training_tokens_per_second",
                        "total_training_tokens", "peak_training_cuda_memory_mb",
                        "parameter_count", "training_seconds", "inference_seconds",
                    }
                    missing = sorted(required - values.keys())
                    if missing:
                        raise RuntimeError(f"successful pipeline omitted required metrics: {', '.join(missing)}")
                    status = "completed"
                    inference_status = "compatible"

        validate_isolate(isolate, canonical_manifest)
        immutable_after = immutable_source_manifest(isolate)
        if immutable_after != immutable_before:
            changed = sorted(set(immutable_before) ^ set(immutable_after) | {name for name in set(immutable_before) & set(immutable_after) if immutable_before[name] != immutable_after[name]})
            raise RuntimeError(f"candidate changed files outside train.py: {', '.join(changed)}")
        if protected_manifest() != canonical_manifest:
            raise RuntimeError("canonical protected files changed during experiment")
    except BaseException as exc:
        status = "protection_failure" if "protected" in str(exc).lower() else "crash"
        failure_reason = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        if training_status == "running":
            training_status = status
            training_failure_reason = failure_reason
        if inference_status == "running":
            inference_status = "failed"
    finally:
        total_runtime = time.time() - started
        update = {
            "status": status,
            "training_status": training_status,
            "inference_status": inference_status,
            # A complete three-objective result requires compatible inference.
            # Training quality remains independently preserved above when inference
            # fails, but such a row cannot enter a Pareto frontier.
            "valid_result": 1 if status == "completed" and training_status == "completed" and inference_status == "compatible" else 0,
            "exclude_from_analysis": 0 if status == "completed" and training_status == "completed" and inference_status == "compatible" else 1,
            "val_bpb": values.get("val_bpb"),
            "inference_tokens_per_second": values.get("inference_tokens_per_second"),
            "median_cached_decode_tokens_per_second": values.get("median_cached_decode_tokens_per_second"),
            "cached_decode_variation_tokens_per_second": values.get("cached_decode_variation_tokens_per_second"),
            "prefill_milliseconds": values.get("prefill_milliseconds"),
            "prefill_tokens_per_second": values.get("prefill_tokens_per_second"),
            "peak_inference_cuda_memory_mb": values.get("peak_inference_cuda_memory_mb"),
            "peak_reserved_inference_cuda_memory_mb": values.get("peak_reserved_inference_cuda_memory_mb"),
            "incremental_kv_cache_memory_mb": values.get("incremental_kv_cache_memory_mb"),
            "inference_metric_mode": values.get("inference_metric_mode"),
            "evaluator_git_commit": values.get("evaluator_git_commit"),
            "evaluator_sha256": values.get("evaluator_sha256"),
            "training_tokens_per_second": values.get("training_tokens_per_second"),
            "total_training_tokens": values.get("total_training_tokens"),
            "peak_training_cuda_memory_mb": values.get("peak_training_cuda_memory_mb"),
            "parameter_count": values.get("parameter_count"),
            "training_seconds": values.get("training_seconds"),
            "inference_seconds": values.get("inference_seconds"),
            "total_runtime_seconds": total_runtime,
            "checkpoint_path": str(checkpoint) if checkpoint.exists() else None,
            "sample_path": str(sample_path) if sample_path.exists() else None,
            "log_path": str(log_path),
            "crash_or_failure_reason": failure_reason,
        }
        db.execute(
            """UPDATE experiments SET status=:status, training_status=:training_status,
               inference_status=:inference_status,
               valid_result=:valid_result,
               exclude_from_analysis=:exclude_from_analysis, val_bpb=:val_bpb,
               inference_tokens_per_second=:inference_tokens_per_second,
               median_cached_decode_tokens_per_second=:median_cached_decode_tokens_per_second,
               cached_decode_variation_tokens_per_second=:cached_decode_variation_tokens_per_second,
               prefill_milliseconds=:prefill_milliseconds,
               prefill_tokens_per_second=:prefill_tokens_per_second,
               peak_inference_cuda_memory_mb=:peak_inference_cuda_memory_mb,
               peak_reserved_inference_cuda_memory_mb=:peak_reserved_inference_cuda_memory_mb,
               incremental_kv_cache_memory_mb=:incremental_kv_cache_memory_mb,
               inference_metric_mode=:inference_metric_mode,
               evaluator_git_commit=:evaluator_git_commit, evaluator_sha256=:evaluator_sha256,
               training_tokens_per_second=:training_tokens_per_second,
               total_training_tokens=:total_training_tokens,
               peak_training_cuda_memory_mb=:peak_training_cuda_memory_mb,
               parameter_count=:parameter_count, training_seconds=:training_seconds,
               inference_seconds=:inference_seconds, total_runtime_seconds=:total_runtime_seconds,
               checkpoint_path=:checkpoint_path, sample_path=:sample_path, log_path=:log_path,
               crash_or_failure_reason=:crash_or_failure_reason, finished_at=CURRENT_TIMESTAMP
               WHERE id=:id""",
            {**update, "id": record_id},
        )
        if training_status == "completed":
            db.execute(
                """UPDATE training_runs SET status=?, val_bpb=?, training_tokens_per_second=?,
                   total_training_tokens=?, peak_training_cuda_memory_mb=?, parameter_count=?,
                   training_seconds=?, checkpoint_path=?, failure_reason=?
                   WHERE id=?""",
                (
                    training_status, values.get("val_bpb"), values.get("training_tokens_per_second"),
                    values.get("total_training_tokens"), values.get("peak_training_cuda_memory_mb"),
                    values.get("parameter_count"), values.get("training_seconds"),
                    str(checkpoint) if checkpoint.exists() else None,
                    training_failure_reason, training_run_id,
                ),
            )
        else:
            db.execute(
                """UPDATE training_runs SET status=?, val_bpb=?, training_tokens_per_second=?,
               total_training_tokens=?, peak_training_cuda_memory_mb=?, parameter_count=?,
               training_seconds=?, checkpoint_path=?, failure_reason=?, finished_at=CURRENT_TIMESTAMP
               WHERE id=?""",
                (
                    training_status, values.get("val_bpb"), values.get("training_tokens_per_second"),
                    values.get("total_training_tokens"), values.get("peak_training_cuda_memory_mb"),
                    values.get("parameter_count"), values.get("training_seconds"),
                    str(checkpoint) if checkpoint.exists() else None,
                    training_failure_reason, training_run_id,
                ),
            )
        campaign = db.execute(
            "SELECT attempt_limit FROM campaign_state WHERE campaign_id=?", (args.campaign_id,)
        ).fetchone()
        # Training checkpoints are persisted before protected inference begins. A
        # post-training inference/cache/memory fault pauses AutoPareto so a human can
        # review the exact recorded failure before any further attempt is reserved.
        pause_for_inference_failure = (
            args.method == "autopareto"
            and training_status == "completed"
            and inference_status != "compatible"
        )
        campaign_status = (
            "paused" if pause_for_inference_failure
            else ("completed" if campaign and args.experiment_number >= int(campaign["attempt_limit"]) else "ready")
        )
        db.execute(
            """UPDATE campaign_state SET next_experiment_number=MAX(next_experiment_number, ?),
               status=?, updated_at=CURRENT_TIMESTAMP WHERE campaign_id=?""",
            (args.experiment_number + 1, campaign_status, args.campaign_id),
        )
        db.execute(
            """UPDATE campaigns SET status=?, updated_at=CURRENT_TIMESTAMP
               WHERE campaign_id=?""",
            (campaign_status, args.campaign_id),
        )
        related_status = (
            "completed" if status == "completed"
            else ("incompatible" if inference_status == "incompatible" else status)
        )
        update_related_run_status(
            db, args.phase, training_run_id, related_status, failure_reason
        )
        db.commit()
        import_event_log(db, event_path)
        record_event(
            db, event_uuid=f"{slug}-terminal-{status}", event_type="experiment_completed",
            campaign_id=args.campaign_id, experiment_id=record_id, training_run_id=training_run_id,
            status=status, elapsed_seconds=total_runtime,
            payload={
                "val_bpb": values.get("val_bpb"),
                "training_status": training_status,
                "inference_status": inference_status,
                "median_cached_decode_tokens_per_second": values.get("median_cached_decode_tokens_per_second"),
                "peak_inference_cuda_memory_mb": values.get("peak_inference_cuda_memory_mb"),
                "failure_reason": failure_reason,
            },
        )
        if pause_for_inference_failure:
            record_event(
                db, event_uuid=f"{slug}-campaign-paused-{time.time_ns()}",
                event_type="campaign_paused_after_inference_failure",
                campaign_id=args.campaign_id, experiment_id=record_id,
                training_run_id=training_run_id, status="paused",
                payload={
                    "experiment_number": args.experiment_number,
                    "checkpoint_path": str(checkpoint) if checkpoint.exists() else None,
                    "inference_status": inference_status,
                    "failure_reason": failure_reason,
                    "next_experiment_is_blocked": True,
                },
            )
        refresh_exports(db)
        if isolate.exists():
            shutil.rmtree(isolate)
    row = db.execute("SELECT * FROM experiments WHERE id=?", (record_id,)).fetchone()
    result = row_to_dict(row)
    print(json.dumps(result, sort_keys=True))
    return result


def run_baselines(args: argparse.Namespace) -> int:
    invalid_seeds = sorted(set(args.seeds) - set(range(5)))
    if invalid_seeds:
        raise ValueError(f"baseline seeds must be drawn from 0..4, got {invalid_seeds}")
    for seed in args.seeds:
        number = seed + 1
        experiment_args = argparse.Namespace(
            candidate=str(ROOT / "train.py"),
            campaign_id=args.campaign_id,
            experiment_number=number,
            method="baseline",
            phase="baseline",
            seed=seed,
            description=f"shared A40-compatible train.py baseline, seed {seed}",
            timeout=args.timeout,
            inference_timeout=args.inference_timeout,
            attempt_limit=5,
        )
        result = execute_experiment(experiment_args)
        if result.get("status") != "completed":
            print(json.dumps({
                "campaign_id": args.campaign_id,
                "failed_seed": seed,
                "status": result.get("status"),
                "message": "baseline suite stopped; later seeds were not started",
            }, sort_keys=True))
            return 1
    return 0


def protection_check() -> int:
    manifest = protected_manifest()
    temporary = WORKTREES / "protection-check"
    if temporary.exists():
        shutil.rmtree(temporary)
    copy_isolate(temporary)
    try:
        validate_isolate(temporary, manifest)
        target = temporary / "prepare.py"
        target.write_text(target.read_text() + "\n# forbidden mutation probe\n")
        try:
            validate_isolate(temporary, manifest)
        except RuntimeError as exc:
            print(json.dumps({"status": "passed", "detected": str(exc), "canonical_prepare_sha256": manifest["prepare.py"]}, sort_keys=True))
            return 0
        raise RuntimeError("protection check failed to detect mutation")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def mark_setup_ooms() -> int:
    """Normalize only the two known pre-baseline A40 feasibility attempts."""
    db = connect()
    targets = db.execute(
        """SELECT * FROM experiments
           WHERE campaign_id IN ('baseline-first', 'baseline-a40-async') ORDER BY id"""
    ).fetchall()
    if len(targets) != 2:
        raise RuntimeError(f"expected exactly two setup OOM rows, found {len(targets)}")
    for row in targets:
        if row["training_seed"] != 0 or row["val_bpb"] is not None or row["checkpoint_path"] is not None:
            raise RuntimeError(f"refusing to reclassify non-setup row id={row['id']}")
        db.execute(
            """UPDATE experiments SET phase='setup', method='setup_oom', status='oom',
               valid_result=0, exclude_from_analysis=1,
               description='setup-phase unchanged-upstream A40 feasibility attempt; excluded from analysis'
               WHERE id=?""",
            (row["id"],),
        )
    db.execute(
        "DELETE FROM campaign_state WHERE campaign_id IN ('baseline-first', 'baseline-a40-async')"
    )
    db.commit()
    refresh_exports(db)
    normalized = [row_to_dict(row) for row in db.execute(
        "SELECT * FROM experiments WHERE phase='setup' ORDER BY id"
    )]
    print(json.dumps(normalized, indent=2, sort_keys=True))
    return 0


def archive_dashboard_campaigns(args: argparse.Namespace) -> int:
    """Hide historical campaigns from the normal dashboard without changing results."""
    db = connect()
    requested = list(dict.fromkeys(args.campaign_ids))
    found = {
        row["campaign_id"] for row in db.execute(
            f"SELECT campaign_id FROM campaigns WHERE campaign_id IN ({','.join('?' for _ in requested)})",
            requested,
        )
    }
    missing = sorted(set(requested) - found)
    if missing:
        raise ValueError(f"campaigns do not exist: {', '.join(missing)}")
    db.executemany(
        """UPDATE campaigns SET dashboard_visible=0, status='archived', archive_reason=?,
           archived_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP WHERE campaign_id=?""",
        [(args.reason, campaign_id) for campaign_id in requested],
    )
    db.commit()
    print(json.dumps({"archived_campaigns": requested, "dashboard_visible": False}, sort_keys=True))
    return 0


def inspect_state() -> int:
    db = connect()
    counts = [dict(row) for row in db.execute("SELECT status, COUNT(*) AS count FROM experiments GROUP BY status")]
    print(json.dumps({"database": str(DB_PATH), "counts": counts, "protected_manifest": protected_manifest()}, indent=2, sort_keys=True))
    return 0


def campaign_view_command(args: argparse.Namespace) -> int:
    """Expose only the registered baseline and one AutoPareto campaign's history."""
    db = connect()
    registration = db.execute(
        "SELECT * FROM campaign_state WHERE campaign_id=?",
        (args.campaign_id,),
    ).fetchone()
    if registration is None or registration["method"] != "autopareto":
        raise ValueError(f"AutoPareto campaign is not registered: {args.campaign_id}")
    baseline_id = registration["baseline_experiment_id"]
    experiments = db.execute(
        """SELECT * FROM experiments
           WHERE campaign_id=? OR id=?
           ORDER BY CASE WHEN id=? THEN 0 ELSE 1 END, experiment_number, id""",
        (args.campaign_id, baseline_id, baseline_id),
    ).fetchall()
    ids = [int(row["id"]) for row in experiments]
    actions = []
    if ids:
        placeholders = ",".join("?" for _ in ids)
        actions = db.execute(
            f"""SELECT * FROM agent_actions
                WHERE experiment_id IN ({placeholders})
                ORDER BY experiment_id, sequence_number""",
            ids,
        ).fetchall()
    print(json.dumps({
        "campaign": row_to_dict(registration),
        "baseline_experiment_id": baseline_id,
        "frontier_ids": sorted(autopareto_frontier_ids(db, args.campaign_id)),
        "experiments": [row_to_dict(row) for row in experiments],
        "agent_actions": [row_to_dict(row) for row in actions],
    }, indent=2, sort_keys=True))
    return 0


def register_campaign_command(args: argparse.Namespace) -> int:
    db = connect()
    row = register_campaign(
        db,
        campaign_id=args.campaign_id,
        method=args.method,
        phase=args.phase,
        attempt_limit=args.attempt_limit,
        training_seed=args.seed,
        algorithm_seed=args.algorithm_seed,
        program_path=args.program,
        search_space_sha256=args.search_space_sha256,
        coding_model=args.coding_model,
        reasoning_level=args.reasoning_level,
        baseline_experiment_id=args.baseline_experiment_id,
    )
    print(json.dumps(row_to_dict(row), indent=2, sort_keys=True))
    return 0


def record_agent_action_command(args: argparse.Namespace) -> int:
    db = connect()
    experiment = db.execute("SELECT id FROM experiments WHERE id=?", (args.experiment_id,)).fetchone()
    if experiment is None:
        raise ValueError(f"experiment does not exist: {args.experiment_id}")
    next_sequence = args.sequence_number
    if next_sequence is None:
        next_sequence = int(db.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) + 1 AS n FROM agent_actions WHERE experiment_id=?",
            (args.experiment_id,),
        ).fetchone()["n"])
    db.execute(
        """INSERT INTO agent_actions
           (experiment_id, sequence_number, action_type, visible_content, tool_name,
            tool_input_json, tool_output_summary, input_tokens, output_tokens, api_cost_usd)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (args.experiment_id, next_sequence, args.action_type, args.visible_content,
         args.tool_name, args.tool_input_json, args.tool_output_summary,
         args.input_tokens, args.output_tokens, args.api_cost_usd),
    )
    db.commit()
    print(json.dumps({"experiment_id": args.experiment_id, "sequence_number": next_sequence}, sort_keys=True))
    return 0


def validate_autoresearch_decision(
    db: sqlite3.Connection,
    experiment: sqlite3.Row,
    decision: str,
) -> None:
    """Enforce the preregistered strict quality-only AutoResearch rule."""
    if experiment["method"] != "autoresearch":
        return
    if experiment["parent_experiment_id"] is None:
        raise ValueError("AutoResearch decision requires an explicit parent")
    parent = db.execute(
        "SELECT id, val_bpb FROM experiments WHERE id=?",
        (experiment["parent_experiment_id"],),
    ).fetchone()
    if parent is None or parent["val_bpb"] is None:
        raise ValueError("AutoResearch parent must have a recorded val_bpb")
    quality_completed = (
        experiment["training_status"] == "completed"
        and experiment["val_bpb"] is not None
        and experiment["status"] != "protection_failure"
    )
    strictly_improved = quality_completed and float(experiment["val_bpb"]) < float(parent["val_bpb"])
    if strictly_improved and decision != "retained":
        raise ValueError("strictly improved AutoResearch quality must be retained")
    if not strictly_improved and decision == "retained":
        raise ValueError("AutoResearch retention requires completed training and strict val_bpb improvement")


def validate_autopareto_decision(
    db: sqlite3.Connection,
    experiment: sqlite3.Row,
    decision: str,
) -> None:
    """Enforce the campaign-local three-objective Pareto decision rule."""
    if experiment["method"] != "autopareto":
        return
    frontier_ids = autopareto_frontier_ids(db, experiment["campaign_id"])
    expected = "retained" if int(experiment["id"]) in frontier_ids else "dominated"
    if decision != expected:
        raise ValueError(f"AutoPareto decision must be {expected} for the campaign-local frontier")


def raw_pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        candidate for candidate in rows
        if not any(
            other["id"] != candidate["id"]
            and other["val_bpb"] <= candidate["val_bpb"]
            and other["speed"] >= candidate["speed"]
            and other["memory"] <= candidate["memory"]
            and (
                other["val_bpb"] < candidate["val_bpb"]
                or other["speed"] > candidate["speed"]
                or other["memory"] < candidate["memory"]
            )
            for other in rows
        )
    ]


def update_related_run_status(
    db: sqlite3.Connection,
    phase: str,
    training_run_id: int,
    status: str,
    failure_reason: str | None,
) -> None:
    if phase == "confirmation":
        db.execute(
            """UPDATE confirmation_runs SET status=?, failure_reason=?
               WHERE training_run_id=?""",
            (status, failure_reason, training_run_id),
        )
    if phase == "ablation":
        db.execute(
            """UPDATE ablation_runs SET status=?, failure_reason=?
               WHERE training_run_id=?""",
            (status, failure_reason, training_run_id),
        )


def record_decision_command(args: argparse.Namespace) -> int:
    db = connect()
    experiment = db.execute(
        """SELECT id, campaign_id, method, status, training_status, inference_status,
                  val_bpb, parent_experiment_id
           FROM experiments WHERE id=?""",
        (args.experiment_id,),
    ).fetchone()
    if experiment is None:
        raise ValueError(f"experiment does not exist: {args.experiment_id}")
    validate_autoresearch_decision(db, experiment, args.decision)
    validate_autopareto_decision(db, experiment, args.decision)
    cursor = db.execute(
        "UPDATE experiments SET decision=?, agent_interpretation=? WHERE id=?",
        (args.decision, args.agent_interpretation, args.experiment_id),
    )
    if cursor.rowcount != 1:
        raise ValueError(f"experiment does not exist: {args.experiment_id}")
    record_event(
        db, event_uuid=f"experiment-{args.experiment_id}-decision-{time.time_ns()}",
        event_type="decision_recorded", campaign_id=experiment["campaign_id"],
        experiment_id=args.experiment_id, status=args.decision,
        payload={"decision": args.decision, "agent_interpretation": args.agent_interpretation},
    )
    refresh_exports(db)
    print(json.dumps({"experiment_id": args.experiment_id, "decision": args.decision}, sort_keys=True))
    return 0


def rebenchmark_cached(args: argparse.Namespace) -> int:
    """Run protected cached inference for an existing checkpoint without retraining."""
    db = connect()
    row = db.execute("SELECT * FROM experiments WHERE id=?", (args.experiment_id,)).fetchone()
    if row is None:
        raise ValueError(f"experiment does not exist: {args.experiment_id}")
    evaluator_correction = bool(getattr(args, "evaluator_correction", False))
    if evaluator_correction:
        if not (
            row["training_status"] == "completed"
            and row["inference_status"] == "incompatible"
            and not row["valid_result"]
            and row["exclude_from_analysis"]
        ):
            raise ValueError(
                "evaluator correction requires a completed training run with only an "
                "incompatible inference result"
            )
    elif row["status"] != "completed" or not row["valid_result"] or row["exclude_from_analysis"]:
        raise ValueError("cached rebenchmark requires a valid, included, completed experiment")
    if row["inference_metric_mode"] == "uncached_legacy":
        archived = db.execute(
            """SELECT COUNT(*) FROM legacy_metrics
               WHERE experiment_id=? AND metric_mode='uncached_legacy'""",
            (args.experiment_id,),
        ).fetchone()[0]
        if archived < 3:
            raise RuntimeError("refusing to replace official mode before legacy speed/memory/time are archived")
    checkpoint = Path(args.checkpoint or row["checkpoint_path"] or "").resolve()
    train_source = Path(args.train).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not train_source.is_file():
        raise FileNotFoundError(train_source)
    training_run = db.execute(
        """SELECT id FROM training_runs WHERE experiment_id=? AND seed=? ORDER BY id LIMIT 1""",
        (args.experiment_id, row["training_seed"]),
    ).fetchone()
    if training_run is None:
        raise RuntimeError("experiment has no normalized training run")
    training_run_id = int(training_run["id"])
    suffix = "evaluator-correction" if evaluator_correction else "cached"
    slug = f"rebenchmark-{args.experiment_id}-{suffix}"
    output = RESULTS / "tmp" / f"{slug}.json"
    sample = RESULTS / "samples" / "cached_rebenchmark" / f"experiment-{args.experiment_id}.txt"
    log = RESULTS / "logs" / f"{slug}.log"
    events = RESULTS / "events" / f"{slug}.jsonl"
    canonical = protected_manifest()
    command = [
        sys.executable, "inference_benchmark.py", "--train", str(train_source),
        "--checkpoint", str(checkpoint), "--output", str(output), "--sample", str(sample),
        "--event-log", str(events), "--campaign-id", row["campaign_id"],
        "--experiment-id", str(args.experiment_id), "--training-run-id", str(training_run_id),
    ]
    if evaluator_correction:
        record_event(
            db, event_uuid=f"{slug}-prior-state-{time.time_ns()}",
            event_type="evaluator_correction_prior_state", campaign_id=row["campaign_id"],
            experiment_id=args.experiment_id, training_run_id=training_run_id,
            status="historical", payload={
                "previous_status": row["status"],
                "previous_inference_status": row["inference_status"],
                "previous_valid_result": bool(row["valid_result"]),
                "previous_failure_reason": row["crash_or_failure_reason"],
                "reason": (
                    "protected cache correctness evaluator correction; every bfloat16 "
                    "correctness failure is adjudicated by strict float32 equivalence"
                ),
            },
        )
    record_event(
        db, event_uuid=f"{slug}-started-{time.time_ns()}", event_type="cached_rebenchmark_started",
        campaign_id=row["campaign_id"], experiment_id=args.experiment_id,
        training_run_id=training_run_id, status="running",
    )
    with gpu_lock():
        returncode, timeout_reason = run_process(command, ROOT, log, args.inference_timeout)
    if protected_manifest() != canonical:
        raise RuntimeError("canonical protected files changed during cached rebenchmark")
    import_event_log(db, events)
    if returncode != 0:
        reason = timeout_reason or tail_failure(log)
        record_event(
            db, event_uuid=f"{slug}-failed-{time.time_ns()}", event_type="cached_rebenchmark_failed",
            campaign_id=row["campaign_id"], experiment_id=args.experiment_id,
            training_run_id=training_run_id, status="timeout" if returncode == 124 else "crash",
            payload={"failure_reason": reason},
        )
        print(json.dumps({"status": "failed", "experiment_id": args.experiment_id, "reason": reason}, sort_keys=True))
        return 1
    metrics = read_json(output)
    metrics["evaluator_git_commit"] = metrics.get("evaluator_git_commit") or current_git_commit()
    if metrics.get("inference_metric_mode") != CACHED_MODE or not metrics.get("correctness", {}).get("passed"):
        raise RuntimeError("protected evaluator did not return a validated cached_decode measurement")
    store_inference_repetitions(db, args.experiment_id, training_run_id, metrics)
    update_fields = "" if not evaluator_correction else """
           status='completed', inference_status='compatible', valid_result=1,
           exclude_from_analysis=0, crash_or_failure_reason=NULL,
    """
    db.execute(
        """UPDATE experiments SET inference_metric_mode=?,
           """ + update_fields + """
           inference_tokens_per_second=?, median_cached_decode_tokens_per_second=?,
           cached_decode_variation_tokens_per_second=?, prefill_milliseconds=?,
           prefill_tokens_per_second=?, peak_inference_cuda_memory_mb=?,
           peak_reserved_inference_cuda_memory_mb=?, incremental_kv_cache_memory_mb=?,
           inference_seconds=?, evaluator_git_commit=?, evaluator_sha256=? WHERE id=?""",
        (
            CACHED_MODE, metrics["median_cached_decode_tokens_per_second"],
            metrics["median_cached_decode_tokens_per_second"],
            metrics["cached_decode_variation_tokens_per_second"], metrics["prefill_milliseconds"],
            metrics["prefill_tokens_per_second"], metrics["peak_inference_cuda_memory_mb"],
            metrics["peak_reserved_inference_cuda_memory_mb"], metrics["incremental_kv_cache_memory_mb"],
            metrics["inference_seconds"], metrics.get("evaluator_git_commit"),
            metrics.get("evaluator_sha256"), args.experiment_id,
        ),
    )
    corrected_decision = None
    corrected_campaign_status = None
    if evaluator_correction:
        frontier_ids = autopareto_frontier_ids(db, row["campaign_id"])
        corrected_decision = "retained" if args.experiment_id in frontier_ids else "dominated"
        db.execute(
            "UPDATE experiments SET decision=? WHERE id=?",
            (corrected_decision, args.experiment_id),
        )
        campaign = db.execute(
            "SELECT attempt_limit FROM campaign_state WHERE campaign_id=?",
            (row["campaign_id"],),
        ).fetchone()
        corrected_campaign_status = (
            "completed"
            if campaign and int(row["experiment_number"]) >= int(campaign["attempt_limit"])
            else "ready"
        )
        db.execute(
            """UPDATE campaign_state SET status=?, updated_at=CURRENT_TIMESTAMP
               WHERE campaign_id=? AND status='paused'""",
            (corrected_campaign_status, row["campaign_id"]),
        )
        db.execute(
            """UPDATE campaigns SET status=?, updated_at=CURRENT_TIMESTAMP
               WHERE campaign_id=? AND status='paused'""",
            (corrected_campaign_status, row["campaign_id"]),
        )
    db.commit()
    if evaluator_correction:
        record_event(
            db, event_uuid=f"{slug}-decision-{time.time_ns()}",
            event_type="evaluator_correction_decision", campaign_id=row["campaign_id"],
            experiment_id=args.experiment_id, training_run_id=training_run_id,
            status=corrected_decision, payload={
                "previous_decision": row["decision"],
                "corrected_decision": corrected_decision,
                "rule": "campaign-local three-objective Pareto frontier after corrected inference",
                "campaign_status": corrected_campaign_status,
            },
        )
    record_event(
        db, event_uuid=f"{slug}-completed-{time.time_ns()}", event_type="cached_rebenchmark_completed",
        campaign_id=row["campaign_id"], experiment_id=args.experiment_id,
        training_run_id=training_run_id, status="completed", payload={
            "evaluator_correction": evaluator_correction,
            "median_cached_decode_tokens_per_second": metrics["median_cached_decode_tokens_per_second"],
            "peak_inference_cuda_memory_mb": metrics["peak_inference_cuda_memory_mb"],
        },
    )
    refresh_exports(db)
    print(json.dumps({"status": "completed", "experiment_id": args.experiment_id, "metrics": metrics}, sort_keys=True))
    return 0


def characterize_inference(args: argparse.Namespace) -> int:
    """Measure supporting deployment workloads without changing headline columns."""
    db = connect()
    row = db.execute("SELECT * FROM experiments WHERE id=?", (args.experiment_id,)).fetchone()
    if row is None:
        raise ValueError("deployment characterization requires an existing experiment")
    training_run = db.execute(
        "SELECT id, status FROM training_runs WHERE experiment_id=? ORDER BY id LIMIT 1",
        (args.experiment_id,),
    ).fetchone()
    if (
        row["training_status"] != "completed"
        and (training_run is None or training_run["status"] != "completed")
    ):
        raise ValueError("deployment characterization requires a completed training result")
    train_source = Path(args.train or (
        RESULTS / "candidates" / row["campaign_id"] / f"{row['experiment_number']:04d}-train.py"
    )).resolve()
    checkpoint = Path(args.checkpoint or row["checkpoint_path"] or "").resolve()
    if not train_source.is_file():
        raise FileNotFoundError(train_source)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if training_run is None:
        raise RuntimeError("experiment has no normalized training run")
    training_run_id = int(training_run["id"])
    canonical = protected_manifest()
    parsed = []
    for workload in args.workloads:
        try:
            context, batch = (int(part) for part in workload.split(":", 1))
        except ValueError as exc:
            raise ValueError(f"invalid workload {workload!r}; expected CONTEXT:BATCH") from exc
        parsed.append((context, batch))
    failures = []
    with gpu_lock():
        for context, batch in parsed:
            name = f"context{context}-batch{batch}-decode{args.generated_tokens}"
            completed = db.execute(
                """SELECT COUNT(*) FROM deployment_measurements
                   WHERE experiment_id=? AND workload_name=? AND status='completed'
                     AND is_warmup=0""",
                (args.experiment_id, name),
            ).fetchone()[0]
            if completed >= args.repetitions:
                continue
            output = RESULTS / "tmp" / f"deployment-{args.experiment_id}-{name}.json"
            log = RESULTS / "logs" / f"deployment-{args.experiment_id}-{name}.log"
            output.unlink(missing_ok=True)
            command = [
                sys.executable, "deployment_benchmark.py",
                "--train", str(train_source), "--checkpoint", str(checkpoint),
                "--output", str(output), "--input-tokens", str(context),
                "--batch-size", str(batch), "--generated-tokens", str(args.generated_tokens),
                "--warmups", str(args.warmups), "--repetitions", str(args.repetitions),
            ]
            returncode, timeout_reason = run_process(command, ROOT, log, args.inference_timeout)
            if returncode:
                reason = timeout_reason or tail_failure(log)
                failures.append({"workload": name, "failure_reason": reason})
                db.execute(
                    """INSERT OR REPLACE INTO deployment_measurements
                       (experiment_id, training_run_id, workload_name, input_tokens,
                        generated_tokens, inference_batch_size, repetition_index,
                        is_warmup, status, failure_reason, raw_json)
                       VALUES (?, ?, ?, ?, ?, ?, -1, 0, 'failed', ?, ?)""",
                    (
                        args.experiment_id, training_run_id, name, context,
                        args.generated_tokens, batch, reason,
                        json.dumps({"failure_reason": reason}, sort_keys=True),
                    ),
                )
                db.commit()
                continue
            metrics = read_json(output)
            correctness = int(bool(metrics.get("correctness", {}).get("passed")))
            for repetition in [
                *metrics.get("warmup_repetitions", []),
                *metrics.get("inference_repetitions", []),
            ]:
                db.execute(
                    """INSERT OR REPLACE INTO deployment_measurements
                       (experiment_id, training_run_id, workload_name, input_tokens,
                        generated_tokens, inference_batch_size, repetition_index, is_warmup,
                        prefill_milliseconds, prefill_tokens_per_second, decode_seconds,
                        decode_tokens_per_second, peak_allocated_cuda_memory_mb,
                        peak_reserved_cuda_memory_mb, prefill_kv_cache_memory_mb,
                        incremental_kv_cache_memory_mb, correctness_passed,
                        evaluator_git_commit, evaluator_sha256, raw_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        args.experiment_id, training_run_id, name, context,
                        args.generated_tokens, batch, repetition["repetition_index"],
                        int(bool(repetition.get("is_warmup"))),
                        repetition.get("prefill_milliseconds"),
                        repetition.get("prefill_tokens_per_second"),
                        repetition.get("decode_seconds"),
                        repetition.get("decode_tokens_per_second"),
                        repetition.get("peak_allocated_cuda_memory_mb"),
                        repetition.get("peak_reserved_cuda_memory_mb"),
                        repetition.get("prefill_kv_cache_memory_mb"),
                        repetition.get("incremental_kv_cache_memory_mb"),
                        correctness, metrics.get("evaluator_git_commit"),
                        metrics.get("evaluator_sha256"),
                        json.dumps(repetition, sort_keys=True),
                    ),
                )
            db.execute(
                """UPDATE deployment_measurements
                   SET status='completed', failure_reason=NULL
                   WHERE experiment_id=? AND workload_name=?""",
                (args.experiment_id, name),
            )
            db.commit()
    if protected_manifest() != canonical:
        raise RuntimeError("protected files changed during deployment characterization")
    print(json.dumps({
        "status": "completed",
        "experiment_id": args.experiment_id,
        "workloads": [f"{context}:{batch}" for context, batch in parsed],
        "failures": failures,
    }, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db")
    sub.add_parser("protection-check")
    sub.add_parser("mark-setup-ooms")
    sub.add_parser("state")
    sub.add_parser("frontiers")
    campaign_view = sub.add_parser(
        "campaign-view",
        help="campaign-isolated AutoPareto baseline, history, and frontier",
    )
    campaign_view.add_argument("--campaign-id", required=True)
    register = sub.add_parser("register-campaign")
    register.add_argument("--campaign-id", required=True)
    register.add_argument("--method", required=True)
    register.add_argument(
        "--phase",
        choices=("setup", "baseline", "campaign", "confirmation", "ablation"),
        default="campaign",
    )
    register.add_argument("--attempt-limit", type=int, required=True)
    register.add_argument("--seed", type=int, required=True)
    register.add_argument("--algorithm-seed", type=int)
    register.add_argument("--program", type=Path)
    register.add_argument("--search-space-sha256")
    register.add_argument("--coding-model", default="gpt-5.6-sol")
    register.add_argument("--reasoning-level", default="high")
    register.add_argument("--baseline-experiment-id", type=int)
    archive = sub.add_parser("archive-dashboard-campaigns", help="hide audited campaigns from normal dashboard views")
    archive.add_argument("campaign_ids", nargs="+")
    archive.add_argument("--reason", default="superseded by clean benchmark campaign")

    run = sub.add_parser("run")
    run.add_argument("--candidate")
    run.add_argument("--campaign-id", required=True)
    run.add_argument("--experiment-number", required=True, type=int)
    run.add_argument("--method", required=True)
    run.add_argument("--phase", choices=("setup", "baseline", "campaign", "confirmation", "ablation"), default="campaign")
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--description", required=True)
    run.add_argument("--hypothesis")
    run.add_argument("--intended-change")
    run.add_argument("--change-category")
    run.add_argument("--parent-experiment-id", type=int)
    run.add_argument("--agent-interpretation")
    run.add_argument("--coding-model", default="gpt-5.6-sol")
    run.add_argument("--reasoning-level", default="high")
    run.add_argument("--program", type=Path)
    run.add_argument("--selection-role")
    run.add_argument("--ablation-name")
    run.add_argument("--timeout", type=int, default=TRAINING_TIMEOUT_SECONDS)
    run.add_argument("--inference-timeout", type=int, default=600)
    run.add_argument("--attempt-limit", type=int, default=25)
    run.add_argument("--algorithm-seed", type=int)
    run.add_argument("--search-space-sha256")
    run.add_argument("--baseline-experiment-id", type=int)

    baseline = sub.add_parser("baseline", help="run the shared A40-compatible train.py for requested seeds")
    baseline.add_argument("--campaign-id", default="baseline-seeds-0-4")
    baseline.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    baseline.add_argument("--timeout", type=int, default=TRAINING_TIMEOUT_SECONDS)
    baseline.add_argument("--inference-timeout", type=int, default=600)

    action = sub.add_parser("record-action", help="record a visible agent message or tool action")
    action.add_argument("--experiment-id", required=True, type=int)
    action.add_argument("--sequence-number", type=int)
    action.add_argument("--action-type", required=True)
    action.add_argument("--visible-content")
    action.add_argument("--tool-name")
    action.add_argument("--tool-input-json")
    action.add_argument("--tool-output-summary")
    action.add_argument("--input-tokens", type=int)
    action.add_argument("--output-tokens", type=int)
    action.add_argument("--api-cost-usd", type=float)

    decision = sub.add_parser("record-decision")
    decision.add_argument("--experiment-id", required=True, type=int)
    decision.add_argument("--decision", required=True, choices=("retained", "dominated", "discarded", "confirmed", "excluded"))
    decision.add_argument("--agent-interpretation", required=True)

    rebenchmark = sub.add_parser("rebenchmark-cached", help="cached-only inference for an existing checkpoint")
    rebenchmark.add_argument("--experiment-id", required=True, type=int)
    rebenchmark.add_argument("--train", type=Path, default=ROOT / "train.py")
    rebenchmark.add_argument("--checkpoint", type=Path)
    rebenchmark.add_argument("--inference-timeout", type=int, default=900)
    rebenchmark.add_argument(
        "--evaluator-correction", action="store_true",
        help="allow one audited inference-only correction for an incompatible completed run",
    )

    characterize = sub.add_parser(
        "characterize-inference",
        help="supporting context/batch workloads that do not overwrite headline metrics",
    )
    characterize.add_argument("--experiment-id", required=True, type=int)
    characterize.add_argument("--train", type=Path)
    characterize.add_argument("--checkpoint", type=Path)
    characterize.add_argument(
        "--workloads", nargs="+",
        default=["128:1", "256:1", "512:1", "1024:1", "256:2", "256:4", "256:8"],
    )
    characterize.add_argument("--generated-tokens", type=int, default=128)
    characterize.add_argument("--warmups", type=int, default=1)
    characterize.add_argument("--repetitions", type=int, default=5)
    characterize.add_argument("--inference-timeout", type=int, default=900)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "init-db":
        db = connect()
        refresh_exports(db)
        print(DB_PATH)
        return 0
    if args.command == "protection-check":
        return protection_check()
    if args.command == "mark-setup-ooms":
        return mark_setup_ooms()
    if args.command == "state":
        return inspect_state()
    if args.command == "frontiers":
        db = connect()
        refresh_exports(db)
        print(json.dumps({"raw": raw_frontier(db), "confirmed": confirmed_frontier(db), "quality_budgets": quality_budget_report(db)}, indent=2, sort_keys=True))
        return 0
    if args.command == "campaign-view":
        return campaign_view_command(args)
    if args.command == "register-campaign":
        return register_campaign_command(args)
    if args.command == "archive-dashboard-campaigns":
        return archive_dashboard_campaigns(args)
    if args.command == "record-action":
        return record_agent_action_command(args)
    if args.command == "record-decision":
        return record_decision_command(args)
    if args.command == "rebenchmark-cached":
        return rebenchmark_cached(args)
    if args.command == "characterize-inference":
        return characterize_inference(args)
    if args.command == "run":
        execute_experiment(args)
        return 0
    if args.command == "baseline":
        return run_baselines(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Authoritative additive schema, migrations, events, and frontier analysis."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "results" / "results.db"
CACHED_MODE = "cached_decode"
LEGACY_MODE = "uncached_legacy"
MISCLASSIFIED_CACHED_MODE = "cached_decode_migration_duplicate"
CONFIRMATION_SEEDS = (0, 1, 2)
QUALITY_BUDGETS = (0.005, 0.01, 0.02, 0.05)


ADDITIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'campaign',
    attempt_limit INTEGER,
    training_seed INTEGER,
    algorithm_seed INTEGER,
    search_space_sha256 TEXT,
    starting_git_commit TEXT,
    coding_model_name TEXT,
    reasoning_level TEXT,
    program_path TEXT,
    program_sha256 TEXT,
    program_instructions TEXT,
    clean_task_identifier TEXT,
    baseline_experiment_id INTEGER REFERENCES experiments(id),
    status TEXT NOT NULL DEFAULT 'planned',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    run_kind TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    val_bpb REAL,
    training_tokens_per_second REAL,
    total_training_tokens INTEGER,
    peak_training_cuda_memory_mb REAL,
    parameter_count INTEGER,
    training_seconds REAL,
    checkpoint_path TEXT,
    failure_reason TEXT,
    started_at TEXT,
    finished_at TEXT,
    UNIQUE(experiment_id, run_kind, seed)
);
CREATE INDEX IF NOT EXISTS idx_training_runs_experiment ON training_runs(experiment_id, run_kind, seed);

CREATE TABLE IF NOT EXISTS inference_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    training_run_id INTEGER REFERENCES training_runs(id),
    repetition_index INTEGER NOT NULL,
    is_warmup INTEGER NOT NULL DEFAULT 0,
    metric_mode TEXT NOT NULL,
    prefill_milliseconds REAL,
    prefill_tokens_per_second REAL,
    decode_seconds REAL,
    decode_tokens_per_second REAL,
    peak_allocated_cuda_memory_mb REAL,
    peak_reserved_cuda_memory_mb REAL,
    prefill_kv_cache_memory_mb REAL,
    incremental_kv_cache_memory_mb REAL,
    correctness_passed INTEGER,
    max_logit_absolute_error REAL,
    evaluator_git_commit TEXT,
    evaluator_sha256 TEXT,
    raw_json TEXT,
    measured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, training_run_id, metric_mode, repetition_index, is_warmup)
);
CREATE INDEX IF NOT EXISTS idx_inference_measurements_experiment ON inference_measurements(experiment_id, metric_mode);

CREATE TABLE IF NOT EXISTS deployment_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name TEXT NOT NULL UNIQUE,
    selection_path TEXT,
    selection_sha256 TEXT,
    workload_spec_sha256 TEXT,
    evaluator_sha256 TEXT,
    hardware_json TEXT,
    status TEXT NOT NULL DEFAULT 'planned',
    failure_reason TEXT,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS deployment_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_session_id INTEGER NOT NULL REFERENCES deployment_sessions(id),
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    training_run_id INTEGER REFERENCES training_runs(id),
    workload_name TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    generated_tokens INTEGER NOT NULL,
    inference_batch_size INTEGER NOT NULL,
    repetition_index INTEGER NOT NULL,
    is_warmup INTEGER NOT NULL DEFAULT 0,
    prefill_milliseconds REAL,
    prefill_tokens_per_second REAL,
    decode_seconds REAL,
    decode_tokens_per_second REAL,
    peak_allocated_cuda_memory_mb REAL,
    peak_reserved_cuda_memory_mb REAL,
    prefill_kv_cache_memory_mb REAL,
    incremental_kv_cache_memory_mb REAL,
    correctness_passed INTEGER,
    evaluator_git_commit TEXT,
    evaluator_sha256 TEXT,
    raw_json TEXT,
    measured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(deployment_session_id, experiment_id, workload_name, repetition_index, is_warmup)
);
CREATE TABLE IF NOT EXISTS downstream_evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    checkpoint_path TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    candidate_path TEXT NOT NULL,
    candidate_sha256 TEXT NOT NULL,
    suite_path TEXT NOT NULL,
    suite_sha256 TEXT NOT NULL,
    evaluator_git_commit TEXT,
    evaluator_sha256 TEXT NOT NULL,
    selection_role TEXT,
    display_name TEXT,
    selection_sha256 TEXT,
    environment_json TEXT NOT NULL,
    status TEXT NOT NULL,
    raw_result_path TEXT,
    failure_reason TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE(experiment_id, checkpoint_sha256, suite_sha256, evaluator_sha256)
);
CREATE INDEX IF NOT EXISTS idx_downstream_runs_experiment
ON downstream_evaluation_runs(experiment_id, status);

CREATE TABLE IF NOT EXISTS downstream_task_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    downstream_run_id INTEGER NOT NULL REFERENCES downstream_evaluation_runs(id),
    task_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_revision TEXT NOT NULL,
    dataset_config TEXT,
    split TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_standard_error REAL,
    confidence_low REAL,
    confidence_high REAL,
    sample_count INTEGER,
    higher_is_better INTEGER NOT NULL,
    raw_json TEXT NOT NULL,
    UNIQUE(downstream_run_id, task_name, dataset_config, metric_name)
);
CREATE INDEX IF NOT EXISTS idx_downstream_task_results_run
ON downstream_task_results(downstream_run_id, task_name);

CREATE TABLE IF NOT EXISTS downstream_generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    downstream_run_id INTEGER NOT NULL REFERENCES downstream_evaluation_runs(id),
    prompt_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    generation_seed INTEGER NOT NULL,
    generated_text TEXT NOT NULL,
    generated_token_count INTEGER,
    generation_seconds REAL,
    cache_correctness_passed INTEGER,
    settings_json TEXT NOT NULL,
    UNIQUE(downstream_run_id, prompt_id, generation_seed)
);
CREATE INDEX IF NOT EXISTS idx_downstream_generations_run
ON downstream_generations(downstream_run_id, prompt_id);

CREATE TABLE IF NOT EXISTS quantization_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    precision TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    candidate_path TEXT NOT NULL,
    candidate_sha256 TEXT NOT NULL,
    suite_path TEXT NOT NULL,
    suite_sha256 TEXT NOT NULL,
    evaluator_sha256 TEXT NOT NULL,
    backend_name TEXT NOT NULL,
    backend_version TEXT,
    status TEXT NOT NULL,
    artifact_path TEXT,
    artifact_bytes INTEGER,
    weight_storage_bytes INTEGER,
    heldout_bpb REAL,
    raw_result_path TEXT,
    failure_reason TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE(experiment_id, precision, checkpoint_sha256, suite_sha256, evaluator_sha256)
);
CREATE INDEX IF NOT EXISTS idx_quantization_runs_experiment
ON quantization_runs(experiment_id, precision, status);

CREATE TABLE IF NOT EXISTS quantization_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quantization_run_id INTEGER NOT NULL REFERENCES quantization_runs(id),
    workload_name TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    generated_tokens INTEGER NOT NULL,
    inference_batch_size INTEGER NOT NULL,
    repetition_index INTEGER NOT NULL,
    is_warmup INTEGER NOT NULL DEFAULT 0,
    prefill_milliseconds REAL,
    decode_tokens_per_second REAL,
    peak_allocated_cuda_memory_mb REAL,
    peak_reserved_cuda_memory_mb REAL,
    incremental_kv_cache_memory_mb REAL,
    correctness_passed INTEGER,
    raw_json TEXT NOT NULL,
    measured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(quantization_run_id, workload_name, repetition_index, is_warmup)
);
CREATE INDEX IF NOT EXISTS idx_quantization_measurements_run
ON quantization_measurements(quantization_run_id, workload_name);

CREATE TABLE IF NOT EXISTS legacy_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    metric_mode TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    unit TEXT,
    source_column TEXT NOT NULL,
    evaluator_sha256 TEXT,
    raw_json TEXT,
    archived_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, metric_mode, metric_name)
);

CREATE TABLE IF NOT EXISTS confirmation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    training_run_id INTEGER NOT NULL REFERENCES training_runs(id),
    seed INTEGER NOT NULL,
    selection_role TEXT,
    status TEXT NOT NULL,
    failure_reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_experiment_id, seed)
);

CREATE TABLE IF NOT EXISTS ablation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    training_run_id INTEGER NOT NULL REFERENCES training_runs(id),
    ablation_name TEXT NOT NULL,
    isolated_change TEXT NOT NULL,
    hypothesis TEXT,
    status TEXT NOT NULL,
    failure_reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_experiment_id, training_run_id, ablation_name)
);

CREATE TABLE IF NOT EXISTS agent_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    sequence_number INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    visible_content TEXT,
    tool_name TEXT,
    tool_input_json TEXT,
    tool_output_summary TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    api_cost_usd REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, sequence_number)
);

CREATE TABLE IF NOT EXISTS experiment_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uuid TEXT NOT NULL UNIQUE,
    campaign_id TEXT,
    experiment_id INTEGER REFERENCES experiments(id),
    training_run_id INTEGER REFERENCES training_runs(id),
    event_type TEXT NOT NULL,
    status TEXT,
    event_time TEXT NOT NULL,
    elapsed_seconds REAL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'live'
);
CREATE INDEX IF NOT EXISTS idx_events_timeline ON experiment_events(event_time, id);
"""


EXPERIMENT_COLUMNS = {
    "hypothesis": "TEXT",
    "intended_change": "TEXT",
    "change_category": "TEXT",
    "parent_experiment_id": "INTEGER REFERENCES experiments(id)",
    "agent_interpretation": "TEXT",
    "decision": "TEXT",
    "coding_model_name": "TEXT",
    "reasoning_level": "TEXT",
    "program_sha256": "TEXT",
    "evaluator_git_commit": "TEXT",
    "evaluator_sha256": "TEXT",
    "training_status": "TEXT",
    "inference_status": "TEXT",
    "inference_metric_mode": "TEXT",
    "median_cached_decode_tokens_per_second": "REAL",
    "cached_decode_variation_tokens_per_second": "REAL",
    "prefill_milliseconds": "REAL",
    "prefill_tokens_per_second": "REAL",
    "peak_reserved_inference_cuda_memory_mb": "REAL",
    "incremental_kv_cache_memory_mb": "REAL",
}

CAMPAIGN_COLUMNS = {
    "training_seed": "INTEGER",
    "algorithm_seed": "INTEGER",
    "coding_model_name": "TEXT",
    "reasoning_level": "TEXT",
    "search_space_sha256": "TEXT",
    "dashboard_visible": "INTEGER NOT NULL DEFAULT 1",
    "archive_reason": "TEXT",
    "archived_at": "TEXT",
    "baseline_experiment_id": "INTEGER REFERENCES experiments(id)",
}

ABLATION_RUN_COLUMNS = {
    "failure_reason": "TEXT",
}

CAMPAIGN_STATE_COLUMNS = {
    "phase": "TEXT NOT NULL DEFAULT 'campaign'",
    "training_seed": "INTEGER",
    "algorithm_seed": "INTEGER",
    "coding_model_name": "TEXT",
    "reasoning_level": "TEXT",
    "program_sha256": "TEXT",
    "search_space_sha256": "TEXT",
    "baseline_experiment_id": "INTEGER REFERENCES experiments(id)",
}

INFERENCE_MEASUREMENT_COLUMNS = {
    "input_tokens": "INTEGER",
    "generated_tokens": "INTEGER",
    "inference_batch_size": "INTEGER NOT NULL DEFAULT 1",
    "workload_name": "TEXT NOT NULL DEFAULT 'headline'",
}

DEPLOYMENT_MEASUREMENT_COLUMNS = {
    "status": "TEXT NOT NULL DEFAULT 'completed'",
    "failure_reason": "TEXT",
    "protocol_version": "TEXT",
    "measurement_pass": "INTEGER",
    "stability_block_index": "INTEGER",
    "prefill_cuda_milliseconds": "REAL",
    "decode_cuda_seconds": "REAL",
    "timing_source": "TEXT",
    "gpu_telemetry_json": "TEXT",
}

DEPLOYMENT_SESSION_COLUMNS = {
    "protocol_version": "TEXT",
    "environment_json": "TEXT",
    "measurement_order_json": "TEXT",
    "supersedes_session_id": "INTEGER REFERENCES deployment_sessions(id)",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    migrate(db)
    return db


def _add_columns(db: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
    for name, declaration in columns.items():
        if name not in existing:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def migrate(db: sqlite3.Connection) -> None:
    """Apply idempotent, additive migrations without replacing scientific values."""
    db.executescript(ADDITIVE_SCHEMA)
    _add_columns(db, "experiments", EXPERIMENT_COLUMNS)
    _add_columns(db, "campaigns", CAMPAIGN_COLUMNS)
    _add_columns(db, "ablation_runs", ABLATION_RUN_COLUMNS)
    _add_columns(db, "campaign_state", CAMPAIGN_STATE_COLUMNS)
    _add_columns(db, "inference_measurements", INFERENCE_MEASUREMENT_COLUMNS)
    _migrate_deployment_sessions(db)
    _add_columns(db, "deployment_measurements", DEPLOYMENT_MEASUREMENT_COLUMNS)
    _add_columns(db, "deployment_sessions", DEPLOYMENT_SESSION_COLUMNS)
    db.execute(
        """CREATE INDEX IF NOT EXISTS idx_deployment_measurements_experiment
           ON deployment_measurements(deployment_session_id, experiment_id, workload_name)"""
    )
    _add_columns(db, "downstream_evaluation_runs", {
        "selection_role": "TEXT",
        "display_name": "TEXT",
        "selection_sha256": "TEXT",
    })
    _add_columns(db, "downstream_generations", {
        "generated_token_count": "INTEGER",
        "generation_seconds": "REAL",
        "cache_correctness_passed": "INTEGER",
    })
    db.execute("CREATE INDEX IF NOT EXISTS idx_experiments_parent ON experiments(parent_experiment_id)")
    _backfill_campaigns_and_training_runs(db)
    _normalize_operational_metadata(db)
    _archive_legacy_inference(db)
    db.commit()


def _migrate_deployment_sessions(db: sqlite3.Connection) -> None:
    """Version deployment evidence without altering historical measurements."""
    columns = {row["name"] for row in db.execute("PRAGMA table_info(deployment_measurements)")}
    if "deployment_session_id" in columns:
        return
    db.execute(
        """INSERT OR IGNORE INTO deployment_sessions
           (session_name, status, started_at, finished_at, hardware_json)
           VALUES ('historical-pre-autopareto-a40-20260727', 'completed',
                   '2026-07-27T00:00:00Z', '2026-07-27T23:59:59Z',
                   '{"gpu":"NVIDIA A40","provenance":"legacy deployment matrix"}')"""
    )
    legacy_session_id = int(db.execute(
        "SELECT id FROM deployment_sessions WHERE session_name=?",
        ("historical-pre-autopareto-a40-20260727",),
    ).fetchone()[0])
    db.execute("ALTER TABLE deployment_measurements RENAME TO deployment_measurements_legacy")
    db.execute("DROP INDEX IF EXISTS idx_deployment_measurements_experiment")
    db.executescript(
        """
        CREATE TABLE deployment_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deployment_session_id INTEGER NOT NULL REFERENCES deployment_sessions(id),
            experiment_id INTEGER NOT NULL REFERENCES experiments(id),
            training_run_id INTEGER REFERENCES training_runs(id),
            workload_name TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            generated_tokens INTEGER NOT NULL,
            inference_batch_size INTEGER NOT NULL,
            repetition_index INTEGER NOT NULL,
            is_warmup INTEGER NOT NULL DEFAULT 0,
            prefill_milliseconds REAL,
            prefill_tokens_per_second REAL,
            decode_seconds REAL,
            decode_tokens_per_second REAL,
            peak_allocated_cuda_memory_mb REAL,
            peak_reserved_cuda_memory_mb REAL,
            prefill_kv_cache_memory_mb REAL,
            incremental_kv_cache_memory_mb REAL,
            correctness_passed INTEGER,
            evaluator_git_commit TEXT,
            evaluator_sha256 TEXT,
            raw_json TEXT,
            measured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'completed',
            failure_reason TEXT,
            UNIQUE(deployment_session_id, experiment_id, workload_name,
                   repetition_index, is_warmup)
        );
        CREATE INDEX idx_deployment_measurements_experiment
        ON deployment_measurements(deployment_session_id, experiment_id, workload_name);
        """
    )
    old_columns = [row["name"] for row in db.execute("PRAGMA table_info(deployment_measurements_legacy)")]
    copied_columns = [name for name in old_columns if name != "deployment_session_id"]
    rendered = ", ".join(copied_columns)
    db.execute(
        f"""INSERT INTO deployment_measurements
              (deployment_session_id, {rendered})
            SELECT ?, {rendered} FROM deployment_measurements_legacy""",
        (legacy_session_id,),
    )
    old_count = db.execute("SELECT COUNT(*) FROM deployment_measurements_legacy").fetchone()[0]
    new_count = db.execute(
        "SELECT COUNT(*) FROM deployment_measurements WHERE deployment_session_id=?",
        (legacy_session_id,),
    ).fetchone()[0]
    if old_count != new_count:
        raise RuntimeError(f"deployment migration count mismatch: {old_count} != {new_count}")
    db.execute("DROP TABLE deployment_measurements_legacy")


def _normalize_operational_metadata(db: sqlite3.Connection) -> None:
    """Correct operational labels without altering any scientific measurement."""
    # Warm-ups prove that execution completed, but the evaluator does not run its
    # cached-vs-uncached correctness comparison until measured repetitions begin.
    db.execute(
        """UPDATE inference_measurements SET correctness_passed=NULL
           WHERE is_warmup=1 AND correctness_passed=0
             AND (raw_json IS NULL OR raw_json NOT LIKE '%\"correctness_passed\"%')"""
    )
    # campaign_state is the authoritative resumability record. Keep the descriptive
    # campaign row synchronized once its declared attempt limit has been reached.
    db.execute(
        """UPDATE campaigns SET status='completed', updated_at=CURRENT_TIMESTAMP
           WHERE status='running' AND campaign_id IN (
             SELECT campaign_id FROM campaign_state WHERE status='completed'
           )"""
    )


def _backfill_campaigns_and_training_runs(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT * FROM experiments ORDER BY id").fetchall()
    for row in rows:
        db.execute(
            """INSERT OR IGNORE INTO campaigns
               (campaign_id, method, phase, attempt_limit, starting_git_commit, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                row["campaign_id"], row["method"], row["phase"],
                5 if row["method"] == "baseline" else (50 if row["phase"] == "campaign" else None),
                row["git_commit"], "completed" if row["finished_at"] else "planned",
            ),
        )
        run_kind = row["phase"] if row["phase"] in {"setup", "baseline"} else "discovery"
        db.execute(
            """INSERT OR IGNORE INTO training_runs
               (experiment_id, run_kind, seed, status, val_bpb, training_tokens_per_second,
                total_training_tokens, peak_training_cuda_memory_mb, parameter_count,
                training_seconds, checkpoint_path, failure_reason, started_at, finished_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["id"], run_kind, row["training_seed"], row["status"], row["val_bpb"],
                row["training_tokens_per_second"], row["total_training_tokens"],
                row["peak_training_cuda_memory_mb"], row["parameter_count"], row["training_seconds"],
                row["checkpoint_path"], row["crash_or_failure_reason"], row["started_at"], row["finished_at"],
            ),
        )


def _archive_legacy_inference(db: sqlite3.Connection) -> None:
    legacy_evaluator = "8e1e094b3420b85431d9b1bae740daa78136ed3e3b10cc66f3d1b3d1be6b228e"
    # An earlier migration archived every populated headline column, including new
    # cached rows. Preserve those audit records but label exact cached duplicates
    # honestly; never confuse them with measurements from the old evaluator.
    db.execute(
        """UPDATE legacy_metrics AS legacy
           SET metric_mode=?, evaluator_sha256=(
             SELECT evaluator_sha256 FROM experiments WHERE id=legacy.experiment_id
           )
           WHERE metric_mode=? AND EXISTS (
             SELECT 1 FROM experiments AS experiment
             WHERE experiment.id=legacy.experiment_id
               AND experiment.inference_metric_mode=?
               AND (
                 (legacy.metric_name='generated_tokens_per_second' AND legacy.metric_value=experiment.inference_tokens_per_second)
                 OR (legacy.metric_name='peak_allocated_cuda_memory' AND legacy.metric_value=experiment.peak_inference_cuda_memory_mb)
                 OR (legacy.metric_name='generation_elapsed' AND legacy.metric_value=experiment.inference_seconds)
               )
           )""",
        (MISCLASSIFIED_CACHED_MODE, LEGACY_MODE, CACHED_MODE),
    )
    rows = db.execute(
        """SELECT id, inference_tokens_per_second, peak_inference_cuda_memory_mb,
                  inference_seconds, sample_path, inference_metric_mode
           FROM experiments WHERE inference_tokens_per_second IS NOT NULL
             AND (inference_metric_mode IS NULL OR inference_metric_mode=?)""",
        (LEGACY_MODE,),
    ).fetchall()
    for row in rows:
        raw = json.dumps({
            "inference_tokens_per_second": row["inference_tokens_per_second"],
            "peak_inference_cuda_memory_mb": row["peak_inference_cuda_memory_mb"],
            "inference_seconds": row["inference_seconds"],
            "sample_path": row["sample_path"],
        }, sort_keys=True)
        for name, value, unit, source in (
            ("generated_tokens_per_second", row["inference_tokens_per_second"], "tokens/second", "inference_tokens_per_second"),
            ("peak_allocated_cuda_memory", row["peak_inference_cuda_memory_mb"], "MiB", "peak_inference_cuda_memory_mb"),
            ("generation_elapsed", row["inference_seconds"], "seconds", "inference_seconds"),
        ):
            db.execute(
                """INSERT OR IGNORE INTO legacy_metrics
                   (experiment_id, metric_mode, metric_name, metric_value, unit, source_column,
                    evaluator_sha256, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (row["id"], LEGACY_MODE, name, value, unit, source, legacy_evaluator, raw),
            )
        if row["inference_metric_mode"] is None:
            db.execute(
                """UPDATE experiments SET inference_metric_mode=?, evaluator_sha256=? WHERE id=?""",
                (LEGACY_MODE, legacy_evaluator, row["id"]),
            )


def record_event(
    db: sqlite3.Connection,
    *,
    event_uuid: str,
    event_type: str,
    campaign_id: str | None = None,
    experiment_id: int | None = None,
    training_run_id: int | None = None,
    status: str | None = None,
    event_time: str | None = None,
    elapsed_seconds: float | None = None,
    payload: dict[str, Any] | None = None,
    source: str = "live",
) -> None:
    db.execute(
        """INSERT OR IGNORE INTO experiment_events
           (event_uuid, campaign_id, experiment_id, training_run_id, event_type, status,
            event_time, elapsed_seconds, payload_json, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (event_uuid, campaign_id, experiment_id, training_run_id, event_type, status,
         event_time or utc_now(), elapsed_seconds, json.dumps(payload or {}, sort_keys=True), source),
    )
    db.commit()


def pareto_front(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = list(rows)
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        dominated = any(
            other is not candidate
            and other["val_bpb"] <= candidate["val_bpb"]
            and other["speed"] >= candidate["speed"]
            and other["memory"] <= candidate["memory"]
            and (
                other["val_bpb"] < candidate["val_bpb"]
                or other["speed"] > candidate["speed"]
                or other["memory"] < candidate["memory"]
            )
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: (row["val_bpb"], -row["speed"], row["memory"]))


def raw_frontier(db: sqlite3.Connection, campaign_id: str | None = None) -> list[dict[str, Any]]:
    clauses = [
        "status='completed'", "valid_result=1", "exclude_from_analysis=0",
        "inference_metric_mode=?", "val_bpb IS NOT NULL",
        "median_cached_decode_tokens_per_second IS NOT NULL",
        "peak_inference_cuda_memory_mb IS NOT NULL",
    ]
    params: list[Any] = [CACHED_MODE]
    if campaign_id:
        clauses.append("campaign_id=?")
        params.append(campaign_id)
    rows = db.execute(
        f"SELECT * FROM experiments WHERE {' AND '.join(clauses)} ORDER BY id", params
    ).fetchall()
    shaped = []
    for row in rows:
        item = dict(row)
        item.update(speed=row["median_cached_decode_tokens_per_second"], memory=row["peak_inference_cuda_memory_mb"])
        shaped.append(item)
    return pareto_front(shaped)


def confirmed_candidates(db: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = db.execute(
        """SELECT e.id, e.campaign_id, e.experiment_number, e.method, e.description,
                  tr.id AS training_run_id, tr.seed, tr.status, tr.val_bpb,
                  im.decode_tokens_per_second, im.peak_allocated_cuda_memory_mb
           FROM experiments e
           JOIN confirmation_runs cr ON cr.candidate_experiment_id=e.id
           JOIN training_runs tr ON tr.id=cr.training_run_id
           LEFT JOIN inference_measurements im ON im.training_run_id=tr.id
                AND im.metric_mode=? AND im.is_warmup=0
           ORDER BY e.id, tr.seed""",
        (CACHED_MODE,),
    ).fetchall()
    grouped_runs: dict[tuple[int, int], list[sqlite3.Row]] = {}
    for row in rows:
        grouped_runs.setdefault((int(row["id"]), int(row["training_run_id"])), []).append(row)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for (experiment_id, _), samples in grouped_runs.items():
        first = samples[0]
        speeds = [float(r["decode_tokens_per_second"]) for r in samples if r["decode_tokens_per_second"] is not None]
        memories = [float(r["peak_allocated_cuda_memory_mb"]) for r in samples if r["peak_allocated_cuda_memory_mb"] is not None]
        grouped.setdefault(experiment_id, []).append({
            **dict(first),
            "seed_speed": statistics.median(speeds) if speeds else None,
            "seed_memory": max(memories) if memories else None,
        })

    # Primary AutoPareto confirmations are intentionally stored as immutable,
    # one-attempt campaigns.  Their parent_experiment_id identifies the frozen
    # discovery recipe; aggregate them alongside the older confirmation_runs
    # representation without rewriting either historical schema.
    primary_rows = db.execute(
        """SELECT parent.id, parent.campaign_id, parent.experiment_number,
                  parent.method, parent.description,
                  run.id AS training_run_id, run.training_seed AS seed,
                  CASE WHEN run.status='completed' AND run.inference_status='compatible'
                       THEN 'completed' ELSE run.status END AS status,
                  run.val_bpb,
                  run.median_cached_decode_tokens_per_second AS seed_speed,
                  run.peak_inference_cuda_memory_mb AS seed_memory
           FROM experiments run
           JOIN experiments parent ON parent.id=run.parent_experiment_id
           WHERE run.phase='confirmation'
             AND run.campaign_id LIKE 'autopareto-primary-%'
           ORDER BY parent.id, run.training_seed"""
    ).fetchall()
    existing = {
        (experiment_id, int(sample["seed"]))
        for experiment_id, samples in grouped.items()
        for sample in samples
    }
    for row in primary_rows:
        key = (int(row["id"]), int(row["seed"]))
        if key in existing:
            continue
        grouped.setdefault(key[0], []).append(dict(row))
        existing.add(key)
    confirmed = []
    required = set(CONFIRMATION_SEEDS)
    for experiment_id, samples in grouped.items():
        valid = [r for r in samples if r["status"] == "completed" and r["val_bpb"] is not None and r["seed_speed"] is not None and r["seed_memory"] is not None]
        if {int(r["seed"]) for r in valid} != required:
            continue
        first = valid[0]
        confirmed.append({
            "id": experiment_id,
            "campaign_id": first["campaign_id"],
            "experiment_number": first["experiment_number"],
            "method": first["method"],
            "description": first["description"],
            "seeds": list(CONFIRMATION_SEEDS),
            "val_bpb": statistics.mean(float(r["val_bpb"]) for r in valid),
            "val_bpb_sample_sd": statistics.stdev(float(r["val_bpb"]) for r in valid),
            "speed": statistics.median(float(r["seed_speed"]) for r in valid),
            "speed_sample_sd": statistics.stdev(float(r["seed_speed"]) for r in valid),
            "memory": max(float(r["seed_memory"]) for r in valid),
        })
    return confirmed


def confirmed_frontier(db: sqlite3.Connection) -> list[dict[str, Any]]:
    return pareto_front(confirmed_candidates(db))


def quality_budget_report(db: sqlite3.Connection) -> dict[str, Any]:
    candidates = confirmed_candidates(db)
    autoresearch = [row for row in candidates if row["method"] == "autoresearch"]
    if not autoresearch:
        return {"reference": None, "budgets": {}}
    reference = min(autoresearch, key=lambda row: row["val_bpb"])
    reports: dict[str, Any] = {}
    for budget in QUALITY_BUDGETS:
        threshold = reference["val_bpb"] * (1 + budget)
        eligible = [row for row in candidates if row["val_bpb"] <= threshold]
        reports[f"{budget * 100:g}%"] = {
            "threshold_val_bpb": threshold,
            "fastest": max(eligible, key=lambda row: row["speed"]) if eligible else None,
            "lowest_memory": min(eligible, key=lambda row: row["memory"]) if eligible else None,
        }
    return {"reference": reference, "budgets": reports}

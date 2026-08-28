"""Runtime-only Phase 1 measurement and milestone infrastructure.

The objects in this module are imported only by the derived execution source.
Frozen model recipe files are not edited.  Step timing is buffered in memory and
written after the training loop so artifact I/O cannot enter the timed workload.
Telemetry runs in a separate sampler thread and never calls CUDA synchronize.
"""

from __future__ import annotations

import hashlib
import atexit
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import threading
import time
from datetime import datetime, timezone
from typing import Any

from event_log import append_event, make_event


COMMON_PHASE1_TOKEN_BUDGET = 201_326_592
STEADY_STATE_START_STEP_INDEX = 11
TELEMETRY_FORMAT = "autopareto_phase1_background_telemetry_v2"
STEP_TIMING_FORMAT = "autopareto_phase1_step_timing_v2"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _proc_cpu_steal_seconds() -> float | None:
    line = _read_text(Path("/proc/stat"))
    if not line:
        return None
    for row in line.splitlines():
        fields = row.split()
        if fields and fields[0] == "cpu":
            # /proc/stat reports USER_HZ ticks: user nice system idle iowait irq
            # softirq steal guest guest_nice.
            if len(fields) > 8:
                try:
                    return int(fields[8]) / float(os.sysconf("SC_CLK_TCK"))
                except (OSError, ValueError):
                    return None
    return None


def _cpu_frequency_mhz() -> float | None:
    values: list[float] = []
    for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
        raw = _read_text(path)
        if raw:
            try:
                values.append(float(raw) / 1000.0)
            except ValueError:
                pass
    if values:
        return statistics.mean(values)
    try:
        import psutil  # type: ignore

        frequencies = [item.current for item in psutil.cpu_freq(percpu=True) or [] if item.current]
        return statistics.mean(frequencies) if frequencies else None
    except (ImportError, OSError, ValueError):
        return None


def _process_scheduling() -> dict[str, Any]:
    result: dict[str, Any] = {}
    sched = _read_text(Path("/proc/self/sched"))
    if sched:
        for line in sched.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip().split()[0] if value.strip() else ""
            if key.strip() in {"se.sum_exec_runtime", "nr_switches", "nr_voluntary_switches", "nr_involuntary_switches"}:
                try:
                    result[key.strip()] = float(value)
                except ValueError:
                    result[key.strip()] = value
    stat = _read_text(Path("/proc/self/status"))
    if stat:
        for line in stat.splitlines():
            if line.startswith(("voluntary_ctxt_switches:", "nonvoluntary_ctxt_switches:")):
                key, value = line.split(":", 1)
                try:
                    result[key] = int(value.strip())
                except ValueError:
                    result[key] = value.strip()
    return result


def _load_nvml() -> tuple[Any | None, str | None]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        return pynvml, None
    except Exception as exc:  # preserve missing NVML as explicit telemetry evidence
        return None, f"NVML unavailable: {type(exc).__name__}: {exc}"


def _nvml_text(value: Any) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def capture_gpu_state() -> dict[str, Any]:
    """Capture non-synchronizing GPU state for run metadata."""
    pynvml, error = _load_nvml()
    if pynvml is None:
        return {"available": False, "telemetry_errors": [error] if error else []}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        result = {
            "available": True,
            "gpu_identity": _nvml_text(pynvml.nvmlDeviceGetName(handle)),
            "gpu_uuid": _nvml_text(pynvml.nvmlDeviceGetUUID(handle)),
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "memory_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
            "pstate": pynvml.nvmlDeviceGetPowerState(handle),
            "power_draw_watts": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        }
        try:
            result["power_limit_watts"] = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except Exception:
            result["power_limit_watts"] = None
        try:
            result["clocks_throttle_reasons"] = int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))
        except Exception:
            result["clocks_throttle_reasons"] = None
        return result
    except Exception as exc:
        return {"available": False, "telemetry_errors": [f"GPU state read failed: {type(exc).__name__}: {exc}"]}


class Phase1TelemetrySampler:
    """Fixed-interval host/NVML sampler outside the optimizer-step critical path."""

    def __init__(self, *, output_path: Path, sample_interval_seconds: float) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("telemetry sample interval must be positive")
        self.output_path = output_path
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml, nvml_error = _load_nvml()
        if nvml_error:
            self.errors.append(nvml_error)
        self._nvml_handle = None
        if self._nvml is not None:
            try:
                self._nvml_handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as exc:
                self.errors.append(f"NVML handle unavailable: {type(exc).__name__}: {exc}")
        self._last_wall_ns: int | None = None
        self._last_process_ns: int | None = None
        self._sampler_thread_cpu_ns = 0
        self._payload: dict[str, Any] | None = None
        atexit.register(self.stop)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry sampler already started")
        self._thread = threading.Thread(target=self._run, name="phase1-telemetry", daemon=True)
        self._thread.start()

    def _gpu_sample(self, row: dict[str, Any]) -> None:
        if self._nvml is not None and self._nvml_handle is not None:
            try:
                nvml = self._nvml
                utilization = nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                memory = nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                row.update({
                    "gpu_identity": _nvml_text(nvml.nvmlDeviceGetName(self._nvml_handle)),
                    "gpu_uuid": _nvml_text(nvml.nvmlDeviceGetUUID(self._nvml_handle)),
                    "sm_clock_mhz": nvml.nvmlDeviceGetClockInfo(self._nvml_handle, nvml.NVML_CLOCK_SM),
                    "memory_clock_mhz": nvml.nvmlDeviceGetClockInfo(self._nvml_handle, nvml.NVML_CLOCK_MEM),
                    "pstate": nvml.nvmlDeviceGetPowerState(self._nvml_handle),
                    "power_draw_watts": nvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0,
                    "temperature_c": nvml.nvmlDeviceGetTemperature(self._nvml_handle, nvml.NVML_TEMPERATURE_GPU),
                    "gpu_utilization_percent": utilization.gpu,
                    "memory_utilization_percent": utilization.memory,
                    "gpu_memory_total_bytes": memory.total,
                    "gpu_memory_used_bytes": memory.used,
                })
                try:
                    row["power_limit_watts"] = nvml.nvmlDeviceGetPowerManagementLimit(self._nvml_handle) / 1000.0
                except Exception:
                    row["power_limit_watts"] = None
                try:
                    row["clocks_throttle_reasons"] = int(
                        nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._nvml_handle)
                    )
                except Exception:
                    row["clocks_throttle_reasons"] = None
            except Exception as exc:
                row.setdefault("telemetry_errors", []).append(
                    f"GPU sample failed: {type(exc).__name__}: {exc}"
                )

        # Deliberately do not import torch or call CUDA runtime allocator APIs
        # here.  Even non-synchronizing allocator queries can initialize or
        # contend with the CUDA runtime from a second thread.  Allocated and
        # reserved memory are recorded by the training/evaluation processes at
        # their explicit measurement boundaries instead.
        row["cuda_runtime_calls_by_sampler"] = False

    def _sample(self) -> dict[str, Any]:
        monotonic_ns = time.perf_counter_ns()
        process_ns = time.process_time_ns()
        row: dict[str, Any] = {
            "monotonic_ns": monotonic_ns,
            "utc_timestamp": utc_now_iso(),
            "cpu_frequency_mhz": _cpu_frequency_mhz(),
            "system_load": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "cpu_steal_time_seconds": _proc_cpu_steal_seconds(),
            "process_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "process_scheduling": _process_scheduling(),
        }
        if self._last_wall_ns is not None and self._last_process_ns is not None:
            wall_delta = monotonic_ns - self._last_wall_ns
            row["process_cpu_utilization_percent"] = (
                (process_ns - self._last_process_ns) / wall_delta * 100.0 if wall_delta > 0 else None
            )
        else:
            row["process_cpu_utilization_percent"] = None
        self._last_wall_ns = monotonic_ns
        self._last_process_ns = process_ns
        self._gpu_sample(row)
        return row

    def _run(self) -> None:
        next_sample = time.perf_counter()
        while True:
            thread_started_ns = time.thread_time_ns()
            try:
                self.samples.append(self._sample())
            except Exception as exc:
                self.errors.append(f"sampler iteration failed: {type(exc).__name__}: {exc}")
            self._sampler_thread_cpu_ns += time.thread_time_ns() - thread_started_ns
            next_sample += self.sample_interval_seconds
            if self._stop.wait(max(0.0, next_sample - time.perf_counter())):
                break

    def stop(self) -> dict[str, Any]:
        if self._payload is not None:
            return self._payload
        if self._thread is None:
            self._payload = {
                "format": TELEMETRY_FORMAT,
                "sampling_interval_seconds": self.sample_interval_seconds,
                "sampling_outside_timed_critical_path": True,
                "cuda_synchronization_called_by_sampler": False,
                "cuda_runtime_calls_by_sampler": False,
                "samples": [],
                "telemetry_errors": ["telemetry sampler was never started"],
            }
            return self._payload
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 2.0))
        if self._thread.is_alive():
            self.errors.append("telemetry sampler did not stop within bounded join timeout")
        payload = {
            "format": TELEMETRY_FORMAT,
            "sampling_interval_seconds": self.sample_interval_seconds,
            "telemetry_mode": "enabled",
            "sampling_outside_timed_critical_path": True,
            "cuda_synchronization_called_by_sampler": False,
            "cuda_runtime_calls_by_sampler": False,
            "sampler_thread_cpu_seconds": self._sampler_thread_cpu_ns / 1_000_000_000.0,
            "telemetry_overhead_estimate_method": "sampler thread CPU time; CUDA runtime and event instrumentation are disabled",
            "started_monotonic_ns": self.samples[0]["monotonic_ns"] if self.samples else None,
            "ended_monotonic_ns": self.samples[-1]["monotonic_ns"] if self.samples else None,
            "telemetry_errors": self.errors,
            "samples": self.samples,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.output_path, payload)
        self._payload = payload
        return payload


class Phase1StepTimingRecorder:
    """Authoritative synchronized monotonic per-optimizer-step accounting."""

    def __init__(
        self,
        *,
        token_budget: int,
        tokens_per_optimizer_step: int,
        step_timing_path: Path,
        summary_path: Path,
    ) -> None:
        if token_budget <= 0 or tokens_per_optimizer_step <= 0:
            raise ValueError("token budget and tokens per optimizer step must be positive")
        if token_budget % tokens_per_optimizer_step:
            raise ValueError("token budget must be divisible by tokens per optimizer step")
        self.token_budget = int(token_budget)
        self.tokens_per_optimizer_step = int(tokens_per_optimizer_step)
        self.step_timing_path = step_timing_path
        self.summary_path = summary_path
        self.records: list[dict[str, Any]] = []
        self._all_ns = 0
        self._steady_ns = 0
        self._interval_start_ns: int | None = None
        self._interval_end_ns: int | None = None

    @property
    def cumulative_scientific_step_wall_seconds(self) -> float:
        return self._all_ns / 1_000_000_000.0

    @property
    def cumulative_steady_state_seconds(self) -> float:
        return self._steady_ns / 1_000_000_000.0

    def record_step(
        self,
        *,
        zero_based_step_index: int,
        step_start_monotonic_ns: int,
        step_end_monotonic_ns: int,
        cuda_stream_step_seconds: float | None,
    ) -> None:
        if zero_based_step_index != len(self.records):
            raise RuntimeError("step timing records are not contiguous from zero")
        duration_ns = int(step_end_monotonic_ns) - int(step_start_monotonic_ns)
        if duration_ns < 0:
            raise RuntimeError("monotonic step duration is negative")
        if self._interval_start_ns is None:
            self._interval_start_ns = int(step_start_monotonic_ns)
        self._interval_end_ns = int(step_end_monotonic_ns)
        self._all_ns += duration_ns
        if zero_based_step_index >= STEADY_STATE_START_STEP_INDEX:
            self._steady_ns += duration_ns
        step_count = zero_based_step_index + 1
        record = {
            "zero_based_step_index": int(zero_based_step_index),
            "optimizer_step_number": step_count,
            "tokens_processed": self.tokens_per_optimizer_step,
            "step_start_monotonic_ns": int(step_start_monotonic_ns),
            "step_end_monotonic_ns": int(step_end_monotonic_ns),
            "step_wall_seconds": duration_ns / 1_000_000_000.0,
            "cuda_stream_step_seconds": (
                float(cuda_stream_step_seconds) if cuda_stream_step_seconds is not None else None
            ),
            "cumulative_scientific_tokens": step_count * self.tokens_per_optimizer_step,
            "cumulative_all_step_seconds": self.cumulative_scientific_step_wall_seconds,
            "cumulative_steady_state_tokens": max(0, step_count - STEADY_STATE_START_STEP_INDEX)
            * self.tokens_per_optimizer_step,
            "cumulative_steady_state_seconds": self.cumulative_steady_state_seconds,
        }
        self.records.append(record)

    def summary(self, *, final_step_count: int) -> dict[str, Any]:
        steady_state_seconds = self.cumulative_steady_state_seconds
        steady_state_tokens = max(0, int(final_step_count) - STEADY_STATE_START_STEP_INDEX) * self.tokens_per_optimizer_step
        steady_state_tps = steady_state_tokens / steady_state_seconds if steady_state_seconds > 0 else None
        common_excluded_tokens = STEADY_STATE_START_STEP_INDEX * 524_288
        if common_excluded_tokens % self.tokens_per_optimizer_step:
            raise RuntimeError("common excluded token boundary is not divisible by recipe batch")
        common_start_step_index = common_excluded_tokens // self.tokens_per_optimizer_step
        common_seconds = sum(
            row["step_wall_seconds"]
            for row in self.records
            if row["zero_based_step_index"] >= common_start_step_index
        )
        common_tokens = max(0, int(final_step_count) * self.tokens_per_optimizer_step - common_excluded_tokens)
        common_tps = common_tokens / common_seconds if common_seconds > 0 else None
        return {
            "format": STEP_TIMING_FORMAT,
            "common_token_budget": COMMON_PHASE1_TOKEN_BUDGET,
            "run_token_budget": self.token_budget,
            "steady_state_start_step_index": STEADY_STATE_START_STEP_INDEX,
            "final_step_count": int(final_step_count),
            "tokens_per_optimizer_step": self.tokens_per_optimizer_step,
            "total_scientific_step_wall_seconds": self.cumulative_scientific_step_wall_seconds,
            "steady_state_training_seconds": steady_state_seconds,
            "steady_state_tokens": steady_state_tokens,
            "steady_state_training_tokens_per_second": steady_state_tps,
            "steady_state_equivalent_seconds": (
                COMMON_PHASE1_TOKEN_BUDGET / steady_state_tps
                if steady_state_tps and steady_state_tps > 0
                else None
            ),
            "common_token_boundary_start_step_index": common_start_step_index,
            "common_token_boundary_excluded_tokens": common_excluded_tokens,
            "common_token_boundary_seconds": common_seconds,
            "common_token_boundary_tokens": common_tokens,
            "common_token_boundary_tokens_per_second": common_tps,
            "common_token_boundary_equivalent_seconds": (
                COMMON_PHASE1_TOKEN_BUDGET / common_tps
                if common_tps and common_tps > 0
                else None
            ),
            "scientific_interval_seconds": (
                (self._interval_end_ns - self._interval_start_ns) / 1_000_000_000.0
                if self._interval_start_ns is not None and self._interval_end_ns is not None
                else None
            ),
            "step_timing_path": str(self.step_timing_path),
            "timing_authority": "sum of per-step synchronized monotonic perf_counter_ns durations",
            "cuda_event_profile": "diagnostic-only; disabled in scientific Phase 1 timing",
            "scientific_interval_note": (
                "Continuous monotonic interval from the first scientific optimizer-step start "
                "to the final optimizer-step end; it may include logging, between-step bookkeeping, "
                "and milestone/manifest writes between steps and must not replace the per-step sum."
            ),
            "common_excluded_token_boundary": STEADY_STATE_START_STEP_INDEX * self.tokens_per_optimizer_step,
            "common_token_boundary_step_count": (
                STEADY_STATE_START_STEP_INDEX
                if self.tokens_per_optimizer_step == 524_288
                else (STEADY_STATE_START_STEP_INDEX * 524_288) // self.tokens_per_optimizer_step
            ),
        }

    def write(self, *, final_step_count: int) -> dict[str, Any]:
        if int(final_step_count) != len(self.records):
            raise RuntimeError(
                f"step timing count mismatch: records={len(self.records)} final_step_count={final_step_count}"
            )
        expected_tokens = int(final_step_count) * self.tokens_per_optimizer_step
        if expected_tokens != self.token_budget:
            raise RuntimeError("step timing final token count does not equal the run token budget")
        self.step_timing_path.parent.mkdir(parents=True, exist_ok=True)
        with self.step_timing_path.open("w") as stream:
            for row in self.records:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        summary = self.summary(final_step_count=final_step_count)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Phase1MilestoneRecorder:
    """Save immutable checkpoints at preregistered token boundaries."""

    def __init__(
        self,
        *,
        token_budget: int,
        total_batch_size: int,
        seed: int,
        source_sha256: str,
        milestone_dir: Path,
        final_checkpoint: Path,
    ) -> None:
        if token_budget <= 0 or token_budget % total_batch_size:
            raise ValueError("token budget must be divisible by total batch size")
        self.token_budget = int(token_budget)
        self.total_batch_size = int(total_batch_size)
        self.target_steps = self.token_budget // self.total_batch_size
        self.seed = int(seed)
        self.source_sha256 = source_sha256
        self.milestone_dir = milestone_dir
        self.final_checkpoint = final_checkpoint
        self.manifest_path = milestone_dir / "manifest.json"
        self.milestone_dir.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        if self.manifest_path.exists():
            prior = json.loads(self.manifest_path.read_text())
            if prior.get("source_sha256") != source_sha256 or prior.get("seed") != seed:
                raise RuntimeError("existing Phase 1 milestone manifest belongs to another run")
            self.records = {str(row["token_count"]): row for row in prior.get("checkpoints", [])}

    def _path_for(self, token_count: int, step: int) -> Path:
        if token_count == self.token_budget:
            return self.final_checkpoint
        return self.milestone_dir / f"step-{step:06d}.pt"

    def _write_checkpoint(self, destination: Path, model, config) -> tuple[int, str]:
        from protected.inference_adapter import save_training_checkpoint

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}-{time.time_ns()}")
        save_training_checkpoint(
            temporary, model, config, self.seed, f"phase1-source-{self.source_sha256}"
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if destination.exists():
            temporary.unlink()
            raise RuntimeError(f"refusing to overwrite immutable checkpoint: {destination}")
        os.replace(temporary, destination)
        return destination.stat().st_size, sha256(destination)

    def record_if_due(
        self,
        *,
        step: int,
        measured_training_seconds: float,
        cumulative_scientific_step_wall_seconds: float,
        cumulative_steady_state_seconds: float,
        model,
        config,
    ) -> None:
        token_count = int(step) * self.total_batch_size
        milestones = {
            self.token_budget // 4,
            self.token_budget // 2,
            (self.token_budget * 3) // 4,
            self.token_budget,
        }
        if token_count not in milestones:
            return
        key = str(token_count)
        if key in self.records:
            return
        destination = self._path_for(token_count, int(step))
        size, digest = self._write_checkpoint(destination, model, config)
        record = {
            "token_count": token_count,
            "optimizer_step": int(step),
            "historical_steady_state_training_seconds": float(measured_training_seconds),
            "total_scientific_step_wall_seconds": float(cumulative_scientific_step_wall_seconds),
            "cumulative_steady_state_seconds": float(cumulative_steady_state_seconds),
            "checkpoint_path": str(destination),
            "checkpoint_bytes": size,
            "checkpoint_sha256": digest,
        }
        self.records[key] = record
        payload = {
            "format": "autopareto_phase1_milestones_v1",
            "token_budget": self.token_budget,
            "total_batch_size": self.total_batch_size,
            "device_batch_size": 64,
            "sequence_length": 2048,
            "seed": self.seed,
            "source_sha256": self.source_sha256,
            "checkpoints": [self.records[k] for k in sorted(self.records, key=int)],
        }
        _atomic_json(self.manifest_path, payload)
        event_path = os.environ.get("AUTOPARETO_PHASE1_EVENT_LOG")
        if event_path:
            append_event(
                Path(event_path),
                make_event(
                    "phase1_milestone_saved",
                    status="running",
                    elapsed_seconds=float(measured_training_seconds),
                    payload=record,
                ),
            )

"""Frozen study launcher surface.

The command builder is deliberately side-effect free unless an operator explicitly
passes ``--execute`` to a deferred command.  The readiness suite exercises command
construction and guards, not model training.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import secrets
import shutil
import subprocess
import sys
import tempfile

from .batching import ALLOWED_TOTAL_BATCH_SIZES, COMMON_TOKEN_QUANTUM
from .calibration import CalibrationRun, calibration_command, finalize_calibration
from .contracts import ContractId, ResourceContract, json_contract, parse_contract
from .crossover import CrossoverPlan
from .provenance import sha256_file
from .benchmark_policy import PROTOCOL_VERSION
from .scheduling import schedule_payload


STUDY_ROOT = pathlib.Path(__file__).resolve().parent
STABLE_INFERENCE_MODULE = "study.contracts_2026.inference_benchmark"
LEGACY_INFERENCE_MARKERS = ("inference_benchmark.py --legacy", "legacy_fixed", "--legacy-diagnostic")
STARTING_SOURCE_SHA256 = "b2fb8788f02517af8bc2fcfa059fb799a6974c9050d00e322168ec414fac4fab"
INVALIDATED_PRE_REPAIR_N_STAR = 93_847_552


def assert_study_inference_command(command: list[str]) -> None:
    rendered = " ".join(command)
    if any(marker in rendered for marker in LEGACY_INFERENCE_MARKERS):
        raise ValueError("legacy inference benchmark is forbidden in the study runner")
    if STABLE_INFERENCE_MODULE not in rendered:
        raise ValueError("study attempts must invoke the frozen stabilized benchmark module")


def training_command(
    *,
    context: pathlib.Path,
    contract: ResourceContract,
    seed: int,
    candidate: str,
    output: pathlib.Path,
    baseline_calibration: bool = False,
) -> list[str]:
    if contract.contract_id is ContractId.N and contract.n_star is None:
        raise ValueError("cannot build fixed-token training command before N_STAR is frozen")
    command = [
        sys.executable,
        "-m",
        "study.contracts_2026.train_launcher",
        "--context",
        str(context),
        "--candidate",
        candidate,
        "--contract",
        contract.contract_id.value,
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]
    if contract.n_star is not None:
        command.extend(["--n-star", str(contract.n_star)])
    if baseline_calibration:
        command.append("--baseline-calibration")
    return command


def inference_command(*, context: pathlib.Path, checkpoint: str, output: pathlib.Path, candidate: str = "train.py") -> list[str]:
    command = [
        sys.executable,
        "-m",
        STABLE_INFERENCE_MODULE,
        "--protocol",
        PROTOCOL_VERSION,
        "--context",
        str(context),
        "--candidate",
        candidate,
        "--checkpoint",
        checkpoint,
        "--output",
        str(output),
    ]
    assert_study_inference_command(command)
    return command


def pair_plan(n_star: int | None = None) -> dict[str, object]:
    if n_star is not None and n_star % COMMON_TOKEN_QUANTUM:
        raise ValueError("N_STAR is not divisible by every allowed TOTAL_BATCH_SIZE")
    if n_star == INVALIDATED_PRE_REPAIR_N_STAR:
        raise ValueError("the pre-repair N_STAR is invalidated; recalibration is required")
    return {
        "pair_id": "PAIR_A",
        "attempts": 25,
        "seed": 3,
        "starting_source_commit": "5ea452c9d86a43ad4bebd507c5872f18249627ce",
        "starting_source_sha256": STARTING_SOURCE_SHA256,
        "baseline_experiment_id": "pair-a-seed3-baseline-pending",
        "agent_model": "gpt-5.6-sol",
        "reasoning_level": "high",
        "allowed_total_batch_sizes": list(ALLOWED_TOTAL_BATCH_SIZES),
        "search_objectives": [
            "minimize validation BPB",
            "maximize canonical BF16 batch-1 cached greedy decode output-token velocity",
            "minimize peak allocated inference memory",
        ],
        "contracts": {
            "T": json_contract(ResourceContract(ContractId.T)),
            "N": json_contract(ResourceContract(ContractId.N, time_seconds=None, n_star=n_star)),
        },
        "history_paths": {
            "T": "runs/pair-a-fixed-time-seed3/history.jsonl",
            "N": "runs/pair-a-fixed-tokens-seed3/history.jsonl",
        },
        "execution_schedule": schedule_payload(),
        "downstream_metrics": "sealed_post_search",
    }


def baseline_plan(*, source: pathlib.Path, seed: int, output: pathlib.Path) -> dict[str, object]:
    if source.name != "train.py":
        raise ValueError("baseline source must be train.py")
    if not source.is_file() or sha256_file(source) != STARTING_SOURCE_SHA256:
        raise ValueError("baseline source does not match the registered starting source hash")
    return {
        "status": "deferred",
        "purpose": "generate and register the common fresh-seed baseline before calibration",
        "seed": seed,
        "source": str(source),
        "source_sha256": STARTING_SOURCE_SHA256,
        "training": training_command(
            context=source.parent,
            contract=ResourceContract(ContractId.T),
            seed=seed,
            candidate="train.py",
            output=output,
            baseline_calibration=True,
        ),
        "post_training_inference": "run the registered Design-B benchmark once and persist all request/block timings plus the 95% velocity interval",
        "registration_output": "runs/pair-a-seed3-baseline-registration.json",
        "cost": "one full baseline training run plus one stabilized inference measurement on the fixed A40 environment",
        "executed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m study.contracts_2026.runner")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--n-star", type=int)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--contract", required=True, choices=("T",))
    calibrate.add_argument("--repetitions", type=int, default=3)
    calibrate.add_argument("--seed", type=int, required=True)
    calibrate.add_argument("--baseline-source", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--execute", action="store_true")
    plan = sub.add_parser("pair-plan")
    plan.add_argument("--n-star", type=int)
    baseline = sub.add_parser("baseline-plan")
    baseline.add_argument("--source", type=pathlib.Path, required=True)
    baseline.add_argument("--seed", type=int, required=True)
    baseline.add_argument("--output", type=pathlib.Path, required=True)
    crossover = sub.add_parser("crossover-plan")
    crossover.add_argument("--candidate-source", required=True)
    crossover.add_argument("--seed", type=int, required=True)
    crossover.add_argument("--n-star", type=int, required=True)
    crossover.add_argument("--post-search-only", action="store_true")
    launch = sub.add_parser("run-campaign")
    launch.add_argument("--campaign-id", required=True)
    launch.add_argument("--contract", choices=("T", "N"), required=True)
    launch.add_argument("--seed", type=int, required=True)
    launch.add_argument("--baseline-experiment-id", required=True)
    launch.add_argument("--program", required=True)
    launch.add_argument("--attempts", type=int, default=25)
    launch.add_argument("--n-star", type=int)
    launch.add_argument("--execute", action="store_true")
    return parser


def execute_calibration(*, seed: int, baseline_source: pathlib.Path, output: pathlib.Path) -> dict[str, object]:
    """Run three identical baseline invocations and freeze the computed choice."""
    if os.environ.get("AUTOPARETO_STUDY_ALLOW_EXECUTION") != "1":
        raise RuntimeError("calibration execution requires AUTOPARETO_STUDY_ALLOW_EXECUTION=1")
    if baseline_source.name != "train.py" or not baseline_source.is_file() or sha256_file(baseline_source) != STARTING_SOURCE_SHA256:
        raise ValueError("baseline source must match the registered train.py SHA256")
    runs = []
    for run_index in range(1, 4):
        run_output = output.with_name(f"{output.stem}.run{run_index}{output.suffix or '.json'}")
        attempt_root_base = pathlib.Path(
            os.environ.get("AUTOPARETO_CANDIDATE_ATTEMPT_ROOT", "/opt/autopareto-candidate-attempts")
        ).resolve()
        attempt_root_base.mkdir(mode=0o711, parents=True, exist_ok=True)
        attempt_root = attempt_root_base / f"calibration-{run_index}-{secrets.token_hex(12)}"
        attempt_root.mkdir(mode=0o711)
        context_root = pathlib.Path(
            os.environ.get("AUTOPARETO_CANDIDATE_CONTEXT_ROOT", "/var/lib/autopareto/candidate_contexts")
        ).resolve()
        context_root.mkdir(mode=0o755, parents=True, exist_ok=True)
        context = pathlib.Path(tempfile.mkdtemp(prefix=f"calibration-{run_index}-", dir=context_root))
        shutil.copy2(baseline_source, context / "train.py")
        from .remote_runner import _candidate_prepare_source

        protected_prepare = baseline_source.parent / "prepare.py"
        (context / "prepare.py").write_text(_candidate_prepare_source(protected_prepare))
        shutil.copy2(STUDY_ROOT / "study_runtime.py", context / "study_runtime.py")
        context.chmod(0o555)
        for path in context.iterdir():
            path.chmod(0o444)
        internal_output_dir = attempt_root / "candidate-output"
        internal_output_dir.mkdir(mode=0o700)
        internal_output = internal_output_dir / "result.json"
        command = training_command(
            context=context,
            contract=ResourceContract(ContractId.T),
            seed=seed,
            candidate="train.py",
            output=internal_output,
            baseline_calibration=True,
        ) + ["--execute"]
        completed = subprocess.run(
            command,
            cwd=STUDY_ROOT.parents[1],
            env={
                **os.environ,
                "AUTOPARETO_REMOTE_SANDBOX": "1",
                "AUTOPARETO_TRUSTED_ROOT": os.environ.get(
                    "AUTOPARETO_TRUSTED_ROOT", str(STUDY_ROOT.parents[1])
                ),
                "AUTOPARETO_TRUSTED_STATE_ROOT": os.environ.get(
                    "AUTOPARETO_TRUSTED_STATE_ROOT", "/var/lib/autopareto/pair_a_state"
                ),
                "AUTOPARETO_CANDIDATE_ROOT": os.environ.get(
                    "AUTOPARETO_CANDIDATE_ROOT", "/var/lib/autopareto/candidate_attempts"
                ),
                "AUTOPARETO_CANDIDATE_CONTEXT_ROOT": os.environ.get(
                    "AUTOPARETO_CANDIDATE_CONTEXT_ROOT", "/opt/autopareto_candidate_contexts"
                ),
                "AUTOPARETO_CANDIDATE_RUNTIME_ROOT": os.environ.get(
                    "AUTOPARETO_CANDIDATE_RUNTIME_ROOT", "/opt/autopareto_candidate_runtime"
                ),
                "AUTOPARETO_CANDIDATE_INPUT_ROOT": os.environ.get(
                    "AUTOPARETO_CANDIDATE_INPUT_ROOT", "/opt/autopareto_candidate_inputs/per_attempts"
                ),
                "AUTOPARETO_REGISTERED_CACHE": os.environ.get(
                    "AUTOPARETO_REGISTERED_CACHE", "/var/cache/autopareto/pair_a_registered_cache"
                ),
            },
            check=False,
        )
        if completed.returncode != 0 or not internal_output.exists():
            runs.append(CalibrationRun(run_index, 0.0, 0, 0, None, None, "baseline launcher failed"))
            continue
        payload = json.loads(internal_output.read_text())
        run_output.parent.mkdir(parents=True, exist_ok=True)
        run_output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        # Checkpoint and run artifacts are persisted in the root-only trusted
        # state tree before the disposable attempt directory can be removed.
        trusted_state_root = pathlib.Path(
            os.environ.get("AUTOPARETO_TRUSTED_STATE_ROOT", "/var/lib/autopareto/pair_a_state")
        ).resolve()
        trusted_run = trusted_state_root / "calibration" / f"run{run_index}"
        trusted_run.mkdir(mode=0o700, parents=True, exist_ok=True)
        trusted_artifacts: dict[str, str] = {}
        for artifact in (
            internal_output,
            internal_output.with_name(internal_output.name + ".worker.json"),
            pathlib.Path(str(payload.get("checkpoint", ""))),
            internal_output.with_name(internal_output.stem + ".inference.json"),
        ):
            if artifact.is_file() and artifact.resolve().is_relative_to(attempt_root):
                destination = trusted_run / artifact.name
                shutil.copy2(artifact, destination)
                destination.chmod(0o400)
                trusted_artifacts[artifact.name] = str(destination)
        payload["trusted_persistence"] = {
            "root": str(trusted_state_root),
            "run": str(trusted_run),
            "artifacts": trusted_artifacts,
        }
        run_output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        runs.append(CalibrationRun(
            run_index=run_index,
            measured_training_seconds=float(payload.get("measured_training_seconds", 0.0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            optimizer_updates=int(payload.get("optimizer_updates", 0)),
            training_throughput_tokens_per_second=payload.get("training_throughput_tokens_per_second"),
            validation_bpb=payload.get("validation_bpb"),
            error=payload.get("error") if payload.get("status") != "completed" else None,
        ))
    choice = finalize_calibration(runs)
    payload = {"status": "frozen", "choice": choice.as_dict(), "runs": [run.__dict__ for run in runs]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        print(json.dumps({"allowed_total_batch_sizes": ALLOWED_TOTAL_BATCH_SIZES, "token_quantum": COMMON_TOKEN_QUANTUM, "pair": pair_plan(args.n_star)}, indent=2))
        return 0
    if args.command == "pair-plan":
        print(json.dumps(pair_plan(args.n_star), indent=2))
        return 0
    if args.command == "baseline-plan":
        print(json.dumps(baseline_plan(source=args.source, seed=args.seed, output=args.output), indent=2))
        return 0
    if args.command == "crossover-plan":
        if not args.post_search_only:
            raise SystemExit("crossover is post-search only")
        print(json.dumps(CrossoverPlan(args.candidate_source, args.seed, args.n_star).as_dict(), indent=2))
        return 0
    if args.command == "run-campaign":
        if args.attempts != 25:
            raise SystemExit("the instrument is frozen to exactly 25 attempts per interleaved campaign")
        contract = parse_contract(args.contract, args.n_star)
        if args.contract == "N" and args.n_star == INVALIDATED_PRE_REPAIR_N_STAR:
            raise SystemExit("the pre-repair N_STAR is invalidated; recalibration is required")
        if args.contract == "N" and contract.n_star is None:
            raise SystemExit("fixed-token campaign requires the frozen N_STAR")
        plan = {
            "status": "deferred" if not args.execute else "execution_guarded",
            "campaign_id": args.campaign_id,
            "contract": json_contract(contract),
            "seed": args.seed,
            "baseline_experiment_id": args.baseline_experiment_id,
            "program": args.program,
            "attempts": args.attempts,
            "execution": "no campaign is launched by the Phase 0.5 checkout",
        }
        print(json.dumps(plan, indent=2))
        if args.execute:
            raise SystemExit("campaign execution is intentionally disabled in Phase 0.5")
        return 0
    if args.command == "calibrate":
        if args.repetitions != 3:
            raise SystemExit("calibration is frozen to exactly three independent baseline runs")
        command = calibration_command(seed=args.seed, baseline_source=args.baseline_source, output=args.output)
        if not args.execute:
            print(json.dumps({"status": "deferred", "command": command, "reason": "no expensive calibration was launched"}, indent=2))
            return 0
        payload = execute_calibration(
            seed=args.seed,
            baseline_source=pathlib.Path(args.baseline_source),
            output=pathlib.Path(args.output),
        )
        print(json.dumps(payload, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

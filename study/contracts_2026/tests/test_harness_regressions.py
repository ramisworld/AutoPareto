"""Regression tests for the 2026-09 Pair-A checkpointing defect.

In the paper-era Pair-A fixed-time arm, a candidate whose own training loop
returned before the protected stop was re-executed by the worker; the rebuilt,
untrained model was checkpointed and scored (validation BPB 3.2159). These
tests run the real protected worker end to end on CPU with a tiny candidate and
a fake trusted parent speaking the event/control protocol.
"""

import argparse
import gc
import json
import os
import pathlib
import tempfile
import threading
import unittest
from unittest import mock

from study.contracts_2026 import candidate_process, remote_runner, train_launcher
from study.contracts_2026.remote_runner import REMOTE_RESULT_SCHEMA


ROOT = pathlib.Path(__file__).resolve().parents[1]
WARMUP_UPDATES = 11

CANDIDATE_TEMPLATE = """\
import torch
import torch.nn as nn
from study_runtime import contract_progress


class TinyConfig:
    def __init__(self):
        self.vocab_size = 16
        self.n_embd = 4


class Tiny(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

    def forward(self, idx, targets):
        return self.emb(idx).float().pow(2).mean()


# Setup: tokenizer, model, optimizer, dataloader
torch.manual_seed(0)
config = TinyConfig()
model = Tiny(config)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.zeros(16, 2048, dtype=torch.long)
y = torch.zeros(16, 2048, dtype=torch.long)
step = 0
total_training_time = 0.0
for _ in range({updates}):
    loss = model(x, y)
    loss.backward()
    contract_progress("T", total_training_time, (step + 1) * 32768)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step += {step_increment}
"""


def _fake_parent(event_read: int, control_write: int, stop_at: int, log: list[dict]) -> None:
    """Answer the worker like the trusted parent: warm-up stop, then STOP at `stop_at`."""
    events = open(event_read, "r", buffering=1, encoding="utf-8")
    control = open(control_write, "w", buffering=1, encoding="utf-8")
    phase = -1
    try:
        for line in events:
            event = json.loads(line)
            log.append(event)
            kind = event.get("kind")
            if kind == "boundary_ready":
                phase += 1
                control.write("START 0\n")
            elif kind == "optimizer_update_completed":
                updates = int(event["optimizer_updates"])
                if phase == 0 and updates == WARMUP_UPDATES:
                    control.write("WARMUP_STOP\n")
                elif phase >= 1 and updates == stop_at:
                    control.write("STOP\n")
                else:
                    control.write("CONTINUE\n")
            control.flush()
    except (OSError, ValueError):
        pass
    finally:
        for stream in (events, control):
            try:
                stream.close()
            except OSError:
                pass


class HarnessRegressionTests(unittest.TestCase):
    def _run_worker(self, *, updates: int, stop_at: int, step_increment: int = 1):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        context = pathlib.Path(tmp.name) / "context"
        context.mkdir()
        (context / "train.py").write_text(
            CANDIDATE_TEMPLATE.format(updates=updates, step_increment=step_increment)
        )
        (context / "prepare.py").write_text("def evaluate_bpb(*args, **kwargs):\n    return 0.0\n")
        (context / "study_runtime.py").write_text((ROOT / "study_runtime.py").read_text())
        event_read, event_write = os.pipe()
        control_read, control_write = os.pipe()
        # The worker wraps its descriptors in file objects that close them when
        # finalized, so it must be their only owner (no double close on reuse).
        worker_event, worker_control = os.dup(event_write), os.dup(control_read)
        os.close(event_write)
        os.close(control_read)
        log: list[dict] = []
        parent = threading.Thread(
            target=_fake_parent, args=(event_read, control_write, stop_at, log), daemon=True
        )
        parent.start()
        args = argparse.Namespace(
            context=context,
            candidate="train.py",
            contract="T",
            n_star=None,
            event_fd=worker_event,
            control_fd=worker_control,
            seed=3,
            warmup_only=False,
            baseline_calibration=False,
            baseline_smoke=False,
            worker_output=str(pathlib.Path(tmp.name) / "worker.json"),
            checkpoint=str(pathlib.Path(tmp.name) / "final.pt"),
        )
        error = None
        with mock.patch("torch.cuda.synchronize", lambda *a, **k: None):
            try:
                returncode = candidate_process.execute(args)
            except RuntimeError as exc:
                returncode, error = None, str(exc)
        gc.collect()  # finalize the worker's stream objects, closing its descriptors
        parent.join(timeout=10)
        output = pathlib.Path(args.worker_output)
        payload = json.loads(output.read_text()) if output.exists() else None
        return returncode, payload, pathlib.Path(args.checkpoint), error, log

    def test_early_return_after_warmup_is_candidate_failure_not_retrained(self):
        # The 2026-09 path: warm-up completes, then the candidate's own loop
        # ends before the protected 300 s stop (here: the parent would stop at 20).
        returncode, payload, checkpoint, error, _log = self._run_worker(updates=14, stop_at=20)
        self.assertIsNone(error)
        self.assertEqual(returncode, 1)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["failure_origin"], "candidate")
        self.assertIn("returned before the protected stop", payload["error"])
        self.assertFalse(checkpoint.exists(), "an untrained or partial model must never be checkpointed")

    def test_early_return_during_warmup_is_candidate_failure(self):
        returncode, payload, checkpoint, error, _log = self._run_worker(updates=3, stop_at=20)
        self.assertIsNone(error)
        self.assertEqual(returncode, 1)
        self.assertEqual(payload["failure_origin"], "candidate")
        self.assertFalse(checkpoint.exists())

    def test_protected_stop_completes_and_checkpoints_the_trained_model(self):
        returncode, payload, checkpoint, error, log = self._run_worker(updates=1000, stop_at=5)
        self.assertIsNone(error)
        self.assertEqual(returncode, 0)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["observed"]["optimizer_updates"], 5)
        self.assertTrue(checkpoint.exists())
        boundaries = [event for event in log if event.get("kind") == "boundary_ready"]
        self.assertEqual(len(boundaries), 2, "exactly one warm-up and one scientific boundary")

    def test_candidate_step_bookkeeping_does_not_gate_a_trained_checkpoint(self):
        # Candidates may count micro-steps or carry a counter across the warm-up;
        # only the protected stop decides whether a trained model is checkpointed.
        returncode, payload, checkpoint, error, _log = self._run_worker(
            updates=1000, stop_at=5, step_increment=0
        )
        self.assertIsNone(error)
        self.assertEqual(returncode, 0)
        self.assertEqual(payload["observed"]["optimizer_updates"], 5)
        self.assertTrue(checkpoint.exists())

    def test_remote_runner_records_early_exit_as_scientific_failure(self):
        payload = {
            "status": "error",
            "failure_origin": "candidate",
            "error": "CandidateExitedBeforeStop: candidate training body returned before the protected stop",
        }
        metadata = {"campaign_id": "opus-fixed-time-seed3", "contract_id": "T", "attempt_number": 1}
        result = remote_runner._classify_worker_failure(payload, metadata)
        self.assertEqual(result["schema"], REMOTE_RESULT_SCHEMA)
        self.assertEqual(result["status"], "scientific_failure")

    def test_launcher_keeps_the_worker_receipt_so_the_failure_is_scientific(self):
        # The parent launcher used to overwrite the worker's "candidate" receipt
        # with "unknown", turning a candidate failure into a campaign halt.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            context = root / "context"
            context.mkdir()
            (context / "train.py").write_text(
                "from study_runtime import contract_progress\n"
                "# Setup: tokenizer, model, optimizer, dataloader\n"
                "for _ in range(0):\n"
                "    contract_progress('T', 0.0, 0)\n"
            )
            (context / "prepare.py").write_text("def evaluate_bpb(*args, **kwargs):\n    return 0.0\n")
            (context / "study_runtime.py").write_text((ROOT / "study_runtime.py").read_text())
            output = root / "out" / "result.json"
            args = train_launcher.build_parser().parse_args([
                "--context", str(context), "--candidate", "train.py", "--contract", "T", "--seed", "3",
                "--output", str(output), "--execute", "--paired-baseline-checkpoint", str(root / "none.pt"),
            ])
            with mock.patch.dict(os.environ, {"AUTOPARETO_STUDY_ALLOW_EXECUTION": "1"}):
                os.environ.pop("AUTOPARETO_REMOTE_SANDBOX", None)
                with self.assertRaises(RuntimeError):
                    train_launcher.execute_candidate(args)
            # Finalize the launcher's pipe objects before later tests reuse the descriptors.
            gc.collect()
            receipt = json.loads(output.with_name(output.name + ".worker.json").read_text())
        self.assertEqual(receipt["failure_origin"], "candidate")
        self.assertIn("returned before the protected stop", receipt["error"])
        metadata = {"campaign_id": "pair-a-fixed-time-seed3", "contract_id": "T", "attempt_number": 1}
        self.assertEqual(remote_runner._classify_worker_failure(receipt, metadata)["status"], "scientific_failure")

if __name__ == "__main__":
    unittest.main()

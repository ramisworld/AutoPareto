import inspect
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from study.contracts_2026.scheduling import build_schedule, validate_schedule
from study.contracts_2026.train_launcher import _ParentEventLedger, _read_events, _worker_command


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ProcessBoundaryTests(unittest.TestCase):
    @staticmethod
    def _event_child(update_count: int) -> str:
        return """
import json
import sys
import time
w = open(int(sys.argv[1]), "w", buffering=1)
r = open(int(sys.argv[2]), "r", buffering=1)
def send(value):
    w.write(json.dumps(value) + "\\n")
    w.flush()
send({"protocol": "autopareto_candidate_event_v1", "kind": "boundary_ready"})
r.readline()
for update in range(1, %d + 1):
    send({"protocol": "autopareto_candidate_event_v1", "kind": "microbatch_started", "device_batch_size": 16, "sequence_length": 2048, "tokens": 32768})
    send({"protocol": "autopareto_candidate_event_v1", "kind": "microbatch_completed", "device_batch_size": 16, "sequence_length": 2048, "tokens": 32768, "total_tokens": update * 32768, "microbatches": update})
    send({"protocol": "autopareto_candidate_event_v1", "kind": "optimizer_update_completed", "optimizer_updates": update, "microbatches": update, "update_tokens": 32768, "total_tokens": update * 32768, "device_batch_size": 16, "sequence_length": 2048, "accumulation_steps": 1})
    decision = r.readline().strip()
    if decision in {"STOP", "WARMUP_STOP"}:
        break
    if update < %d:
        continue
""" % (update_count, update_count)

    def test_parent_owns_fresh_contract_t_timer(self):
        event_read, event_write = __import__("os").pipe()
        control_read, control_write = __import__("os").pipe()
        process = subprocess.Popen(
            [sys.executable, "-c", self._event_child(1), str(event_write), str(control_read)],
            pass_fds=(event_write, control_read),
        )
        __import__("os").close(event_write)
        __import__("os").close(control_read)
        control = __import__("os").fdopen(control_write, "w", buffering=1)
        try:
            with patch("study.contracts_2026.train_launcher.PROTECTED_TRAINING_SECONDS", 0.0):
                observed, seconds, _events = _read_events(
                    process, event_read, control, contract_id="T", n_star=None, strict_events=True
                )
            self.assertEqual(observed["total_tokens"], 32768)
            self.assertEqual(observed["optimizer_updates"], 1)
            self.assertGreaterEqual(seconds, 0.0)
            process.wait(timeout=2)
        finally:
            control.close()
            try:
                __import__("os").close(event_read)
            except OSError:
                pass

    def test_parent_stops_disposable_warmup_after_eleven_updates(self):
        event_read, event_write = __import__("os").pipe()
        control_read, control_write = __import__("os").pipe()
        process = subprocess.Popen(
            [sys.executable, "-c", self._event_child(11), str(event_write), str(control_read)],
            pass_fds=(event_write, control_read),
        )
        __import__("os").close(event_write)
        __import__("os").close(control_read)
        control = __import__("os").fdopen(control_write, "w", buffering=1)
        try:
            observed, _seconds, _events = _read_events(
                process,
                event_read,
                control,
                contract_id="T",
                n_star=None,
                warmup_only=True,
                strict_events=True,
            )
            self.assertEqual(observed["optimizer_updates"], 11)
            self.assertEqual(observed["total_tokens"], 11 * 32768)
            process.wait(timeout=2)
        finally:
            control.close()
            try:
                __import__("os").close(event_read)
            except OSError:
                pass

    def test_parent_reopens_clock_after_same_process_warmup_reset(self):
        event_read, event_write = __import__("os").pipe()
        control_read, control_write = __import__("os").pipe()
        child = r'''
import json, os, sys
w = os.fdopen(int(sys.argv[1]), "w", buffering=1)
r = os.fdopen(int(sys.argv[2]), "r", buffering=1)
def send(kind, update, total):
    w.write(json.dumps({"protocol":"autopareto_candidate_event_v1","kind":kind,
        "device_batch_size":16,"sequence_length":2048,"tokens":32768,
        "total_tokens":total,"microbatches":update,"optimizer_updates":update,
        "update_tokens":32768,"accumulation_steps":1}) + "\n")
    w.flush()
def phase(count):
    w.write(json.dumps({"protocol":"autopareto_candidate_event_v1","kind":"boundary_ready"}) + "\n")
    w.flush(); r.readline()
    for update in range(1, count + 1):
        send("microbatch_started", update, update * 32768); send("microbatch_completed", update, update * 32768)
        send("optimizer_update_completed", update, update * 32768)
        decision = r.readline().strip()
        if decision == "WARMUP_STOP": return
        if decision == "STOP": return
phase(11); phase(1)
'''
        process = subprocess.Popen(
            [sys.executable, "-c", child, str(event_write), str(control_read)],
            pass_fds=(event_write, control_read),
        )
        __import__("os").close(event_write)
        __import__("os").close(control_read)
        control = __import__("os").fdopen(control_write, "w", buffering=1)
        try:
            with patch("study.contracts_2026.train_launcher.PROTECTED_TRAINING_SECONDS", 0.0):
                observed, seconds, events = _read_events(
                    process,
                    event_read,
                    control,
                    contract_id="T",
                    n_star=None,
                    warmup_then_scientific=True,
                    strict_events=True,
                )
            boundaries = [row for row in events if row["kind"] == "boundary_ready"]
            self.assertEqual(len(boundaries), 2)
            self.assertEqual(observed["optimizer_updates"], 1)
            self.assertEqual(observed["total_tokens"], 32768)
            self.assertGreaterEqual(seconds, 0.0)
            process.wait(timeout=2)
        finally:
            control.close()
            try:
                __import__("os").close(event_read)
            except OSError:
                pass

    def test_retired_worker_cannot_execute_candidate_source(self):
        source = (ROOT / "candidate_worker.py").read_text()
        self.assertNotIn("exec(", source)
        args = type(
            "Args",
            (),
            {
                "context": ROOT,
                "candidate": "train.py",
                "contract": "T",
                "seed": 3,
                "n_star": None,
                "baseline_calibration": False,
                "output": ROOT / "unused-output.json",
            },
        )()
        command = _worker_command(args, 10, 11, ROOT / "worker.json", ROOT / "checkpoint.pt")
        self.assertIn("study.contracts_2026.candidate_process", " ".join(command))

    def test_sandbox_replaces_wrapper_with_clean_candidate_process(self):
        from study.contracts_2026 import candidate_sandbox

        command = [sys.executable, "-m", "study.contracts_2026.candidate_process", "--context", "/candidate"]
        with patch.object(candidate_sandbox.os, "chdir"), patch.object(
            candidate_sandbox.os, "execve", side_effect=AssertionError("sentinel")
        ) as execve:
            with self.assertRaises(AssertionError):
                candidate_sandbox._exec_candidate_python(
                    command,
                    context=pathlib.Path("/candidate"),
                    process_runtime=pathlib.Path("/runtime"),
                )
        execve.assert_called_once()
        self.assertNotIn("runpy", (ROOT / "candidate_sandbox.py").read_text())

    def test_parent_rejects_forged_cumulative_event(self):
        ledger = _ParentEventLedger(contract_id="T", n_star=None)
        ledger.handle({
            "protocol": "autopareto_candidate_event_v1",
            "kind": "microbatch_started",
            "device_batch_size": 16,
            "sequence_length": 2048,
            "tokens": 32768,
        })
        with self.assertRaises(RuntimeError):
            ledger.handle({
                "protocol": "autopareto_candidate_event_v1",
                "kind": "microbatch_completed",
                "device_batch_size": 16,
                "sequence_length": 2048,
                "tokens": 32768,
                "total_tokens": 999999,
                "microbatches": 1,
            })

    def test_parent_rejects_unknown_or_out_of_order_events(self):
        ledger = _ParentEventLedger(contract_id="N", n_star=524288)
        with self.assertRaises(RuntimeError):
            ledger.handle({
                "protocol": "autopareto_candidate_event_v1",
                "kind": "forged_optimizer_update_completed",
            })
        with self.assertRaises(RuntimeError):
            ledger.handle({
                "protocol": "autopareto_candidate_event_v1",
                "kind": "microbatch_completed",
                "tokens": 32768,
                "total_tokens": 32768,
                "microbatches": 1,
            })

    def test_candidate_child_has_no_parent_python_frame(self):
        marker = "protected-secret"
        script = "\n".join((
            "import inspect, json",
            "frame = inspect.currentframe()",
            "names = []",
            "while frame:",
            "    names.extend(frame.f_locals.keys())",
            "    frame = frame.f_back",
            "print(json.dumps({'secret_visible': %r in names}))" % marker,
        ))
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(completed.stdout)["secret_visible"], False)

    def test_protected_inference_refuses_in_process_candidate_loading(self):
        from study.contracts_2026.inference_benchmark import _load_candidate_definitions

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "model-host IPC boundary"):
                _load_candidate_definitions(pathlib.Path(tmp), "train.py")

    def test_parent_rejects_forged_inference_logits(self):
        from study.contracts_2026.inference_ipc import InferenceIPCError, validate_logits_payload

        with self.assertRaises(InferenceIPCError):
            validate_logits_payload(["not-a-logit"])
        with self.assertRaises(InferenceIPCError):
            validate_logits_payload([float("nan")])

    def test_inference_ipc_is_process_only_and_parent_protocol_is_typed(self):
        source = (ROOT / "inference_benchmark.py").read_text()
        self.assertIn("torch.amp.autocast(device_type=\"cuda\", dtype=torch.bfloat16)", source)
        self.assertIn("OFFICIAL_INFERENCE_METRIC_MODE", source)

    def test_candidate_prepare_dependency_is_complete_and_parent_owns_validation(self):
        from study.contracts_2026.remote_runner import _candidate_prepare_source

        source = """MAX_SEQ_LEN = 2048\n# Evaluation (DO NOT CHANGE — this is the fixed metric)\ndef evaluate_bpb(model, tokenizer, batch_size):\n    return 1.0\n"""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "prepare.py"
            path.write_text(source)
            shim = _candidate_prepare_source(path)
        self.assertEqual(shim, source)


class ScheduleTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_block_balanced(self):
        first = build_schedule()
        second = build_schedule()
        self.assertEqual(first, second)
        validate_schedule(first)
        self.assertEqual(sum(row.contract_id == "T" for row in first), 25)
        self.assertEqual(sum(row.contract_id == "N" for row in first), 25)
        self.assertEqual(
            [row.campaign_attempt for row in first if row.contract_id == "T"],
            list(range(1, 26)),
        )
        self.assertEqual(
            [row.campaign_attempt for row in first if row.contract_id == "N"],
            list(range(1, 26)),
        )
        self.assertTrue(all({left.contract_id, right.contract_id} == {"T", "N"} for left, right in zip(first[::2], first[1::2])))
        self.assertTrue(all(left.campaign_attempt == right.campaign_attempt for left, right in zip(first[::2], first[1::2])))


if __name__ == "__main__":
    unittest.main()

import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from study.contracts_2026.batching import COMMON_TOKEN_QUANTUM
from study.contracts_2026.isolation import SandboxUnavailableError, run_agent_command
from study.contracts_2026.protected_runtime import AuthoritativeObserver, ProtectedStop
from study.contracts_2026.train_launcher import _read_events, validate_training_result
try:
    import torch
    from torch import nn
    from protected.inference_adapter import CanonicalGPTAdapter
except ModuleNotFoundError:
    torch = None
    nn = None
    CanonicalGPTAdapter = None


class FakeTensor:
    ndim = 2

    def __init__(self, batch_size=64, sequence_length=2048):
        self.shape = (batch_size, sequence_length)


class ProtectedInstrumentTests(unittest.TestCase):
    def test_canonical_adapter_supports_independent_value_embedding_heads(self):
        if CanonicalGPTAdapter is None:
            self.skipTest("protected CUDA runtime dependencies are not installed in the Mac test interpreter")
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_head = 8
                self.n_kv_head = 4
                self.n_ve_head = 2
                self.head_dim = 2
                self.ve_gate_channels = 4
                self.c_q = nn.Linear(8, 16, bias=False)
                self.c_k = nn.Linear(8, 8, bias=False)
                self.c_v = nn.Linear(8, 8, bias=False)
                self.c_proj = nn.Linear(16, 8, bias=False)
                self.ve_gate = nn.Linear(4, 2, bias=False)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

            def mlp(self, x):
                return x

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.value_embeds = nn.ModuleDict({"0": nn.Embedding(16, 4)})
                self.window_sizes = [[-1]]
                self.config = type("Config", (), {"sequence_len": 16})()
                self.cos = torch.ones(1, 16, 1, 1)
                self.sin = torch.zeros(1, 16, 1, 1)

        adapter = CanonicalGPTAdapter.__new__(CanonicalGPTAdapter)
        adapter.model = Model()
        block = Block()
        x = torch.randn(1, 3, 8)
        token_ids = torch.tensor([[1, 2, 3]])
        cos = torch.ones(1, 3, 1, 1)
        sin = torch.zeros(1, 3, 1, 1)
        output, cache = adapter._layer_attention(block, x, token_ids, 0, 0, None)
        self.assertEqual(tuple(output.shape), (1, 3, 8))
        self.assertEqual(tuple(cache.value.shape), (1, 3, 4, 2))

    def test_canonical_adapter_supports_shared_value_embedding_table(self):
        if CanonicalGPTAdapter is None:
            self.skipTest("protected CUDA runtime dependencies are not installed in the Mac test interpreter")

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_head = 8
                self.n_kv_head = 4
                self.head_dim = 2
                self.ve_gate_channels = 4
                self.c_q = nn.Linear(8, 16, bias=False)
                self.c_k = nn.Linear(8, 8, bias=False)
                self.c_v = nn.Linear(8, 8, bias=False)
                self.c_proj = nn.Linear(16, 8, bias=False)
                self.ve_gate = nn.Linear(4, 4, bias=False)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

            def mlp(self, x):
                return x

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.value_embed = nn.Embedding(16, 8)
                self.window_sizes = [[-1]]
                self.config = type("Config", (), {"sequence_len": 16})()
                self.cos = torch.ones(1, 16, 1, 1)
                self.sin = torch.zeros(1, 16, 1, 1)

        adapter = CanonicalGPTAdapter.__new__(CanonicalGPTAdapter)
        adapter.model = Model()
        block = Block()
        x = torch.randn(1, 3, 8)
        token_ids = torch.tensor([[1, 2, 3]])
        output, cache = adapter._layer_attention(block, x, token_ids, 0, 0, None)
        self.assertEqual(tuple(output.shape), (1, 3, 8))
        self.assertEqual(tuple(cache.value.shape), (1, 3, 4, 2))

    def test_canonical_adapter_supports_shared_value_embedding_moduledict(self):
        if CanonicalGPTAdapter is None:
            self.skipTest("protected CUDA runtime dependencies are not installed in the Mac test interpreter")

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_head = 8
                self.n_kv_head = 4
                self.head_dim = 2
                self.ve_gate_channels = 4
                self.c_q = nn.Linear(8, 16, bias=False)
                self.c_k = nn.Linear(8, 8, bias=False)
                self.c_v = nn.Linear(8, 8, bias=False)
                self.c_proj = nn.Linear(16, 8, bias=False)
                self.ve_gate = nn.Linear(4, 4, bias=False)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

            def mlp(self, x):
                return x

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # Exact T7 representation: value_embeds is a one-entry shared
                # table rather than a singular value_embed attribute.
                self.value_embeds = nn.ModuleDict({"shared": nn.Embedding(16, 8)})
                self.window_sizes = [[-1]]
                self.config = type("Config", (), {"sequence_len": 16})()
                self.cos = torch.ones(1, 16, 1, 1)
                self.sin = torch.zeros(1, 16, 1, 1)

        adapter = CanonicalGPTAdapter.__new__(CanonicalGPTAdapter)
        adapter.model = Model()
        block = Block()
        x = torch.randn(1, 3, 8)
        token_ids = torch.tensor([[1, 2, 3]])
        output, cache = adapter._layer_attention(block, x, token_ids, 0, 0, None)
        self.assertEqual(tuple(output.shape), (1, 3, 8))
        self.assertEqual(tuple(cache.value.shape), (1, 3, 4, 2))

    def test_canonical_adapter_supports_early_shared_value_embedding_moduledict(self):
        if CanonicalGPTAdapter is None:
            self.skipTest("protected CUDA runtime dependencies are not installed in the Mac test interpreter")

        class Attention:
            ve_gate = object()

        class Block:
            attn = Attention()

        class Config:
            n_layer = 8

        class Model:
            config = Config()
            value_embeds = nn.ModuleDict({
                "early": nn.Embedding(16, 8),
                "5": nn.Embedding(16, 8),
                "7": nn.Embedding(16, 8),
            })

        adapter = CanonicalGPTAdapter.__new__(CanonicalGPTAdapter)
        adapter.model = Model()
        token_ids = torch.tensor([[1, 2, 3]])
        early = adapter._value_embedding(Block(), token_ids, 1)
        deep = adapter._value_embedding(Block(), token_ids, 5)
        self.assertTrue(torch.equal(early, adapter.model.value_embeds["early"](token_ids)))
        self.assertTrue(torch.equal(deep, adapter.model.value_embeds["5"](token_ids)))
        self.assertEqual(tuple(early.shape), (1, 3, 8))
        self.assertEqual(tuple(deep.shape), (1, 3, 8))

    def test_canonical_adapter_broadcasts_single_head_value_embedding(self):
        if CanonicalGPTAdapter is None:
            self.skipTest("protected CUDA runtime dependencies are not installed in the Mac test interpreter")

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_head = 4
                self.n_kv_head = 2
                self.head_dim = 2
                self.ve_gate_channels = 4
                self.c_q = nn.Linear(8, 8, bias=False)
                self.c_k = nn.Linear(8, 4, bias=False)
                self.c_v = nn.Linear(8, 4, bias=False)
                self.c_proj = nn.Linear(8, 8, bias=False)
                self.ve_gate = nn.Linear(4, 2, bias=False)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

            def mlp(self, x):
                return x

        class Config:
            n_layer = 7
            sequence_len = 16

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Config()
                self.value_embeds = nn.ModuleDict({"4": nn.Embedding(16, 2)})
                self.window_sizes = [[-1] for _ in range(7)]
                self.cos = torch.ones(1, 16, 1, 1)
                self.sin = torch.zeros(1, 16, 1, 1)

        adapter = CanonicalGPTAdapter.__new__(CanonicalGPTAdapter)
        adapter.model = Model()
        token_ids = torch.tensor([[1, 2, 3]])
        output, cache = adapter._layer_attention(
            Block(), torch.randn(1, 3, 8), token_ids, 4, 0, None
        )
        self.assertEqual(tuple(output.shape), (1, 3, 8))
        self.assertEqual(tuple(cache.value.shape), (1, 3, 2, 2))

    def test_canonical_adapter_supports_stage_shared_value_embedding_modulelist(self):
        if CanonicalGPTAdapter is None:
            self.skipTest("protected CUDA runtime dependencies are not installed in the Mac test interpreter")

        class Attention(nn.Module):
            def __init__(self, gated):
                super().__init__()
                self.n_head = 4
                self.n_kv_head = 4
                self.head_dim = 2
                self.ve_gate_channels = 4
                self.c_q = nn.Linear(8, 8, bias=False)
                self.c_k = nn.Linear(8, 8, bias=False)
                self.c_v = nn.Linear(8, 8, bias=False)
                self.c_proj = nn.Linear(8, 8, bias=False)
                self.ve_gate = nn.Linear(4, 4, bias=False) if gated else None

        class Block(nn.Module):
            def __init__(self, gated):
                super().__init__()
                self.attn = Attention(gated)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"sequence_len": 16, "n_layer": 8})()
                self.transformer = nn.ModuleDict({
                    "h": nn.ModuleList([Block(index in (1, 3, 5, 7)) for index in range(8)])
                })
                self.value_embeds = nn.ModuleList([
                    nn.Embedding(16, 8),
                    nn.Embedding(16, 8),
                ])
                self.window_sizes = [[-1] for _ in range(8)]
                self.cos = torch.ones(1, 16, 1, 1)
                self.sin = torch.zeros(1, 16, 1, 1)

        adapter = CanonicalGPTAdapter.__new__(CanonicalGPTAdapter)
        adapter.model = Model()
        x = torch.randn(1, 3, 8)
        token_ids = torch.tensor([[1, 2, 3]])
        block = adapter.model.transformer.h[5]
        output, cache = adapter._layer_attention(
            block, x, token_ids, 5, 0, None
        )
        self.assertEqual(tuple(output.shape), (1, 3, 8))
        self.assertEqual(tuple(cache.value.shape), (1, 3, 4, 2))

    def test_canonical_adapter_rejects_gated_model_without_value_embedding(self):
        if CanonicalGPTAdapter is None:
            self.skipTest("protected CUDA runtime dependencies are not installed in the Mac test interpreter")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.ModuleDict()
                self.resid_lambdas = nn.Parameter(torch.ones(1))
                self.x0_lambdas = nn.Parameter(torch.zeros(1))
                self.lm_head = nn.Linear(8, 16, bias=False)
                self.cos = torch.ones(1, 16, 1, 1)
                self.sin = torch.zeros(1, 16, 1, 1)

        model = Model()
        adapter = CanonicalGPTAdapter.__new__(CanonicalGPTAdapter)
        adapter.model = model
        class GatedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = type("Attention", (), {"ve_gate": object()})()
        with self.assertRaisesRegex(RuntimeError, "no value_embeds or value_embed"):
            adapter._value_embedding(GatedBlock(), torch.tensor([[1]]), 0)

    @staticmethod
    def _legacy_bwrap_available() -> bool:
        if shutil.which("bwrap") is None:
            return False
        probe = subprocess.run(
            ["bwrap", "--die-with-parent", "--unshare-all", "--ro-bind", "/", "/", "--", "true"],
            capture_output=True,
            text=True,
            check=False,
        )
        return probe.returncode == 0

    def test_contract_t_excludes_exactly_updates_zero_through_ten(self):
        events = io.StringIO()
        controls = io.StringIO(
            "START 0\n"
            + "CONTINUE_UNTIMED\n" * 10
            + "CONTINUE_TIMED 0.0\n"
            + "CONTINUE_TIMED 1.75\n"
        )
        observer = AuthoritativeObserver(
            contract_id="T",
            n_star=None,
            event_stream=events,
            control_stream=controls,
            require_progress=False,
        )
        batch = FakeTensor()
        for update_number in range(1, 13):
            for _ in range(4):
                observed = observer.observe_microbatch_start(batch, batch)
                observer.observe_microbatch_completed(observed)
            observer.before_optimizer_update()
            observer.after_optimizer_update()
            if update_number <= 11:
                self.assertEqual(observer.note_progress(999.0), 0.0)
        self.assertAlmostEqual(observer.note_progress(0.0), 1.75 / 300.0)
        rows = [json.loads(line) for line in events.getvalue().splitlines()]
        boundary = [row for row in rows if row["kind"] == "timing_boundary_started"]
        self.assertEqual(len(boundary), 1)
        self.assertEqual(boundary[0]["excluded_optimizer_updates"], 11)
        self.assertEqual(boundary[0]["first_timed_zero_based_step_index"], 11)

    def test_authoritative_observer_counts_tokens_and_updates(self):
        events = io.StringIO()
        controls = io.StringIO("START 0\nSTOP\n")
        observer = AuthoritativeObserver(
            contract_id="N",
            n_star=COMMON_TOKEN_QUANTUM,
            event_stream=events,
            control_stream=controls,
        )
        batch = FakeTensor()
        for _ in range(4):
            observed = observer.observe_microbatch_start(batch, batch)
            observer.observe_microbatch_completed(observed)
        observer.note_progress(0.0)
        observer.before_optimizer_update()
        with self.assertRaises(ProtectedStop):
            observer.after_optimizer_update()
        summary = observer.summary()
        self.assertEqual(summary["total_tokens"], COMMON_TOKEN_QUANTUM)
        self.assertEqual(summary["microbatches"], 4)
        self.assertEqual(summary["optimizer_updates"], 1)
        self.assertEqual(summary["sequence_length"], 2048)
        self.assertEqual(summary["total_batch_size"], COMMON_TOKEN_QUANTUM)
        self.assertEqual(summary["accumulation_steps"], 4)

    def test_parent_clock_is_authoritative_and_ignores_reported_time(self):
        event_read, event_write = os.pipe()
        control_read, control_write = os.pipe()
        child_code = (
            "import os,sys,json; "
            "w=os.fdopen(int(sys.argv[1]),'w',1); r=os.fdopen(int(sys.argv[2]),'r',1); "
            "w.write(json.dumps({'kind':'boundary_ready'})+'\\n'); "
            "r.readline(); "
            "[(w.write(json.dumps({'kind':'optimizer_update_completed','optimizer_updates':i,'total_tokens':i*32768})+'\\n'),r.readline()) for i in range(1,13)]; "
            "w.write(json.dumps({'kind':'worker_completed','total_tokens':12*32768})+'\\n')"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", child_code, str(event_write), str(control_read)],
            pass_fds=(event_write, control_read),
        )
        os.close(event_write)
        os.close(control_read)
        control = os.fdopen(control_write, "w", buffering=1)
        try:
            with patch("study.contracts_2026.train_launcher.PROTECTED_TRAINING_SECONDS", 0.0), \
                 patch("study.contracts_2026.train_launcher.WORKER_STOP_TIMEOUT_SECONDS", 2.0), \
                 patch("study.contracts_2026.train_launcher.WORKER_START_TIMEOUT_SECONDS", 2.0):
                _, seconds, events = _read_events(
                    process,
                    event_read,
                    control,
                    contract_id="T",
                    n_star=None,
                )
            self.assertGreaterEqual(seconds, 0.0)
            updates = [row for row in events if row["kind"] == "optimizer_update_completed"]
            self.assertEqual(len(updates), 12)
            self.assertEqual(updates[10]["optimizer_updates"], 11)
            self.assertEqual(updates[11]["optimizer_updates"], 12)
        finally:
            control.close()
            try:
                os.close(event_read)
            except OSError:
                pass
            process.wait(timeout=2)

        result = {
            "total_tokens": 32768,
            "protected_measured_training_seconds": 300.001,
            "optimizer_updates": 1,
            "microbatches": 2,
            "sequence_length": 2048,
            "total_batch_size": 32768,
            "device_batch_size": 8,
            "accumulation_steps": 2,
            "candidate_reported_training_seconds": 0.001,
        }
        validate_training_result("T", result, None)

    def test_contract_progress_is_contract_specific(self):
        from study.contracts_2026.contracts import ContractId, ResourceContract

        time_contract = ResourceContract(ContractId.T)
        token_contract = ResourceContract(ContractId.N, time_seconds=None, n_star=COMMON_TOKEN_QUANTUM)
        self.assertEqual(time_contract.progress(measured_training_time=150.0, total_tokens=1), 0.5)
        self.assertEqual(time_contract.progress(measured_training_time=150.0, total_tokens=COMMON_TOKEN_QUANTUM), 0.5)
        self.assertEqual(token_contract.progress(measured_training_time=999.0, total_tokens=COMMON_TOKEN_QUANTUM // 2), 0.5)
        self.assertEqual(token_contract.progress(measured_training_time=0.0, total_tokens=COMMON_TOKEN_QUANTUM), 1.0)

    def test_filesystem_escape_is_denied(self):
        if not self._legacy_bwrap_available():
            self.skipTest("legacy bwrap fixture is superseded by the final permission/seccomp boundary")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            context = root / "context"
            context.mkdir()
            forbidden_root = root / "forbidden-fixtures"
            (forbidden_root / "docs").mkdir(parents=True)
            (forbidden_root / "results/candidates/other-campaign").mkdir(parents=True)
            secrets = [
                forbidden_root / "README.md",
                forbidden_root / "PROJECT_STATE.md",
                forbidden_root / "FORENSIC_RESEARCH_AUDIT.md",
                forbidden_root / "docs/ABSTRACT_DRAFT.md",
                forbidden_root / "results/results.db",
                forbidden_root / "results/candidates/other-campaign/0001-train.py",
            ]
            for secret in secrets:
                secret.write_text("protected fixture")
            paths_repr = repr([str(secret) for secret in secrets])
            script = f"""from pathlib import Path
paths = [Path(value) for value in {paths_repr}]
denied = 0
for p in paths:
    try:
        p.read_text()
    except (PermissionError, FileNotFoundError):
        denied += 1
        continue
    print('READ', p)
    raise SystemExit(2)
print('DENIED', denied)
"""
            try:
                completed = run_agent_command(
                    [sys.executable, "-c", script],
                    context,
                    capture_output=True,
                    text=True,
                )
            except SandboxUnavailableError as exc:
                self.skipTest(str(exc))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("DENIED 6", completed.stdout)

    def test_only_train_py_is_writable(self):
        if not self._legacy_bwrap_available():
            self.skipTest("legacy bwrap fixture is superseded by the final permission/seccomp boundary")
        with tempfile.TemporaryDirectory() as tmp:
            context = pathlib.Path(tmp) / "context"
            context.mkdir()
            for name in ("train.py", "prepare.py", "study_runtime.py", "program.md", "campaign_view.json"):
                (context / name).write_text("original")
            script = """from pathlib import Path
Path('train.py').write_text('changed')
try:
    Path('prepare.py').write_text('forbidden')
except PermissionError:
    print('TRAIN_ONLY')
else:
    raise SystemExit(2)
"""
            try:
                completed = run_agent_command(
                    [sys.executable, "-c", script],
                    context,
                    capture_output=True,
                    text=True,
                )
            except SandboxUnavailableError as exc:
                self.skipTest(str(exc))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("TRAIN_ONLY", completed.stdout)
            self.assertEqual((context / "train.py").read_text(), "changed")
            self.assertEqual((context / "prepare.py").read_text(), "original")


if __name__ == "__main__":
    unittest.main()

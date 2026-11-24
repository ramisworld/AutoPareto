"""Tests for the Claude Code CLI Scientist backend and the single-arm study mode.

No test contacts a model provider: the CLI is the offline fake in
tools/fake_claude_cli and the A40 is the in-process mock of the dry-run tool.
"""

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from study.contracts_2026 import campaign_controller as controller
from study.contracts_2026.bridge import _claude_scientist_environment, _read_claude_stream
from study.contracts_2026.isolation import (
    CLAUDE_SCIENTIST_TOOLS,
    SCIENTIST_CHILD_EDIT_SCHEMA,
    SandboxUnavailableError,
    _validate_claude_command,
)
from study.contracts_2026.scientist_config import ScientistConfig
from study.contracts_2026.tools.dry_run_claude_campaign import DryRunTransport, dry_run_config, run_dry_campaign


def _stream(model="claude-opus-5-5", *, reported_model=None, api_key_source="none", tools=None,
            usage_model=None, subtype="success", proposal=None, version="0.0.0", answer_model=None,
            extra_usage=None, structured=True, rate_limit=None):
    proposal = proposal or {"hypothesis": "h", "intended_change": "i", "change_category": "c"}
    init = {
        "type": "system", "subtype": "init", "model": reported_model or model,
        "tools": tools or CLAUDE_SCIENTIST_TOOLS.split(",") + ["StructuredOutput"], "permissionMode": "acceptEdits",
        "apiKeySource": api_key_source, "mcp_servers": [], "claude_code_version": version,
    }
    assistant = {"type": "assistant", "message": {"model": answer_model or model, "content": []}}
    usage = {usage_model or model: {"outputTokens": 1}, **(extra_usage or {})}
    result = {
        "type": "result", "subtype": subtype, "is_error": subtype != "success",
        "result": json.dumps(proposal), "modelUsage": usage,
    }
    if structured:
        result["structured_output"] = proposal
    limits = {"type": "rate_limit_event", "rate_limit_info": {
        "status": "allowed", "isUsingOverage": False, "overageStatus": "allowed",
        "unifiedWindows": {"five_hour": {"utilization": 0.02}, "seven_day": {"utilization": 0.10}},
        **(rate_limit or {}),
    }}
    return "\n".join(json.dumps(event) for event in (init, limits, assistant, result))


class ClaudeBackendTests(unittest.TestCase):
    def setUp(self):
        self.config = dry_run_config()

    def test_codex_configuration_is_byte_identical_to_frozen_manifests(self):
        codex = ScientistConfig(model="gpt-5.6-sol", reasoning="high")
        self.assertNotIn("agent_backend", codex.as_dict())
        self.assertEqual(self.config.as_dict()["agent_backend"], "claude")

    def test_claude_command_reuses_the_codex_prompt_and_schema(self):
        context = pathlib.Path("/tmp/context")
        claude = controller._scientist_command(context, "pair-a-fixed-time-seed3", 1, self.config, "select_parent")
        codex = controller._scientist_command(
            context, "pair-a-fixed-time-seed3", 1, ScientistConfig(model="gpt-5.6-sol", reasoning="high"), "select_parent"
        )
        self.assertEqual(claude[2], codex[-1], "the Scientist prompt must be identical across backends")
        values = {claude[index]: claude[index + 1] for index in range(len(claude) - 1)}
        self.assertEqual(values["--model"], "claude-opus-5-5")
        self.assertEqual(values["--effort"], "xhigh")
        self.assertEqual(values["--tools"], CLAUDE_SCIENTIST_TOOLS)
        self.assertNotIn("--dangerously-skip-permissions", claude)
        _validate_claude_command(claude, self.config)

    def test_unsafe_or_mismatched_commands_are_rejected(self):
        command = controller._scientist_command(
            pathlib.Path("/tmp/context"), "pair-a-fixed-time-seed3", 1, self.config, "edit_child", "p", {"hypothesis": "h"}
        )
        variants = {
            "skip permissions": command + ["--dangerously-skip-permissions"],
            "no safe mode": [token for token in command if token != "--safe-mode"],
            "fallback model": command + ["--fallback-model", "claude-sonnet-5-5"],
            "wider tools": [token if token != CLAUDE_SCIENTIST_TOOLS else "Read,Edit,Bash" for token in command],
            "other model": [token if token != "claude-opus-5-5" else "claude-sonnet-5-5" for token in command],
        }
        for name, variant in variants.items():
            with self.subTest(name=name), self.assertRaises(SandboxUnavailableError):
                _validate_claude_command(variant, self.config)
        # Same registered schema, minus the draft 2020-12 declaration Claude Code cannot load.
        schema = json.loads(SCIENTIST_CHILD_EDIT_SCHEMA.read_text())
        passed = json.loads(command[command.index("--json-schema") + 1])
        self.assertNotIn("$schema", passed)
        self.assertEqual(passed, {key: value for key, value in schema.items() if key != "$schema"})

    def test_stream_parser_never_accepts_a_substituted_model_or_api_billing(self):
        self.assertEqual(_read_claude_stream(_stream(), self.config)["hypothesis"], "h")
        rejected = {
            "substituted model": _stream(reported_model="claude-sonnet-5-5"),
            "no usage for requested model": _stream(usage_model="claude-sonnet-5-5"),
            "API key billing": _stream(api_key_source="ANTHROPIC_API_KEY"),
            "extra tool": _stream(tools=["Read", "Edit", "Bash"]),
            "failed run": _stream(subtype="error_max_turns"),
            "different CLI version": _stream(version="9.9.9"),
            "fallback model answered": _stream(answer_model="claude-sonnet-5-5"),
            "another model produced output": _stream(extra_usage={"claude-sonnet-5-5": {"outputTokens": 50_000}}),
            "free text instead of structured output": _stream(structured=False),
            "paid overage in use": _stream(rate_limit={"isUsingOverage": True}),
            "usage limit reached": _stream(rate_limit={"status": "rejected"}),
            "seat window nearly exhausted": _stream(
                rate_limit={"unifiedWindows": {"seven_day": {"utilization": 0.96}}}),
        }
        for name, stream in rejected.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                _read_claude_stream(stream, self.config)
        # A line separator inside the result text must not split the result event.
        separated = _stream(proposal={"hypothesis": "a\u2028b", "intended_change": "i", "change_category": "c"})
        self.assertEqual(_read_claude_stream(separated, self.config)["hypothesis"], "a\u2028b")

    def test_environment_does_not_inherit_provider_credentials(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-not-a-key", "ANTHROPIC_BASE_URL": "http://x"}):
            env = _claude_scientist_environment(self.config, None)
        self.assertFalse(any(key.startswith("ANTHROPIC_") for key in env))
        self.assertEqual([key for key in env if key.startswith("CLAUDE_")], ["CLAUDE_CODE_NO_MODEL_FALLBACK"])
        self.assertEqual(env["DISABLE_AUTOUPDATER"], "1")
        self.assertEqual(env["CLAUDE_CODE_NO_MODEL_FALLBACK"], "1")
        with self.assertRaises(ValueError):
            _claude_scientist_environment(self.config, {"ANTHROPIC_API_KEY": "x"})

    def test_full_single_arm_campaign_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_dry_campaign(pathlib.Path(tmp) / "state", fail_attempts=(3,), interrupt_attempts=(4,))
        self.assertEqual(report["events"][-1], "run_completed")
        self.assertTrue(any(event.startswith("interrupted") for event in report["events"]))
        self.assertEqual(report["active_contracts"], ["T"])
        self.assertEqual(report["not_in_study"], 25)
        self.assertEqual(report["unique_invocations"], 25, "no attempt may be trained twice")
        self.assertEqual(report["verified_absolute_files"], ["/dry-run/T/baseline.pt"],
                         "the paired T baseline checkpoint must be hash-verified on the pod")
        self.assertTrue(all(run_id.startswith("claude-dry-run-position-") for run_id in report["remote_run_ids"]),
                        "remote payloads must not reuse the Codex arm's pair-a namespace")
        history = report["history"]
        self.assertEqual([record["attempt"] for record in history], list(range(1, 26)))
        self.assertEqual(history[2]["status"], "scientific_failure")
        self.assertNotEqual(history[3]["parent"], "pair-a-fixed-time-seed3-attempt-03",
                            "a failed attempt must never become a parent")
        self.assertEqual({(r["agent_model"], r["effort"]) for r in history}, {("claude-opus-5-5", "xhigh")})

    def test_manifest_frozen_contracts_must_match_the_requested_arms(self):
        # A single-arm manifest must not start a paired run, and vice versa.
        frozen = {"derivation": {"active_contracts": ["T"]}}
        transport = DryRunTransport()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(controller, "_verify_final_manifest", lambda *_a, **_k: frozen), \
                mock.patch.object(controller, "_verify_transport_config", lambda *_a, **_k: None), \
                mock.patch.object(controller, "_verify_local_state_config", lambda *_a, **_k: None), \
                self.assertRaisesRegex(RuntimeError, "manifest freezes contracts"):
            controller.run_campaign(
                state_root=pathlib.Path(tmp) / "state", baseline_path=None,
                manifest_path=pathlib.Path(tmp) / "m.json", transport=transport,
                scientist_config=self.config, active_contracts=("T", "N"),
            )


if __name__ == "__main__":
    unittest.main()

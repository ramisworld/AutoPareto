#!/usr/bin/env python3
"""Capture the immutable repository inventory used to start the contract study.

This script intentionally inventories the repository's existing evidence only.  It
excludes the study directory itself, dependency caches, virtual environments, and
Python bytecode.  It is a provenance utility, not an experiment launcher.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import platform
import shutil
import subprocess
import sys
from collections import Counter


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "study" / "contracts_2026" / "provenance"
EXCLUDED_PARTS = {".git", ".venv", "venv", "node_modules", ".cache", ".pytest_cache", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_inventory_file(path: pathlib.Path) -> bool:
    rel = path.relative_to(ROOT)
    if "study" in rel.parts:
        return False
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def git_status() -> list[str]:
    return run("git", "status", "--porcelain=v1", "--ignored", "--untracked-files=all").stdout.splitlines()


def status_for(rel: str, lines: list[str]) -> str:
    for line in lines:
        if len(line) >= 4 and line[3:] == rel:
            return line[:2]
    return "  "


def tracked(path: pathlib.Path) -> bool:
    return run("git", "ls-files", "--error-unmatch", path.relative_to(ROOT).as_posix()).returncode == 0


def last_commit(path: pathlib.Path) -> str | None:
    value = run("git", "log", "-1", "--format=%H", "--", path.relative_to(ROOT).as_posix()).stdout.strip()
    return value or None


def python_info(executable: str | None) -> dict[str, object] | None:
    if executable is None:
        return None
    code = (
        "import sys, torch; print(sys.version); print(getattr(torch, '__version__', 'unavailable')); "
        "print(getattr(torch.version, 'cuda', None)); print(torch.cuda.is_available())"
    )
    result = subprocess.run([executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "executable": executable,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def historical_blobs() -> list[dict[str, object]]:
    commit = "b8eab2894a5f2592a3a6e80b74e9c39ced80d67e"
    paths = [
        "runner.py",
        "programs/program_autopareto.md",
        "scripts/run_runner.sh",
        "protected/train_launcher.py",
        "RESEARCH_PROTOCOL.md",
        "docs/candidate_interface.md",
        "inference_benchmark.py",
    ]
    records = []
    for path in paths:
        blob = run("git", "show", f"{commit}:{path}")
        if blob.returncode != 0:
            continue
        content = blob.stdout.encode()
        records.append(
            {
                "commit": commit,
                "path": path,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captured-at", default=None, help="ISO timestamp for a previously observed capture")
    args = parser.parse_args()

    lines = git_status()
    status_text = "\n".join(lines)
    now = args.captured_at or dt.datetime.now().astimezone().isoformat()
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not is_inventory_file(path):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("results/"):
            category = "result_database_or_historical_artifact"
        elif rel.startswith("queue/"):
            category = "candidate_source_or_queue_artifact"
        elif path.suffix.lower() in {".md", ".markdown"}:
            category = "relevant_markdown"
        else:
            continue
        files.append(
            {
                "path": rel,
                "category": category,
                "size": path.stat().st_size,
                "sha256": sha256(path),
                "git_status": status_for(rel, lines),
                "tracked": tracked(path),
                "last_commit": last_commit(path) if tracked(path) else None,
            }
        )

    environment_files = []
    for rel in ("pyproject.toml", "requirements-release.txt", ".python-version"):
        path = ROOT / rel
        if path.exists():
            environment_files.append({"path": rel, "size": path.stat().st_size, "sha256": sha256(path)})

    status_codes = Counter(line[:2] for line in lines if len(line) >= 2)
    head = run("git", "rev-parse", "HEAD").stdout.strip()
    branch = run("git", "branch", "--show-current").stdout.strip()
    manifest = {
        "format": "autopareto_prestudy_manifest_v1",
        "captured_at": now,
        "repository": str(ROOT),
        "branch": branch,
        "head": head,
        "git_status": {
            "command": "git status --porcelain=v1 --ignored --untracked-files=all",
            "sha256_of_exact_status_text": hashlib.sha256(status_text.encode()).hexdigest(),
            "counts_by_code": dict(sorted(status_codes.items())),
            "research_relevant_entries": [
                line for line in lines
                if len(line) >= 4 and (
                    line[3:].startswith("results/")
                    or line[3:].startswith("queue/")
                    or line[3:].lower().endswith((".md", ".markdown"))
                )
            ],
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python_default": python_info(shutil.which("python3")),
        "python_venv": python_info(str(ROOT / ".venv/bin/python") if (ROOT / ".venv/bin/python").exists() else None),
        "environment_lock_files": environment_files,
        "historical_git_versions": historical_blobs(),
        "inventory_definition": {
            "included": ["results/**", "queue/**", "all repository Markdown files", "excluding study/** and caches"],
            "excluded": sorted(EXCLUDED_PARTS | EXCLUDED_SUFFIXES),
            "purpose": "Pre-study SHA256 inventory; no evidence was rewritten or deleted.",
        },
        "scope_summary": {
            "file_count": len(files),
            "total_bytes": sum(item["size"] for item in files),
            "by_category": dict(sorted(Counter(item["category"] for item in files).items())),
        },
        "files": files,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prestudy_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    by_category = Counter(item["category"] for item in files)
    md = [
        "# Pre-study repository manifest",
        "",
        "This is the Phase 0.5 starting-state record. It was captured before any study harness file was added.",
        "The JSON companion contains the complete per-file SHA256 inventory; this Markdown file is a readable index.",
        "",
        f"- Repository: `{ROOT}`",
        f"- Branch: `{branch}`",
        f"- HEAD: `{head}`",
        f"- Captured at: `{now}`",
        f"- Exact status-text SHA256: `{manifest['git_status']['sha256_of_exact_status_text']}`",
        f"- Inventory files: `{len(files)}`",
        f"- Inventory bytes: `{sum(item['size'] for item in files)}`",
        "",
        "## Scope",
        "",
        "The inventory includes all files under `results/` and `queue/`, all repository Markdown files, and their Git status/commit metadata, while excluding the study directory, Git internals, virtual environments, dependency/vendor directories, caches, and Python bytecode.",
        "",
        "| Category | Files |",
        "|---|---:|",
    ]
    md.extend(f"| `{category}` | {count} |" for category, count in sorted(by_category.items()))
    def version_line(info: dict[str, object] | None) -> str:
        if not info:
            return "unavailable"
        lines = str(info.get("stdout", "")).splitlines()
        return lines[0] if lines else f"unavailable (returncode={info.get('returncode')})"

    md.extend([
        "",
        "## Environment",
        "",
        f"- Default Python: `{version_line(manifest['python_default'])}`",
        f"- Virtual-environment Python: `{version_line(manifest['python_venv'])}`",
        "- Environment lock/configuration file hashes are in the JSON companion.",
        "",
        "## Historical instrument blobs",
        "",
        "The exact historical Git blob records used to recover the AutoPareto instrument are listed under `historical_git_versions` in the JSON companion.",
        "",
        "No tracked or untracked research evidence was cleaned, deleted, rewritten, or committed by this capture.",
    ])
    (OUT / "prestudy_manifest.md").write_text("\n".join(md) + "\n")
    print(json.dumps({
        "manifest": str(OUT / "prestudy_manifest.json"),
        "companion": str(OUT / "prestudy_manifest.md"),
        "head": head,
        "branch": branch,
        "captured_at": now,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "status_sha256": manifest["git_status"]["sha256_of_exact_status_text"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

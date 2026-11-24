"""Evidence-preserving A40 clean-room preparation.

This is an operator tool, not candidate code.  It snapshots every existing
trusted/evidence path before cleanup, verifies the root-only copy byte-for-byte,
then removes only the listed sensitive paths from the candidate-visible
``/workspace``.  Approved data/cache is copied to a separate immutable input
tree used by the candidate process boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time
from typing import Any


EVIDENCE_CACHE_DIR_NAMES = {"triton-cache", "torchinductor-cache", "__pycache__"}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(
    root: pathlib.Path,
    source: pathlib.Path,
    *,
    exclude_cache_dirs: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = []
        for directory, directory_names, file_names in os.walk(source):
            if exclude_cache_dirs:
                directory_names[:] = [
                    name for name in directory_names if name not in EVIDENCE_CACHE_DIR_NAMES
                ]
            paths.extend(pathlib.Path(directory) / name for name in sorted(file_names))
    else:
        return rows
    for path in sorted(paths):
        if exclude_cache_dirs and any(part in EVIDENCE_CACHE_DIR_NAMES for part in path.relative_to(source).parts):
            continue
        rows.append({
            "source": str(path),
            "relative_to_source": str(path.relative_to(source)) if source.is_dir() else path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def copy_verified(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    exclude_cache_dirs: bool = False,
) -> list[dict[str, Any]]:
    if not source.exists():
        return []
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite existing trusted destination: {destination}")
    if source.is_dir():
        ignore = (
            shutil.ignore_patterns(*sorted(EVIDENCE_CACHE_DIR_NAMES))
            if exclude_cache_dirs
            else None
        )
        shutil.copytree(source, destination, symlinks=False, ignore=ignore)
    else:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    before = file_inventory(source, source, exclude_cache_dirs=exclude_cache_dirs)
    after = file_inventory(destination, destination)
    before_hashes = [row["sha256"] for row in before]
    after_hashes = [row["sha256"] for row in after]
    if before_hashes != after_hashes or len(before) != len(after):
        raise RuntimeError(f"byte-for-byte copy verification failed: {source} -> {destination}")
    return before


def readonly_tree(root: pathlib.Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o444)
    if root.is_dir():
        root.chmod(0o555)


def remove_source(path: pathlib.Path) -> None:
    """Remove a previously snapshotted source, accommodating FUSE modes."""
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        for child in sorted(path.rglob("*"), reverse=True):
            try:
                child.chmod(0o700 if child.is_dir() else 0o600)
            except OSError:
                pass
        path.chmod(0o700)
        shutil.rmtree(path)
    else:
        try:
            path.chmod(0o600)
        except OSError:
            pass
        path.unlink()


def finish_existing(args: argparse.Namespace) -> dict[str, Any]:
    trusted_root = args.trusted_root.resolve(strict=True)
    state_root = trusted_root / "state"
    manifest_path = state_root / "evidence_snapshot_before_workspace_clean.json"
    manifest = json.loads(manifest_path.read_text())
    for raw_path in manifest["sensitive_paths"]:
        remove_source(pathlib.Path(raw_path))
    remove_source(args.source_root.resolve(strict=True) / ".cache")
    autopareto = args.source_root.resolve(strict=True) / "autopareto"
    if autopareto.exists():
        for path in autopareto.iterdir():
            if path.name == ".venv":
                continue
            if path.is_file():
                path.chmod(0o444)
        autopareto.chmod(0o555)
    candidate_root = args.candidate_root.resolve()
    candidate_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    (candidate_root / "contexts").mkdir(mode=0o755, exist_ok=True)
    manifest["cleanup_status"] = "completed"
    manifest_without_hash = dict(manifest)
    manifest_without_hash.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        (json.dumps(manifest_without_hash, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve(strict=True)
    trusted_root = args.trusted_root.resolve()
    state_root = trusted_root / "state"
    recovery_root = trusted_root / "recovery_before_clean"
    approved_root = args.approved_root.resolve()
    candidate_root = args.candidate_root.resolve()
    if not args.execute:
        raise RuntimeError("preparation is destructive; pass --execute after reviewing the path list")
    if trusted_root.exists() and any(trusted_root.iterdir()):
        raise RuntimeError(f"trusted root is not empty: {trusted_root}")
    if candidate_root.exists() and any(candidate_root.iterdir()):
        raise RuntimeError(f"candidate clean-room root is not empty: {candidate_root}")

    trusted_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_root.mkdir(mode=0o700, parents=True)
    recovery_root.mkdir(mode=0o700, parents=True)
    approved_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    candidate_root.mkdir(mode=0o755, parents=True, exist_ok=True)

    autopareto = source_root / "autopareto"
    sensitive: list[pathlib.Path] = [
        autopareto / name
        for name in (
            "protected",
            "scripts",
            "study",
            "search",
            "preflight_frozen_checkpoints",
            "registered_model_refs",
            "recovery_evidence_20260902",
            "study_remote_runs",
        )
    ]
    sensitive.extend(sorted(source_root.glob("*.tar.zst")))
    recv_log = source_root / "hf_recv.log"
    if recv_log.exists():
        sensitive.append(recv_log)
    sensitive.extend(sorted(autopareto.glob(".candidate-process-runtime-*")))
    sensitive = [path for path in sensitive if path.exists()]

    inventory: list[dict[str, Any]] = []
    copied: list[dict[str, Any]] = []
    for source in sensitive:
        relative = source.relative_to(source_root)
        destination = recovery_root / relative
        rows = file_inventory(source_root, source, exclude_cache_dirs=True)
        inventory.extend(rows)
        copy_verified(source, destination, exclude_cache_dirs=True)
        copied.extend(rows)

    # Approved data/tokenizer/kernel caches are candidate inputs, not trusted
    # evidence.  They are still copied and hashed before the writable source
    # cache is removed from /workspace.
    source_cache = source_root / ".cache"
    approved_cache = approved_root / "cache"
    cache_inventory = copy_verified(source_cache, approved_cache)
    readonly_tree(approved_cache)

    manifest: dict[str, Any] = {
        "schema": "autopareto_clean_room_evidence_snapshot_v1",
        "created_at_unix_ns": time.time_ns(),
        "source_root": str(source_root),
        "trusted_recovery_root": str(recovery_root),
        "sensitive_paths": [str(path) for path in sensitive],
        "evidence_files": inventory,
        "evidence_file_count": len(inventory),
        "approved_cache": {
            "source": str(source_cache),
            "destination": str(approved_cache),
            "files": cache_inventory,
            "file_count": len(cache_inventory),
        },
        "cleanup_status": "copied_and_verified_pending_removal",
    }
    manifest_path = state_root / "evidence_snapshot_before_workspace_clean.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_hash = sha256_file(manifest_path)

    for source in sensitive:
        if source.is_dir() and not source.is_symlink():
            shutil.rmtree(source)
        else:
            source.unlink()
    if source_cache.exists():
        shutil.rmtree(source_cache)

    # Only candidate-neutral runtime files remain in the legacy top-level
    # directory.  New attempt contexts are staged below the fresh clean-room
    # root and never beside trusted study files.
    if autopareto.exists():
        for path in autopareto.iterdir():
            if path.name == ".venv":
                continue
            if path.is_file():
                path.chmod(0o444)
        autopareto.chmod(0o555)
    (candidate_root / "contexts").mkdir(mode=0o755)
    (candidate_root / "contexts").chmod(0o755)

    manifest["cleanup_status"] = "completed"
    manifest["manifest_sha256_before_cleanup_status_update"] = manifest_hash
    manifest_without_hash = dict(manifest)
    manifest_without_hash.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        (json.dumps(manifest_without_hash, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/workspace"))
    parser.add_argument("--trusted-root", type=pathlib.Path, default=pathlib.Path("/opt/autopareto_trusted"))
    parser.add_argument("--approved-root", type=pathlib.Path, default=pathlib.Path("/opt/autopareto_candidate_inputs"))
    parser.add_argument("--candidate-root", type=pathlib.Path, default=pathlib.Path("/workspace/autopareto_cleanroom"))
    parser.add_argument("--finish-existing", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.finish_existing:
        if not args.execute:
            raise RuntimeError("finishing cleanup requires --execute")
        result = finish_existing(args)
    else:
        result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

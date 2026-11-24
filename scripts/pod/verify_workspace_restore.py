#!/usr/bin/env python3
"""Verify a /workspace restored from an off-pod backup against the hash lists computed on the source pod.

usage: python3 -I verify_workspace_restore.py <backup folder on this volume> <report.json>

Checks, all against lists written before the restore (doc 19):
1. every file the source pod listed in pod_manifests/workspace.sha256 exists here with the same SHA-256;
2. no other regular file exists under /workspace, outside the rebuildable Env A .venv and the backup folder;
3. the symlinks equal pod_manifests/workspace.symlinks;
4. the backup folder's own files (var_lib_autopareto copy, manifests, receipts) match SHA256SUMS_ALL.txt,
   and that list's workspace entries agree with the source pod's workspace.sha256.
Writes one JSON report and exits 0 only if everything matches. Reads only; changes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

ROOT = pathlib.Path("/workspace")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_list(path: pathlib.Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        if line.startswith("\\") or line[64:66] != "  ":
            raise SystemExit(f"unexpected line in {path}: {line!r}")
        entries[line[66:]] = line[:64]
    return entries


def command(*argv: str) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def walk(root: pathlib.Path, excluded: set[str]) -> tuple[list[str], list[str]]:
    files, links = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = "." + dirpath[len(str(root)):]
        kept = []
        for name in dirnames:
            rel = f"{rel_dir}/{name}"
            if os.path.islink(os.path.join(dirpath, name)):
                links.append(f"{rel} -> {os.readlink(os.path.join(dirpath, name))}")
            elif rel not in excluded:
                kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            full, rel = os.path.join(dirpath, name), f"{rel_dir}/{name}"
            if os.path.islink(full):
                links.append(f"{rel} -> {os.readlink(full)}")
            else:
                files.append(rel)
    return sorted(files), sorted(links)


def main() -> int:
    backup = pathlib.Path(sys.argv[1]).resolve()
    report_path = pathlib.Path(sys.argv[2])
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite {report_path}")
    started = time.time()
    excluded = {"./autopareto-camera-ready/.venv", "./" + str(backup.relative_to(ROOT))}
    expected = read_list(backup / "pod_manifests/workspace.sha256")
    expected_links = sorted((backup / "pod_manifests/workspace.symlinks").read_text().splitlines())
    files, links = walk(ROOT, excluded)
    present = set(files)
    missing = sorted(set(expected) - present)
    extra = sorted(present - set(expected))
    mismatched = sorted(name for name in expected if name in present and sha256(ROOT / name) != expected[name])

    all_sums = read_list(backup / "SHA256SUMS_ALL.txt")
    backup_own = {name: digest for name, digest in all_sums.items() if not name.startswith("./workspace/")}
    backup_missing = sorted(name for name in backup_own if not (backup / name).is_file())
    backup_mismatched = sorted(name for name in backup_own
                               if (backup / name).is_file() and sha256(backup / name) != backup_own[name])
    mac_view = {"./" + name[len("./workspace/"):]: digest for name, digest in all_sums.items()
                if name.startswith("./workspace/")}
    lists_agree = mac_view == expected

    report = {
        "schema": "autopareto_workspace_restore_verification_v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "gpu": command("nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"),
        "volume": command("findmnt", "-no", "SOURCE,FSTYPE", str(ROOT)),
        "backup_folder": str(backup),
        "workspace": {"expected_files": len(expected), "present_files": len(files), "missing": missing,
                      "extra": extra, "hash_mismatches": mismatched},
        "symlinks": {"expected": len(expected_links), "present": len(links), "identical": links == expected_links},
        "backup_folder_files": {"expected": len(backup_own), "missing": backup_missing,
                                "hash_mismatches": backup_mismatched},
        "mac_and_pod_workspace_lists_agree": lists_agree,
        "seconds": round(time.time() - started, 1),
    }
    report["verified"] = (not missing and not extra and not mismatched and links == expected_links
                          and not backup_missing and not backup_mismatched and lists_agree)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

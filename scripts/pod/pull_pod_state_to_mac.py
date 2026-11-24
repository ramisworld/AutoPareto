#!/usr/bin/env python3
"""Pull the pod's research state to the Mac every 15 minutes and verify each new or changed file by SHA-256.

usage (on the Mac; the SSH endpoint is passed at run time and never stored in the repository):
  caffeinate -i python3 -I scripts/pod/pull_pod_state_to_mac.py --target root@HOST --port PORT \
      --identity ~/.ssh/id_ed25519 --dest DIR [--interval 900] [--once]

Pulled: /var/lib/autopareto -> DIR/var_lib_autopareto, and /workspace -> DIR/workspace without the
rebuildable Env A .venv, /workspace/pod_backups (already on the Mac) and /workspace/mirror (a copy of
/var/lib/autopareto). Copies only add or update files; nothing is ever deleted on the Mac. On the pod,
rsync, find and sha256sum run at nice 19 and idle I/O priority so they stay out of the way of the
CPU-bound decode measurement (doc 18). A file is verified when its pod-side and Mac-side SHA-256 agree;
a mismatch (usually a file written during the transfer) is re-pulled by checksum and checked again, and
anything still unverified is listed in status.json and raises a macOS notification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOTS = {
    "var_lib_autopareto": ("/var/lib/autopareto", []),
    "workspace": ("/workspace", ["autopareto-camera-ready/.venv", "pod_backups", "mirror"]),
}
LOW = "nice -n 19 ionice -c3"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Pod:
    def __init__(self, args: argparse.Namespace):
        self.ssh = ["ssh", "-p", str(args.port), "-i", args.identity, "-o", "BatchMode=yes",
                    "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]
        self.target = args.target

    def run(self, command: str, stdin: bytes | None = None) -> bytes:
        result = subprocess.run(self.ssh + [self.target, command], input=stdin, capture_output=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"pod command failed ({result.returncode}): {result.stderr.decode()[-500:]}")
        return result.stdout

    def rsync(self, source: str, dest: pathlib.Path, excludes: list[str], extra: list[str] | None = None) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        command = ["rsync", "-rlpt", f"--rsync-path={LOW} rsync", "-e", " ".join(self.ssh)]
        command += [f"--exclude=/{item}" for item in excludes] + (extra or [])
        command += [f"{self.target}:{source}/", f"{dest}/"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=7200)
        # 24 = some files vanished during the transfer (they are picked up or verified next cycle)
        if result.returncode not in (0, 24):
            raise RuntimeError(f"rsync {source} failed ({result.returncode}): {result.stderr[-500:]}")

    def listing(self, root: str, excludes: list[str]) -> dict[str, tuple[int, str]]:
        prunes = " ".join(f"-path {root}/{item} -prune -o" for item in excludes)
        out = self.run(f"{LOW} find {root} {prunes} -type f -printf '%P\\t%s\\t%T@\\0'")
        entries = {}
        for record in out.split(b"\0"):
            if record:
                name, size, mtime = record.decode().rsplit("\t", 2)
                entries[name] = (int(size), mtime)
        return entries

    def hashes(self, root: str, names: list[str]) -> dict[str, str]:
        if not names:
            return {}
        payload = b"\0".join(name.encode() for name in names) + b"\0"
        out = self.run(f"cd {root} && {LOW} xargs -0 sha256sum --zero --", stdin=payload)
        result = {}
        for record in out.split(b"\0"):
            if record:
                digest, name = record.decode().split("  ", 1)
                result[name] = digest
        return result


def notify(message: str) -> None:
    subprocess.run(["osascript", "-e", f'display notification "{message}" with title "AutoPareto pod pull"'],
                   capture_output=True)


def cycle(pod: Pod, dest: pathlib.Path, state: dict) -> dict:
    started = time.time()
    summary = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "roots": {}}
    for label, (root, excludes) in ROOTS.items():
        local = dest / label
        pod.rsync(root, local, excludes)
        listing = pod.listing(root, excludes)
        verified = state.setdefault(label, {})
        todo = sorted(name for name, (size, mtime) in listing.items()
                      if verified.get(name, [None, None])[:2] != [size, mtime])
        remote = pod.hashes(root, todo)
        bad = [name for name in todo
               if name not in remote or not (local / name).is_file() or sha256(local / name) != remote[name]]
        if bad:  # re-pull just these by checksum, then check once more
            files_from = dest / f".retry_{label}.txt"
            files_from.write_text("".join(f"{name}\n" for name in bad))
            pod.rsync(root, local, excludes, ["--checksum", f"--files-from={files_from}"])
            files_from.unlink()
            remote.update(pod.hashes(root, bad))
            bad = [name for name in bad
                   if name not in remote or not (local / name).is_file() or sha256(local / name) != remote[name]]
        for name in todo:
            if name not in bad:
                size, mtime = listing[name]
                verified[name] = [size, mtime, remote[name]]
        summary["roots"][label] = {"pod_files": len(listing), "verified_files": sum(n in verified for n in listing),
                                   "newly_verified": len(todo) - len(bad), "unverified": bad}
    summary["seconds"] = round(time.time() - started, 1)
    summary["all_verified"] = all(not r["unverified"] for r in summary["roots"].values())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--dest", type=pathlib.Path, required=True)
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    pod = Pod(args)
    state_path, status_path, log_path = dest / "verified_state.json", dest / "status.json", dest / "pull.log"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    while True:
        try:
            summary = cycle(pod, dest, state)
        except Exception as exc:  # keep going: the next cycle retries; the failure is logged and notified
            summary = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "error": str(exc)[-800:],
                       "all_verified": False}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True))
        os.replace(tmp, state_path)
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=1) + "\n")
        os.replace(tmp, status_path)
        with open(log_path, "a") as log:
            log.write(json.dumps(summary, sort_keys=True) + "\n")
        if not summary["all_verified"]:
            notify("pull cycle incomplete: see status.json")
        if args.once:
            return 0 if summary["all_verified"] else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

"""Compatibility entry point for the retired in-process worker.

Candidate source must never be executed by this module. The trusted launcher now
starts study.contracts_2026.candidate_process in a separate restricted process.
"""
from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "in-process candidate worker is retired; use candidate_process through the "
        "trusted train launcher"
    )


if __name__ == "__main__":
    raise SystemExit(main())


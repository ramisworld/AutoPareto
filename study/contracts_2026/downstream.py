"""Post-search-only downstream evaluation registry."""

from __future__ import annotations

from dataclasses import dataclass


SEALED_TASKS = (
    "held_out_bpb",
    "BLiMP",
    "PIQA",
    "ARC-Easy",
    "LAMBADA_perplexity",
    "LAMBADA_exact_match",
)


@dataclass(frozen=True)
class DownstreamPlan:
    candidate_source: str
    checkpoint_hash: str
    output_namespace: str = "post_search_downstream"

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_source": self.candidate_source,
            "checkpoint_hash": self.checkpoint_hash,
            "tasks": list(SEALED_TASKS),
            "search_visible": False,
            "returns_to_search_frontier": False,
            "output_namespace": self.output_namespace,
        }


def assert_sealed_from_search(payload: dict[str, object]) -> None:
    if payload.get("search_visible", True):
        raise ValueError("downstream results cannot be visible during search")
    if payload.get("returns_to_search_frontier", True):
        raise ValueError("downstream results cannot return to the search frontier")

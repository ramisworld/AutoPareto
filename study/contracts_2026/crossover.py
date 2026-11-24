"""Post-search crossover plans; this module never feeds results into search."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class CrossoverPlan:
    candidate_source: str
    seed: int
    n_star: int
    fixed_time_seconds: float = 300.0
    include_matched_updates: bool = True
    output_namespace: str = "post_search_crossover"

    def conditions(self) -> tuple[dict[str, object], ...]:
        conditions: list[dict[str, object]] = [
            {"condition": "T", "training_seconds": self.fixed_time_seconds, "tokens": None},
            {"condition": "N", "training_seconds": None, "tokens": self.n_star},
        ]
        if self.include_matched_updates:
            conditions.append(
                {
                    "condition": "N_matched_updates",
                    "training_seconds": None,
                    "tokens": self.n_star,
                    "match_updates": "when a legal TOTAL_BATCH_SIZE makes the requested update count exact",
                }
            )
        return tuple(conditions)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_source": self.candidate_source,
            "seed": self.seed,
            "n_star": self.n_star,
            "conditions": list(self.conditions()),
            "sealed_from_search": True,
            "results_return_to_search": False,
        }


def crossover_command(candidate_source: str, n_star: int, seed: int = 3) -> list[str]:
    return [
        "python",
        "-m",
        "study.contracts_2026.runner",
        "crossover-plan",
        "--candidate-source",
        candidate_source,
        "--seed",
        str(seed),
        "--n-star",
        str(n_star),
        "--post-search-only",
    ]

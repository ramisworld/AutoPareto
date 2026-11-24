"""Pre-registered, single-GPU block-balanced Pair-A execution order.

Each block contains the same-numbered T and N attempt. Only the within-block
order is randomized, so each adaptive campaign remains in internal order.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


SCHEDULE_VERSION = "autopareto_pair_a_interleaved_same_index_block_balanced_v2"
SCHEDULE_SEED = 20260902
ATTEMPTS_PER_CONDITION = 25


@dataclass(frozen=True)
class ScheduledAttempt:
    global_position: int
    contract_id: str
    campaign_id: str
    campaign_attempt: int

    def as_dict(self) -> dict[str, object]:
        return {
            "global_position": self.global_position,
            "contract_id": self.contract_id,
            "campaign_id": self.campaign_id,
            "campaign_attempt": self.campaign_attempt,
        }


def build_schedule(
    *,
    seed: int = SCHEDULE_SEED,
    attempts_per_condition: int = ATTEMPTS_PER_CONDITION,
    time_campaign_id: str = "pair-a-fixed-time-seed3",
    token_campaign_id: str = "pair-a-fixed-tokens-seed3",
) -> list[ScheduledAttempt]:
    if attempts_per_condition <= 0:
        raise ValueError("attempt count must be positive")
    rng = random.Random(seed)
    result: list[ScheduledAttempt] = []
    for position, attempt in enumerate(range(1, attempts_per_condition + 1)):
        contract_order = ["T", "N"]
        if rng.randrange(2):
            contract_order.reverse()
        pair = [
            ScheduledAttempt(
                position * 2 + offset,
                contract_id,
                time_campaign_id if contract_id == "T" else token_campaign_id,
                attempt,
            )
            for offset, contract_id in enumerate(contract_order, start=1)
        ]
        result.extend(pair)
    validate_schedule(
        result,
        attempts_per_condition=attempts_per_condition,
        time_campaign_id=time_campaign_id,
        token_campaign_id=token_campaign_id,
    )
    return result


def validate_schedule(
    schedule: Iterable[ScheduledAttempt],
    *,
    attempts_per_condition: int = ATTEMPTS_PER_CONDITION,
    time_campaign_id: str = "pair-a-fixed-time-seed3",
    token_campaign_id: str = "pair-a-fixed-tokens-seed3",
) -> None:
    rows = list(schedule)
    if len(rows) != 2 * attempts_per_condition:
        raise ValueError("schedule must contain exactly 25 T and 25 N attempts")
    if [row.global_position for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("schedule positions must be contiguous and one-based")
    if sum(row.contract_id == "T" for row in rows) != attempts_per_condition:
        raise ValueError("schedule must contain exactly the registered T attempts")
    if sum(row.contract_id == "N" for row in rows) != attempts_per_condition:
        raise ValueError("schedule must contain exactly the registered N attempts")
    for row in rows:
        expected_campaign = time_campaign_id if row.contract_id == "T" else token_campaign_id
        if row.campaign_id != expected_campaign:
            raise ValueError("T/N histories are not isolated")
        if not 1 <= row.campaign_attempt <= attempts_per_condition:
            raise ValueError("campaign attempt is outside the registered range")
    for left, right in zip(rows[::2], rows[1::2]):
        if {left.contract_id, right.contract_id} != {"T", "N"}:
            raise ValueError("each execution block must contain one T and one N attempt")
        if left.campaign_id == right.campaign_id:
            raise ValueError("paired T/N attempts cannot share a history")
        if left.campaign_attempt != right.campaign_attempt:
            raise ValueError("each block must pair the same-numbered T and N attempts")
    for contract_id, campaign_id in (("T", time_campaign_id), ("N", token_campaign_id)):
        attempts = [row.campaign_attempt for row in rows if row.contract_id == contract_id]
        if attempts != list(range(1, attempts_per_condition + 1)):
            raise ValueError("campaign attempts are not a complete permutation")


def schedule_payload(**kwargs: object) -> dict[str, object]:
    rows = build_schedule(**kwargs)
    return {
        "schema": SCHEDULE_VERSION,
        "schedule_seed": kwargs.get("seed", SCHEDULE_SEED),
        "attempts_per_condition": kwargs.get("attempts_per_condition", ATTEMPTS_PER_CONDITION),
        "simultaneous_execution": False,
        "history_isolation": {
            "T": "runs/pair-a-fixed-time-seed3/history.jsonl",
            "N": "runs/pair-a-fixed-tokens-seed3/history.jsonl",
        },
        "blocks": [row.as_dict() for row in rows],
        "scientific_schedule_frozen_before_results": True,
    }


__all__ = ["SCHEDULE_SEED", "SCHEDULE_VERSION", "ScheduledAttempt", "build_schedule", "schedule_payload", "validate_schedule"]

"""Formal resource contracts shared by both study conditions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class ContractId(str, Enum):
    T = "T"
    N = "N"


MAX_TIMING_OVERRUN_SECONDS = 15.0


@dataclass(frozen=True)
class ResourceContract:
    """A contract is immutable once a campaign is registered."""

    contract_id: ContractId
    time_seconds: float | None = 300.0
    n_star: int | None = None

    def __post_init__(self) -> None:
        if self.contract_id is ContractId.T and self.time_seconds != 300.0:
            raise ValueError("the study timing contract is exactly 300.000 seconds")
        if self.contract_id is ContractId.N and self.time_seconds is not None:
            raise ValueError("CONTRACT N has no wall-clock resource field")
        if self.contract_id is ContractId.T and self.n_star is not None:
            raise ValueError("CONTRACT T cannot carry N_STAR")
        if self.contract_id is ContractId.N and self.n_star is not None:
            if not isinstance(self.n_star, int) or self.n_star <= 0:
                raise ValueError("N_STAR must be a positive integer")

    @property
    def requires_calibration(self) -> bool:
        return self.contract_id is ContractId.N and self.n_star is None

    def progress(self, *, measured_training_time: float, total_tokens: int) -> float:
        if measured_training_time < 0 or total_tokens < 0:
            raise ValueError("resource measurements cannot be negative")
        if self.contract_id is ContractId.T:
            return min(measured_training_time / (self.time_seconds or 300.0), 1.0)
        if self.n_star is None:
            raise ValueError("CONTRACT N requires N_STAR before execution")
        return min(total_tokens / self.n_star, 1.0)

    def validate_completion(self, *, measured_training_time: float, total_tokens: int) -> None:
        if self.contract_id is ContractId.T:
            if measured_training_time < self.time_seconds:
                raise ValueError("CONTRACT T candidate stopped before its measured 300-second budget")
            if measured_training_time > self.time_seconds + MAX_TIMING_OVERRUN_SECONDS:
                raise ValueError("CONTRACT T candidate exceeded the bounded update-boundary overrun")
        else:
            if self.n_star is None or total_tokens != self.n_star:
                raise ValueError("CONTRACT N candidate did not process exactly N_STAR tokens")


def parse_contract(value: str, n_star: int | None = None) -> ResourceContract:
    try:
        contract_id = ContractId(value.upper())
    except ValueError as exc:
        raise ValueError("contract must be T or N") from exc
    return ResourceContract(contract_id, time_seconds=300.0 if contract_id is ContractId.T else None, n_star=n_star)


def json_contract(contract: ResourceContract) -> dict[str, object]:
    return {
        "contract_id": contract.contract_id.value,
        "time_seconds": contract.time_seconds,
        "n_star": contract.n_star,
        "n_star_status": "pending_calibration" if contract.requires_calibration else "frozen",
    }

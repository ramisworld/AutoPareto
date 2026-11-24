"""Small agent-visible contract API staged with a candidate context."""

from __future__ import annotations

MAX_SEQ_LEN = 2048
ALLOWED_TOTAL_BATCH_SIZES = (32768, 65536, 131072, 262144, 524288)


def validate_batch_config(total_batch_size, device_batch_size, accumulation_steps, sequence_length=MAX_SEQ_LEN):
    if total_batch_size not in ALLOWED_TOTAL_BATCH_SIZES:
        raise ValueError("TOTAL_BATCH_SIZE is outside the common allowed set")
    if device_batch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("batch dimensions must be positive")
    if device_batch_size * sequence_length * accumulation_steps != total_batch_size:
        raise ValueError("TOTAL_BATCH_SIZE must equal DEVICE_BATCH_SIZE * sequence_length * accumulation_steps")


def accounted_tokens(optimizer_updates, total_batch_size):
    if optimizer_updates < 0:
        raise ValueError("optimizer updates cannot be negative")
    return optimizer_updates * total_batch_size


def contract_progress(contract_id, measured_training_time, total_tokens, n_star=None):
    """Declare the next protected optimizer boundary.

    Call this with the shown four-argument public signature immediately before
    every optimizer update. Under protected execution the runtime preserves
    this signature but derives authoritative timing and token counts from its
    observer; supplied numeric values are diagnostic only.
    """
    if contract_id == "T":
        return min(measured_training_time / 300.0, 1.0)
    if contract_id == "N" and n_star and n_star > 0:
        return min(total_tokens / n_star, 1.0)
    raise ValueError("CONTRACT N requires a positive N_STAR")


def contract_update_completed():
    """Protected boundary for optimizers that do not subclass torch Optimizer.

    The candidate calls this immediately after its real update.  The protected
    worker replaces this stub with an observer that has already counted the actual
    model forwards and token shapes; a candidate cannot supply token totals here.
    """
    raise RuntimeError("protected worker did not install the update boundary")


def estimate_training_flops(total_tokens, num_params, multiplier=6.0):
    return float(total_tokens) * float(num_params) * float(multiplier)


class TrainingLedger:
    """Standalone agent-visible counter; the protected launcher verifies its output."""

    def __init__(self, total_batch_size, device_batch_size, accumulation_steps, contract_id, n_star=None):
        validate_batch_config(total_batch_size, device_batch_size, accumulation_steps)
        self.total_batch_size = total_batch_size
        self.tokens_per_microbatch = device_batch_size * MAX_SEQ_LEN
        self.contract_id = contract_id
        self.n_star = n_star
        self.total_tokens = 0
        self.optimizer_updates = 0
        self.microbatches = 0

    def before_microbatch(self):
        if self.contract_id == "N" and self.total_tokens + self.tokens_per_microbatch > self.n_star:
            raise RuntimeError("partial fixed-token microbatch")

    def after_microbatch(self):
        self.total_tokens += self.tokens_per_microbatch
        self.microbatches += 1

    def after_optimizer_update(self):
        if self.total_tokens != (self.optimizer_updates + 1) * self.total_batch_size:
            raise RuntimeError("optimizer update is not backed by complete counted tokens")
        self.optimizer_updates += 1


__all__ = [
    "ALLOWED_TOTAL_BATCH_SIZES",
    "MAX_SEQ_LEN",
    "accounted_tokens",
    "contract_progress",
    "contract_update_completed",
    "validate_batch_config",
    "TrainingLedger",
    "estimate_training_flops",
]

from __future__ import annotations

import math
from dataclasses import dataclass

from reliquary.constants import MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW, M_ROLLOUTS


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def suggest_parallel_workers(
    cycle_p90_seconds: float,
    *,
    window_seconds: float = 45.0,
    submission_cap: int = MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
) -> int:
    if cycle_p90_seconds <= 0:
        return submission_cap
    cycles_per_window = window_seconds / cycle_p90_seconds
    if cycles_per_window <= 0:
        return submission_cap
    return max(1, math.ceil(submission_cap / cycles_per_window))


@dataclass
class LatencyStats:
    samples: int
    median_seconds: float
    p90_seconds: float


@dataclass
class LatencyReport:
    checkpoint_revision: str
    generation: LatencyStats
    proof_per_rollout: LatencyStats
    cycle_p90_seconds: float
    suggested_workers: int
    window_seconds: float
    submission_cap: int

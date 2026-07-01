from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reliquary.constants import M_ROLLOUTS, T_PROTO

from harness.scoring import ProbeOutcome

ProbeDecision = Literal["continue", "in_zone_band", "too_easy", "too_hard", "budget_exhausted"]


@dataclass(frozen=True)
class ProbeConfig:
    max_samples: int = 6
    max_probe_tokens: int = 512
    temperature: float = T_PROTO
    extreme_same_threshold: int = 3


@dataclass
class ProbeResult:
    prompt_idx: int
    p_hat: float
    probe_samples_used: int
    successes: int
    failures: int
    unknowns: int
    resolved_samples: int
    decision: ProbeDecision
    confidence: Literal["low", "medium", "high"]
    predicted_in_zone: bool
    outcomes: list[ProbeOutcome]
    rewards: list[float | None]


def binary_in_zone_band(*, bootstrap: bool = False) -> tuple[float, float]:
    """Inclusive p_hat band for binary OpenMath rewards (k successes out of M=8).

    OpenCode / fractional rewards need a different probe band — do not reuse
    this helper when extending the harness beyond OpenMath.
    """
    if bootstrap:
        return (1 / M_ROLLOUTS, 6 / M_ROLLOUTS)
    return (2 / M_ROLLOUTS, 6 / M_ROLLOUTS)


def predicts_in_zone(
    p_hat: float,
    *,
    bootstrap: bool = False,
    band: tuple[float, float] | None = None,
) -> bool:
    lo, hi = band if band is not None else binary_in_zone_band(bootstrap=bootstrap)
    return lo <= p_hat <= hi


def _p_hat_from_outcomes(outcomes: list[ProbeOutcome]) -> float:
    resolved = [o for o in outcomes if o != "unknown"]
    if not resolved:
        return 0.0
    successes = sum(1 for o in resolved if o == "success")
    return successes / len(resolved)


def _tail_streak(resolved: list[ProbeOutcome]) -> tuple[ProbeOutcome | None, int]:
    if not resolved:
        return None, 0
    last = resolved[-1]
    streak = 0
    for o in reversed(resolved):
        if o != last:
            break
        streak += 1
    return last, streak


def probe_step(
    outcomes: list[ProbeOutcome],
    *,
    config: ProbeConfig,
) -> tuple[ProbeDecision, float, Literal["low", "medium", "high"]]:
    n = len(outcomes)
    if n == 0:
        return "continue", 0.0, "low"

    resolved = [o for o in outcomes if o != "unknown"]
    p_hat = _p_hat_from_outcomes(outcomes)

    if resolved:
        has_success = any(o == "success" for o in resolved)
        has_failure = any(o == "failure" for o in resolved)
        if has_success and has_failure:
            return "in_zone_band", p_hat, "high"

    last_kind, streak = _tail_streak(resolved)
    if (
        streak >= config.extreme_same_threshold
        and last_kind in {"success", "failure"}
    ):
        if last_kind == "success":
            return "too_easy", p_hat, "high"
        return "too_hard", p_hat, "high"

    if n >= config.max_samples:
        return "budget_exhausted", p_hat, "medium"

    return "continue", p_hat, "low"

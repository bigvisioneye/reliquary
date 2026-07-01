from __future__ import annotations

from typing import Literal

from reliquary.environment.openmathinstruct import _compute_omi_reward

ProbeOutcome = Literal["success", "failure", "unknown"]


def compute_openmath_reward(problem: dict, completion_text: str) -> float:
    return float(_compute_omi_reward(problem, completion_text))


def classify_probe_outcome(
    problem: dict,
    completion_text: str,
    *,
    truncated: bool,
) -> ProbeOutcome:
    """Classify a probe sample for difficulty estimation.

    Truncated completions (cap hit before natural EOS) are ``unknown``: they
    often lack a ``\\boxed{}`` answer not because the model failed but because
    the probe budget cut the rollout short. Counting those as failures biases
    p_hat downward.

    Natural-EOS completions without a boxed answer are real failures (0).
    """
    if truncated:
        return "unknown"
    reward = compute_openmath_reward(problem, completion_text)
    return "success" if reward > 0.0 else "failure"


def compute_sigma(rewards: list[float]) -> float:
    from reliquary.validator.verifier import rewards_std

    return float(rewards_std(rewards))


def in_zone(sigma: float, *, bootstrap: bool = False) -> bool:
    from reliquary.validator.verifier import is_in_zone

    return bool(is_in_zone(sigma, bootstrap=bootstrap))

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.generation import generate_m_rollouts
from harness.scoring import compute_openmath_reward, compute_sigma, in_zone


@dataclass
class TrueLabelResult:
    prompt_idx: int
    checkpoint_repo_id: str
    checkpoint_revision: str
    k: int
    sigma: float
    in_zone: bool
    rewards: list[float]
    rollouts: list[str]


def true_label(
    *,
    model: Any,
    tokenizer: Any,
    env: Any,
    prompt_idx: int,
    checkpoint_repo_id: str,
    checkpoint_revision: str,
    bootstrap: bool = False,
) -> TrueLabelResult:
    problem = env.get_problem(prompt_idx)
    rollouts = generate_m_rollouts(model=model, tokenizer=tokenizer, prompt=problem["prompt"])
    rewards = [compute_openmath_reward(problem, completion) for completion in rollouts]
    sigma = compute_sigma(rewards)
    # OpenMath only: rewards are binary {0,1}, so k = count(r>0) and σ = √(p(1−p)).
    # For fractional envs (e.g. OpenCode), use rewards_std directly and do not use k.
    k = sum(1 for r in rewards if r > 0.0)
    return TrueLabelResult(
        prompt_idx=prompt_idx,
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        k=k,
        sigma=sigma,
        in_zone=in_zone(sigma, bootstrap=bootstrap),
        rewards=rewards,
        rollouts=rollouts,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

from harness.generation import generate_m_rollout_records
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
    label_rollouts: int
    truncated_unscorable_rollouts: int


def true_label(
    *,
    model: Any,
    tokenizer: Any,
    env: Any,
    prompt_idx: int,
    checkpoint_repo_id: str,
    checkpoint_revision: str,
    bootstrap: bool = False,
    max_label_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    gen_backend: str = "hf",
    vllm_engine: Any | None = None,
) -> TrueLabelResult:
    from reliquary.environment.openmathinstruct import _last_boxed_only_string

    problem = env.get_problem(prompt_idx)
    records = generate_m_rollout_records(
        model=model,
        tokenizer=tokenizer,
        prompt=problem["prompt"],
        max_new_tokens=max_label_tokens,
        gen_backend=gen_backend,  # type: ignore[arg-type]
        vllm_engine=vllm_engine,
    )
    rollouts = [tokenizer.decode(r.completion_token_ids) for r in records]
    rewards = [compute_openmath_reward(problem, completion) for completion in rollouts]
    truncated_unscorable = sum(
        1
        for rec, completion in zip(records, rollouts, strict=True)
        if rec.truncated and _last_boxed_only_string(completion) is None
    )
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
        label_rollouts=len(rollouts),
        truncated_unscorable_rollouts=truncated_unscorable,
    )

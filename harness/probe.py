from __future__ import annotations

from typing import Any

from harness.probe_logic import (
    ProbeConfig,
    ProbeResult,
    predicts_in_zone,
    probe_step,
)
from harness.scoring import ProbeOutcome, classify_probe_outcome


def run_probe(
    *,
    model: Any,
    tokenizer: Any,
    problem: dict,
    prompt_idx: int,
    config: ProbeConfig | None = None,
    bootstrap: bool = False,
    band: tuple[float, float] | None = None,
) -> ProbeResult:
    from harness.generation import generate_rollout_tokens

    cfg = config or ProbeConfig()
    outcomes: list[ProbeOutcome] = []
    rewards: list[float | None] = []
    decision = "continue"
    p_hat = 0.0
    confidence: str = "low"

    while decision == "continue":
        record = generate_rollout_tokens(
            model,
            tokenizer,
            problem["prompt"],
            max_new_tokens=cfg.max_probe_tokens,
            temperature=cfg.temperature,
        )
        completion = tokenizer.decode(record.completion_token_ids)
        outcome = classify_probe_outcome(
            problem, completion, truncated=record.truncated,
        )
        outcomes.append(outcome)
        if outcome == "unknown":
            rewards.append(None)
        else:
            rewards.append(1.0 if outcome == "success" else 0.0)
        decision, p_hat, confidence = probe_step(outcomes, config=cfg)

    resolved = [o for o in outcomes if o != "unknown"]
    successes = sum(1 for o in resolved if o == "success")
    failures = sum(1 for o in resolved if o == "failure")
    predicted = decision == "in_zone_band" or (
        decision == "budget_exhausted" and predicts_in_zone(p_hat, bootstrap=bootstrap, band=band)
    )
    return ProbeResult(
        prompt_idx=prompt_idx,
        p_hat=p_hat,
        probe_samples_used=len(outcomes),
        successes=successes,
        failures=failures,
        unknowns=len(outcomes) - len(resolved),
        resolved_samples=len(resolved),
        decision=decision,
        confidence=confidence,  # type: ignore[arg-type]
        predicted_in_zone=predicted,
        outcomes=outcomes,
        rewards=rewards,
    )

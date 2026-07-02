from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from reliquary.constants import M_ROLLOUTS

from harness.generation import (
    generate_m_rollout_dicts,
    generate_m_rollouts,
    generate_rollout_tokens,
    rollout_tokens_to_generation_dict,
)
from harness.grail_build import build_grail_commit
from harness.grail_check import check_grail_commit
from harness.latency_metrics import LatencyReport, LatencyStats, percentile, suggest_parallel_workers


def _time_proof_only(
    *,
    model: Any,
    tokenizer: Any,
    generation: dict,
    randomness: str,
) -> float:
    t0 = time.perf_counter()
    commit = build_grail_commit(model, tokenizer, generation, randomness)
    check_grail_commit(commit, model, randomness, tokenizer=tokenizer)
    return time.perf_counter() - t0


def measure_latency(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    randomness: str,
    checkpoint_revision: str,
    generation_runs: int = 3,
    proof_runs: int = 3,
    window_seconds: float = 45.0,
    proofs_batched: bool = False,
    gen_backend: str = "hf",
    vllm_engine: Any | None = None,
) -> LatencyReport:
    gen_times: list[float] = []
    for _ in range(generation_runs):
        t0 = time.perf_counter()
        generate_m_rollouts(
            model,
            tokenizer,
            prompt,
            n_rollouts=M_ROLLOUTS,
            gen_backend=gen_backend,  # type: ignore[arg-type]
            vllm_engine=vllm_engine,
        )
        gen_times.append(time.perf_counter() - t0)

    proof_times: list[float] = []
    proof_batch_times: list[float] = []

    if proofs_batched:
        batch_generations = generate_m_rollout_dicts(
            model,
            tokenizer,
            prompt,
            n_rollouts=M_ROLLOUTS,
            gen_backend=gen_backend,  # type: ignore[arg-type]
            vllm_engine=vllm_engine,
        )
        for _ in range(proof_runs):
            t0 = time.perf_counter()
            for generation in batch_generations:
                commit = build_grail_commit(model, tokenizer, generation, randomness)
                check_grail_commit(commit, model, randomness, tokenizer=tokenizer)
            proof_batch_times.append(time.perf_counter() - t0)
    else:
        proof_generation = rollout_tokens_to_generation_dict(
            generate_rollout_tokens(
                model,
                tokenizer,
                prompt,
                gen_backend=gen_backend,  # type: ignore[arg-type]
                vllm_engine=vllm_engine,
            ),
        )
        for _ in range(proof_runs):
            proof_times.append(
                _time_proof_only(
                    model=model,
                    tokenizer=tokenizer,
                    generation=proof_generation,
                    randomness=randomness,
                )
            )

    gen_median = percentile(gen_times, 0.5)
    gen_p90 = percentile(gen_times, 0.9)

    if proofs_batched:
        proof_batch_median = percentile(proof_batch_times, 0.5)
        proof_batch_p90 = percentile(proof_batch_times, 0.9)
        cycle_p90 = gen_p90 + proof_batch_p90
        proof_per_rollout = LatencyStats(samples=0, median_seconds=0.0, p90_seconds=0.0)
        proof_batch = LatencyStats(
            samples=len(proof_batch_times),
            median_seconds=proof_batch_median,
            p90_seconds=proof_batch_p90,
        )
    else:
        proof_median = percentile(proof_times, 0.5)
        proof_p90 = percentile(proof_times, 0.9)
        cycle_p90 = gen_p90 + proof_p90 * M_ROLLOUTS
        proof_per_rollout = LatencyStats(
            samples=len(proof_times),
            median_seconds=proof_median,
            p90_seconds=proof_p90,
        )
        proof_batch = None

    return LatencyReport(
        checkpoint_revision=checkpoint_revision,
        generation=LatencyStats(
            samples=len(gen_times), median_seconds=gen_median, p90_seconds=gen_p90,
        ),
        proof_per_rollout=proof_per_rollout,
        proof_batch=proof_batch,
        cycle_p90_seconds=cycle_p90,
        suggested_workers=suggest_parallel_workers(cycle_p90, window_seconds=window_seconds),
        window_seconds=window_seconds,
        submission_cap=M_ROLLOUTS,
        generation_is_batched=True,
        proofs_batched=proofs_batched,
    )


def latency_report_dict(report: LatencyReport) -> dict:
    payload = asdict(report)
    if report.proof_batch is None:
        payload.pop("proof_batch", None)
    return payload

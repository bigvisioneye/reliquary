from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from reliquary.constants import M_ROLLOUTS

from harness.generation import generate_m_rollouts
from harness.grail_build import build_grail_commit
from harness.grail_check import check_grail_commit
from harness.latency_metrics import LatencyReport, LatencyStats, percentile, suggest_parallel_workers


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
) -> LatencyReport:
    from harness.grail_check import make_generation_dict

    gen_times: list[float] = []
    for _ in range(generation_runs):
        t0 = time.perf_counter()
        generate_m_rollouts(model, tokenizer, prompt, n_rollouts=M_ROLLOUTS)
        gen_times.append(time.perf_counter() - t0)

    proof_times: list[float] = []
    for _ in range(proof_runs):
        t0 = time.perf_counter()
        generation = make_generation_dict(model, tokenizer, prompt)
        commit = build_grail_commit(model, tokenizer, generation, randomness)
        check_grail_commit(commit, model, randomness, tokenizer=tokenizer)
        proof_times.append(time.perf_counter() - t0)

    gen_median = percentile(gen_times, 0.5)
    gen_p90 = percentile(gen_times, 0.9)
    proof_median = percentile(proof_times, 0.5)
    proof_p90 = percentile(proof_times, 0.9)
    cycle_p90 = gen_p90 + proof_p90 * M_ROLLOUTS

    return LatencyReport(
        checkpoint_revision=checkpoint_revision,
        generation=LatencyStats(
            samples=len(gen_times), median_seconds=gen_median, p90_seconds=gen_p90,
        ),
        proof_per_rollout=LatencyStats(
            samples=len(proof_times), median_seconds=proof_median, p90_seconds=proof_p90,
        ),
        cycle_p90_seconds=cycle_p90,
        suggested_workers=suggest_parallel_workers(cycle_p90, window_seconds=window_seconds),
        window_seconds=window_seconds,
        submission_cap=M_ROLLOUTS,
    )


def latency_report_dict(report: LatencyReport) -> dict:
    return asdict(report)

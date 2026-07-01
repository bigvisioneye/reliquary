from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harness.calibrate import run_calibration
from harness.calibrate_metrics import CalibrationReport, write_calibration_csv
from harness.latency import latency_report_dict, measure_latency
from harness.grail_check import GrailCheckResult, run_grail_fidelity
from harness.latency_metrics import LatencyReport
from harness.probe_logic import ProbeConfig
from harness.slice import filter_prompt_indices, sample_prompt_indices, slice_for_window


@dataclass
class FullHarnessReport:
    checkpoint_repo_id: str
    checkpoint_revision: str
    randomness: str
    enforce_slice: bool
    prompt_indices: list[int]
    slice_bounds: tuple[int, int]
    calibration: CalibrationReport
    grail_checks: list[dict[str, Any]]
    grail_all_passed: bool
    latency: LatencyReport | None
    jitter_band_recommendation: tuple[float, float] | None
    suggested_workers: int | None


def _aggregate_jitter_band(calibration: CalibrationReport) -> tuple[float, float] | None:
    if not calibration.jitter:
        return None
    lo = min(j.recommended_band[0] for j in calibration.jitter)
    hi = max(j.recommended_band[1] for j in calibration.jitter)
    return (lo, hi)


def run_full_report(
    *,
    model: Any,
    tokenizer: Any,
    env: Any,
    checkpoint_repo_id: str,
    checkpoint_revision: str,
    randomness: str,
    prompt_indices: list[int] | None = None,
    prompt_count: int = 10,
    seed: int = 0,
    enforce_slice: bool = False,
    env_name: str = "openmathinstruct",
    probe_config: ProbeConfig | None = None,
    bootstrap: bool = False,
    jitter_repeats: int = 2,
    run_grail: bool = True,
    run_latency: bool = True,
    grail_rollouts: int = 1,
    latency_prompt_idx: int | None = None,
    generation_runs: int = 3,
    proof_runs: int = 3,
    window_seconds: float = 45.0,
) -> FullHarnessReport:
    import random

    rng = random.Random(seed)
    bounds = slice_for_window(randomness, env_name, len(env))
    if prompt_indices is None:
        indices = sample_prompt_indices(
            env,
            count=prompt_count,
            randomness=randomness,
            env_name=env_name,
            enforce_slice=enforce_slice,
            rng=rng,
        )
    else:
        indices = filter_prompt_indices(prompt_indices, bounds, enforce=enforce_slice)

    calibration = run_calibration(
        model=model,
        tokenizer=tokenizer,
        env=env,
        prompt_indices=indices,
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        probe_config=probe_config,
        bootstrap=bootstrap,
        jitter_repeats=jitter_repeats,
    )

    grail_results: list[GrailCheckResult] = []
    if run_grail and indices:
        probe_prompt = env.get_problem(indices[0])["prompt"]
        grail_results = run_grail_fidelity(
            model=model,
            tokenizer=tokenizer,
            prompt=probe_prompt,
            randomness=randomness,
            rollouts=grail_rollouts,
        )

    latency_report: LatencyReport | None = None
    if run_latency:
        idx = latency_prompt_idx if latency_prompt_idx is not None else indices[0]
        problem = env.get_problem(idx)
        latency_report = measure_latency(
            model=model,
            tokenizer=tokenizer,
            prompt=problem["prompt"],
            randomness=randomness,
            checkpoint_revision=checkpoint_revision,
            generation_runs=generation_runs,
            proof_runs=proof_runs,
            window_seconds=window_seconds,
        )

    grail_payload = [asdict(r) for r in grail_results]
    jitter_band = _aggregate_jitter_band(calibration)
    suggested = latency_report.suggested_workers if latency_report else None

    return FullHarnessReport(
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        randomness=randomness,
        enforce_slice=enforce_slice,
        prompt_indices=indices,
        slice_bounds=bounds,
        calibration=calibration,
        grail_checks=grail_payload,
        grail_all_passed=all(r.passed for r in grail_results) if grail_results else False,
        latency=latency_report,
        jitter_band_recommendation=jitter_band,
        suggested_workers=suggested,
    )


def write_full_report(path: str | Path, report: FullHarnessReport) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_repo_id": report.checkpoint_repo_id,
        "checkpoint_revision": report.checkpoint_revision,
        "randomness": report.randomness,
        "enforce_slice": report.enforce_slice,
        "slice_bounds": list(report.slice_bounds),
        "prompt_indices": report.prompt_indices,
        "calibration": {
            "n_prompts": report.calibration.n_prompts,
            "metrics": asdict(report.calibration.metrics),
            "compute_saved_per_in_zone": report.calibration.compute_saved_per_in_zone,
            "pr_curve": [asdict(p) for p in report.calibration.pr_curve],
            "jitter": [asdict(j) for j in report.calibration.jitter],
        },
        "grail_all_passed": report.grail_all_passed,
        "grail_checks": report.grail_checks,
        "latency": latency_report_dict(report.latency) if report.latency else None,
        "jitter_band_recommendation": report.jitter_band_recommendation,
        "suggested_workers": report.suggested_workers,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_report_artifacts(
    report: FullHarnessReport,
    *,
    report_json: str | Path,
    calibration_csv: str | Path,
) -> None:
    write_full_report(report_json, report)
    write_calibration_csv(calibration_csv, report.calibration.rows)

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harness.calibrate import run_calibration
from harness.calibrate_metrics import (
    CalibrationReport,
    calibration_report_dict,
    write_calibration_csv,
    write_calibration_report,
)
from harness.latency import latency_report_dict, measure_latency
from harness.grail_check import GrailCheckResult, run_grail_fidelity
from harness.latency_metrics import LatencyReport
from harness.probe_logic import ProbeConfig
from harness.sampling import SampleMode, resolve_calibration_indices
from harness.slice import slice_for_window


@dataclass
class FullHarnessReport:
    checkpoint_repo_id: str
    checkpoint_revision: str
    randomness: str
    enforce_slice: bool
    sample_mode: str
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
    sample_mode: SampleMode = "uniform",
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
    proofs_batched: bool = False,
) -> FullHarnessReport:
    import random

    rng = random.Random(seed)
    bounds = slice_for_window(randomness, env_name, len(env))
    if prompt_indices is None:
        indices = resolve_calibration_indices(
            env,
            count=prompt_count,
            seed=seed,
            randomness=randomness,
            env_name=env_name,
            sample_mode=sample_mode,
            enforce_slice=enforce_slice,
            rng=rng,
        )
    else:
        indices = prompt_indices

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
        randomness=randomness,
        enforce_slice=enforce_slice,
        sample_mode=sample_mode,
        requested_prompts=prompt_count,
        env_name=env_name,
        rng=rng,
    )

    grail_results: list[GrailCheckResult] = []
    if run_grail and calibration.rows:
        first_idx = calibration.rows[0].idx
        probe_prompt = env.get_problem(first_idx)["prompt"]
        grail_results = run_grail_fidelity(
            model=model,
            tokenizer=tokenizer,
            prompt=probe_prompt,
            randomness=randomness,
            rollouts=grail_rollouts,
        )

    latency_report: LatencyReport | None = None
    if run_latency and calibration.rows:
        idx = latency_prompt_idx if latency_prompt_idx is not None else calibration.rows[0].idx
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
            proofs_batched=proofs_batched,
        )

    grail_payload = [asdict(r) for r in grail_results]
    jitter_band = _aggregate_jitter_band(calibration)
    suggested = latency_report.suggested_workers if latency_report else None

    return FullHarnessReport(
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        randomness=randomness,
        enforce_slice=enforce_slice,
        sample_mode=calibration.sample_mode,
        prompt_indices=[r.idx for r in calibration.rows],
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
        "sample_mode": report.sample_mode,
        "slice_bounds": list(report.slice_bounds),
        "prompt_indices": report.prompt_indices,
        "calibration": calibration_report_dict(report.calibration),
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
    calibration_report_json: str | Path | None = None,
) -> None:
    write_full_report(report_json, report)
    write_calibration_csv(calibration_csv, report.calibration.rows)
    cal_path = calibration_report_json or (Path(report_json).parent / "calibration_report.json")
    write_calibration_report(cal_path, report.calibration)

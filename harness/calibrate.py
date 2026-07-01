from __future__ import annotations

import random
from typing import Any

from harness.calibrate_metrics import (
    CalibrationReport,
    CalibrationRow,
    build_pr_curve,
    classification_metrics,
    compute_saved_per_in_zone,
    summarize_jitter,
)
from harness.label import true_label
from harness.probe import run_probe
from harness.probe_logic import ProbeConfig
from harness.slice import filter_prompt_indices, slice_for_window


def run_calibration(
    *,
    model: Any,
    tokenizer: Any,
    env: Any,
    prompt_indices: list[int],
    checkpoint_repo_id: str,
    checkpoint_revision: str,
    probe_config: ProbeConfig | None = None,
    bootstrap: bool = False,
    jitter_repeats: int = 0,
    randomness: str | None = None,
    enforce_slice: bool = False,
    env_name: str = "openmathinstruct",
    rng: random.Random | None = None,
) -> CalibrationReport:
    del rng
    cfg = probe_config or ProbeConfig()
    bounds = (
        slice_for_window(randomness, env_name, len(env))
        if randomness
        else (0, len(env))
    )
    indices = filter_prompt_indices(prompt_indices, bounds, enforce=enforce_slice)

    rows: list[CalibrationRow] = []
    jitter_summaries = []

    for idx in indices:
        problem = env.get_problem(idx)
        probe = run_probe(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            prompt_idx=idx,
            config=cfg,
            bootstrap=bootstrap,
        )
        label = true_label(
            model=model,
            tokenizer=tokenizer,
            env=env,
            prompt_idx=idx,
            checkpoint_repo_id=checkpoint_repo_id,
            checkpoint_revision=checkpoint_revision,
            bootstrap=bootstrap,
        )
        rows.append(
            CalibrationRow(
                idx=idx,
                k=label.k,
                true_sigma=label.sigma,
                in_zone=label.in_zone,
                p_hat=probe.p_hat,
                probe_samples_used=probe.probe_samples_used,
                decision=probe.decision,
                predicted_in_zone=probe.predicted_in_zone,
            )
        )

        if jitter_repeats > 0:
            k_vals: list[int] = []
            in_zone_flags: list[bool] = []
            for _ in range(jitter_repeats):
                j = true_label(
                    model=model,
                    tokenizer=tokenizer,
                    env=env,
                    prompt_idx=idx,
                    checkpoint_repo_id=checkpoint_repo_id,
                    checkpoint_revision=checkpoint_revision,
                    bootstrap=bootstrap,
                )
                k_vals.append(j.k)
                in_zone_flags.append(j.in_zone)
            jitter_summaries.append(
                summarize_jitter(idx, k_vals, in_zone_flags, repeats=jitter_repeats)
            )

    p_hats = [r.p_hat for r in rows]
    actual = [r.in_zone for r in rows]
    predicted = [r.predicted_in_zone for r in rows]
    metrics = classification_metrics(predicted, actual)
    pr_curve = build_pr_curve(p_hats, actual)
    saved = compute_saved_per_in_zone(rows)

    return CalibrationReport(
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        n_prompts=len(rows),
        metrics=metrics,
        pr_curve=pr_curve,
        compute_saved_per_in_zone=saved,
        jitter=jitter_summaries,
        rows=rows,
    )

from __future__ import annotations

import random
import time
from typing import Any

from harness.calibrate_metrics import (
    CalibrationReport,
    CalibrationRow,
    build_calibration_balance,
    build_pr_curve,
    classification_metrics,
    compute_saved_per_in_zone,
    summarize_jitter,
    validate_calibration_report,
)
from harness.label import true_label
from harness.log import get_logger
from harness.probe import run_probe
from harness.probe_logic import ProbeConfig, ProbeResult
from harness.sampling import SampleMode, effective_sample_mode
from harness.slice import filter_prompt_indices, slice_for_window

logger = get_logger("calibrate")


def _prefilter_category(probe: ProbeResult) -> str:
    if probe.decision == "in_zone_band":
        return "frontier"
    if probe.successes > 0 and probe.failures > 0:
        return "frontier"
    if probe.decision in ("too_easy", "too_hard"):
        return "extreme"
    return "other"


def prefilter_prompt_indices(
    *,
    model: Any,
    tokenizer: Any,
    env: Any,
    candidates: list[int],
    count: int,
    probe_config: ProbeConfig,
    rng: random.Random,
    gen_backend: str = "hf",
    vllm_engine: Any | None = None,
) -> list[int]:
    cheap_cfg = ProbeConfig(
        max_samples=2,
        max_probe_tokens=probe_config.max_probe_tokens,
        temperature=probe_config.temperature,
        extreme_same_threshold=probe_config.extreme_same_threshold,
    )
    buckets: dict[str, list[int]] = {"frontier": [], "other": [], "extreme": []}
    n_candidates = len(candidates)
    logger.info(
        "prefilter: probing %d candidates (cheap 2-sample probe, max_probe_tokens=%d)",
        n_candidates,
        cheap_cfg.max_probe_tokens,
    )
    prefilter_started = time.perf_counter()
    for i, idx in enumerate(candidates, start=1):
        logger.info("prefilter: candidate %d/%d prompt_idx=%d", i, n_candidates, idx)
        t0 = time.perf_counter()
        problem = env.get_problem(idx)
        probe = run_probe(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            prompt_idx=idx,
            config=cheap_cfg,
            gen_backend=gen_backend,  # type: ignore[arg-type]
            vllm_engine=vllm_engine,
        )
        category = _prefilter_category(probe)
        buckets[category].append(idx)
        logger.info(
            "prefilter: candidate %d/%d done in %.1fs category=%s decision=%s p_hat=%.3f",
            i,
            n_candidates,
            time.perf_counter() - t0,
            category,
            probe.decision,
            probe.p_hat,
        )

    for pool in buckets.values():
        rng.shuffle(pool)

    selected: list[int] = []
    target_frontier = max(1, count * 2 // 5)
    target_other = max(1, count * 2 // 5)
    picked = {"frontier": 0, "other": 0, "extreme": 0}

    for category in ("frontier", "other", "extreme"):
        limit = (
            target_frontier
            if category == "frontier"
            else target_other if category == "other" else count
        )
        for idx in buckets[category]:
            if len(selected) >= count:
                break
            if category in ("frontier", "other") and picked[category] >= limit:
                continue
            if idx not in selected:
                selected.append(idx)
                picked[category] += 1

    if len(selected) < count:
        for category in ("frontier", "other", "extreme"):
            for idx in buckets[category]:
                if len(selected) >= count:
                    break
                if idx not in selected:
                    selected.append(idx)
    selected = selected[:count]
    logger.info(
        "prefilter: selected %d/%d prompts in %.1fs "
        "(frontier=%d other=%d extreme=%d) indices=%s",
        len(selected),
        count,
        time.perf_counter() - prefilter_started,
        picked["frontier"],
        picked["other"],
        picked["extreme"],
        selected,
    )
    return selected


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
    sample_mode: SampleMode = "uniform",
    requested_prompts: int = 0,
    max_label_tokens: int = 2048,
    gen_backend: str = "hf",
    vllm_engine: Any | None = None,
    env_name: str = "openmathinstruct",
    rng: random.Random | None = None,
) -> CalibrationReport:
    rng = rng or random.Random()
    cfg = probe_config or ProbeConfig()
    mode = effective_sample_mode(sample_mode, enforce_slice)
    bounds = (
        slice_for_window(randomness, env_name, len(env))
        if randomness
        else (0, len(env))
    )

    indices = list(prompt_indices)
    logger.info(
        "calibration start: sample_mode=%s requested=%d candidates=%d "
        "slice=[%d,%d) max_label_tokens=%d gen_backend=%s",
        mode,
        requested_prompts or len(indices),
        len(indices),
        bounds[0],
        bounds[1],
        max_label_tokens,
        gen_backend,
    )
    if mode == "prefilter":
        indices = prefilter_prompt_indices(
            model=model,
            tokenizer=tokenizer,
            env=env,
            candidates=indices,
            count=requested_prompts or len(indices),
            probe_config=cfg,
            rng=rng,
            gen_backend=gen_backend,
            vllm_engine=vllm_engine,
        )
    elif mode == "slice":
        before = len(indices)
        indices = filter_prompt_indices(indices, bounds, enforce=True)
        if before != len(indices):
            logger.info("slice filter: %d -> %d prompts", before, len(indices))

    n_prompts = len(indices)
    logger.info("calibrating %d prompts (probe then true_label per prompt)", n_prompts)
    calibration_started = time.perf_counter()

    rows: list[CalibrationRow] = []
    jitter_summaries = []
    probe_samples_total = 0
    probe_unknowns_total = 0
    label_rollouts_total = 0
    label_truncated_unscorable_total = 0

    for prompt_i, idx in enumerate(indices, start=1):
        prompt_started = time.perf_counter()
        logger.info(
            "prompt %d/%d: idx=%d — running probe (max_samples=%d max_probe_tokens=%d)",
            prompt_i,
            n_prompts,
            idx,
            cfg.max_samples,
            cfg.max_probe_tokens,
        )
        problem = env.get_problem(idx)
        probe_t0 = time.perf_counter()
        probe = run_probe(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            prompt_idx=idx,
            config=cfg,
            bootstrap=bootstrap,
            gen_backend=gen_backend,  # type: ignore[arg-type]
            vllm_engine=vllm_engine,
        )
        probe_samples_total += probe.probe_samples_used
        probe_unknowns_total += probe.unknowns
        logger.info(
            "prompt %d/%d: probe done in %.1fs decision=%s p_hat=%.3f "
            "samples=%d unknowns=%d predicted_in_zone=%s",
            prompt_i,
            n_prompts,
            time.perf_counter() - probe_t0,
            probe.decision,
            probe.p_hat,
            probe.probe_samples_used,
            probe.unknowns,
            probe.predicted_in_zone,
        )
        logger.info(
            "prompt %d/%d: running true_label (8 rollouts, max_label_tokens=%d)",
            prompt_i,
            n_prompts,
            max_label_tokens,
        )
        label_t0 = time.perf_counter()
        label = true_label(
            model=model,
            tokenizer=tokenizer,
            env=env,
            prompt_idx=idx,
            checkpoint_repo_id=checkpoint_repo_id,
            checkpoint_revision=checkpoint_revision,
            bootstrap=bootstrap,
            max_label_tokens=max_label_tokens,
            gen_backend=gen_backend,
            vllm_engine=vllm_engine,
        )
        label_rollouts_total += label.label_rollouts
        label_truncated_unscorable_total += label.truncated_unscorable_rollouts
        logger.info(
            "prompt %d/%d: label done in %.1fs k=%d sigma=%.3f in_zone=%s "
            "trunc_unscorable=%d/%d (prompt total %.1fs, elapsed %.1fs)",
            prompt_i,
            n_prompts,
            time.perf_counter() - label_t0,
            label.k,
            label.sigma,
            label.in_zone,
            label.truncated_unscorable_rollouts,
            label.label_rollouts,
            time.perf_counter() - prompt_started,
            time.perf_counter() - calibration_started,
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
            logger.info(
                "prompt %d/%d: jitter repeats=%d",
                prompt_i,
                n_prompts,
                jitter_repeats,
            )
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
                    max_label_tokens=max_label_tokens,
                    gen_backend=gen_backend,
                    vllm_engine=vllm_engine,
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
    balance = build_calibration_balance(
        rows,
        probe_samples_total=probe_samples_total,
        probe_unknowns_total=probe_unknowns_total,
        label_rollouts_total=label_rollouts_total,
        label_truncated_unscorable_total=label_truncated_unscorable_total,
    )

    report = CalibrationReport(
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        n_prompts=len(rows),
        requested_prompts=requested_prompts or len(prompt_indices),
        sample_mode=mode,
        metrics=metrics,
        pr_curve=pr_curve,
        compute_saved_per_in_zone=saved,
        balance=balance,
        jitter=jitter_summaries,
        rows=rows,
    )
    validate_calibration_report(report)
    logger.info(
        "calibration complete: n_prompts=%d precision=%.3f recall=%.3f f1=%.3f "
        "unknown_rate=%.3f label_truncation_rate=%.3f total_time=%.1fs",
        report.n_prompts,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        balance.unknown_rate,
        balance.label_truncation_rate,
        time.perf_counter() - calibration_started,
    )
    return report

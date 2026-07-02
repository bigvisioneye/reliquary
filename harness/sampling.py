from __future__ import annotations

import random
from typing import Any, Literal

from harness.slice import filter_prompt_indices, sample_prompt_indices, slice_for_window

SampleMode = Literal["uniform", "slice", "prefilter"]


def effective_sample_mode(sample_mode: SampleMode, enforce_slice: bool) -> SampleMode:
    if enforce_slice and sample_mode == "uniform":
        return "slice"
    return sample_mode


def resolve_calibration_indices(
    env: Any,
    *,
    count: int,
    seed: int,
    randomness: str | None,
    env_name: str = "openmathinstruct",
    sample_mode: SampleMode = "uniform",
    enforce_slice: bool = False,
    explicit_prompts: str = "",
    rng: random.Random | None = None,
) -> list[int]:
    """Resolve prompt indices for calibration/report using a single sampling policy."""
    mode = effective_sample_mode(sample_mode, enforce_slice)
    rng = rng or random.Random(seed)

    if explicit_prompts.strip():
        indices = [int(p.strip()) for p in explicit_prompts.split(",") if p.strip()]
        if mode == "slice":
            if not randomness:
                raise ValueError("--sample-mode slice requires --randomness")
            bounds = slice_for_window(randomness, env_name, len(env))
            indices = filter_prompt_indices(indices, bounds, enforce=True)
        return indices

    if mode == "uniform":
        return sample_prompt_indices(
            env,
            count=count,
            randomness=None,
            env_name=env_name,
            enforce_slice=False,
            rng=rng,
        )

    if mode in ("slice", "prefilter"):
        if not randomness:
            raise ValueError(f"--sample-mode {mode} requires --randomness")
        pool_size = max(count * 5, count + 20) if mode == "prefilter" else count
        return sample_prompt_indices(
            env,
            count=pool_size,
            randomness=randomness,
            env_name=env_name,
            enforce_slice=True,
            rng=rng,
        )

    raise ValueError(f"unknown sample_mode: {sample_mode}")

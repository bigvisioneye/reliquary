from __future__ import annotations

import random
from typing import Any

from reliquary.constants import PROMPT_RANGE_SIZE
from reliquary.shared.prompt_range import window_prompt_range


def slice_for_window(
    randomness: str,
    env_name: str,
    universe_n: int,
    *,
    size: int = PROMPT_RANGE_SIZE,
) -> tuple[int, int]:
    return window_prompt_range(randomness, env_name, universe_n, size)


def in_slice(prompt_idx: int, slice_bounds: tuple[int, int]) -> bool:
    lo, hi = slice_bounds
    return lo <= prompt_idx < hi


def filter_prompt_indices(
    indices: list[int],
    slice_bounds: tuple[int, int],
    *,
    enforce: bool,
) -> list[int]:
    if not enforce:
        return list(indices)
    lo, hi = slice_bounds
    return [idx for idx in indices if lo <= idx < hi]


def sample_prompt_indices(
    env: Any,
    *,
    count: int,
    randomness: str | None = None,
    env_name: str = "openmathinstruct",
    enforce_slice: bool = False,
    rng: random.Random | None = None,
    max_attempts: int = 10_000,
) -> list[int]:
    """Sample distinct prompt indices, optionally restricted to the window slice."""
    rng = rng or random.Random()
    universe_n = len(env)
    bounds = (
        slice_for_window(randomness, env_name, universe_n)
        if randomness
        else (0, universe_n)
    )
    lo, hi = bounds
    span = hi - lo if enforce_slice and randomness else universe_n
    start = lo if enforce_slice and randomness else 0

    if span <= 0:
        raise ValueError("empty prompt slice for sampling")

    chosen: list[int] = []
    seen: set[int] = set()
    attempts = 0
    while len(chosen) < count and attempts < max_attempts:
        attempts += 1
        idx = start + rng.randrange(span)
        if enforce_slice and randomness and not in_slice(idx, bounds):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        chosen.append(idx)
    if len(chosen) < count:
        raise RuntimeError(
            f"could only sample {len(chosen)}/{count} distinct prompts in slice"
        )
    return chosen

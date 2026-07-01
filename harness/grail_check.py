from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from reliquary.constants import (
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    PROOF_SKETCH_TOLERANCE_BASE,
    PROOF_SKETCH_TOLERANCE_GROWTH,
)


@dataclass
class SketchMargin:
    position: int
    sketch_diff: int
    tolerance: int
    margin: int
    passed: bool
    is_fallback_estimate: bool = False


@dataclass
class GrailCheckResult:
    passed: bool
    sketch_diff_max: int
    worst_margin: int
    margin_is_fallback: bool
    positions_checked: int
    positions_passed: int
    sketch_margins: list[SketchMargin]
    randomness: str


def _margins_from_proof(proof: Any, *, seq_len: int) -> list[SketchMargin]:
    """Conservative fallback when per-position margins could not be computed.

    Uses tolerance at position 0 (tightest), so ``margin`` is pessimistic —
    it will not hide a real failure but may under-report headroom.
    """
    if proof.sketch_diff_max <= 0:
        return []
    tolerance0 = tolerance_at_position(0, max(1, seq_len))
    margin = tolerance0 - proof.sketch_diff_max
    return [
        SketchMargin(
            position=-1,
            sketch_diff=proof.sketch_diff_max,
            tolerance=tolerance0,
            margin=margin,
            passed=proof.all_passed,
            is_fallback_estimate=True,
        )
    ]


def check_grail_commit(
    commit: dict,
    model: Any,
    randomness: str,
    *,
    tokenizer: Any = None,
) -> GrailCheckResult:
    from reliquary.validator.verifier import verify_commitment_proofs

    proof = verify_commitment_proofs(
        commit, model, randomness, tokenizer=tokenizer,
    )
    seq_len = len(commit.get("tokens") or [])
    margins = _detailed_sketch_margins(commit, model, randomness)
    margin_is_fallback = False
    if not margins:
        margins = _margins_from_proof(proof, seq_len=seq_len)
        margin_is_fallback = bool(margins)
    detailed = [m for m in margins if not m.is_fallback_estimate]
    worst_margin = min((m.margin for m in detailed), default=0)
    if not detailed and margins:
        worst_margin = margins[0].margin
        margin_is_fallback = True
    return GrailCheckResult(
        passed=proof.all_passed,
        sketch_diff_max=proof.sketch_diff_max,
        worst_margin=worst_margin,
        margin_is_fallback=margin_is_fallback,
        positions_checked=proof.checked,
        positions_passed=proof.passed,
        sketch_margins=margins,
        randomness=randomness,
    )


def _detailed_sketch_margins(
    commit: dict,
    model: Any,
    randomness: str,
) -> list[SketchMargin]:
    """Re-run challenged positions to expose sketch_diff vs tolerance per index."""
    import torch

    from reliquary.constants import CHALLENGE_K, LAYER_INDEX
    from reliquary.protocol.crypto import indices_from_root
    from reliquary.protocol.grail_verifier import GRAILVerifier
    from reliquary.shared.forward import forward_single_layer
    from reliquary.shared.hf_compat import resolve_hidden_size

    tokens = commit["tokens"]
    commitments = commit["commitments"]
    seq_len = len(tokens)
    if seq_len == 0:
        return []

    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness)
    expected = min(CHALLENGE_K, seq_len)
    challenge_indices = indices_from_root(tokens, randomness, seq_len, expected)

    device = next(model.parameters()).device
    input_ids = torch.tensor([tokens], device=device)
    with torch.no_grad():
        hidden_states_gpu, _ = forward_single_layer(
            model, input_ids, None, LAYER_INDEX
        )
    hidden_states = hidden_states_gpu[0].detach().to("cpu")

    margins: list[SketchMargin] = []
    for idx in challenge_indices:
        if idx >= seq_len:
            continue
        valid, diag = verifier.verify_commitment(
            hidden_states[idx], commitments[idx], r_vec, seq_len, idx,
        )
        diff = int((diag or {}).get("sketch_diff", 0))
        tol = int((diag or {}).get("sketch_tolerance", tolerance_at_position(idx, seq_len)))
        margins.append(
            SketchMargin(
                position=idx,
                sketch_diff=diff,
                tolerance=tol,
                margin=tol - diff,
                passed=bool(valid),
            )
        )
    return margins


def tolerance_at_position(position: int, sequence_length: int = 8192) -> int:
    return int(
        PROOF_SKETCH_TOLERANCE_BASE
        + PROOF_SKETCH_TOLERANCE_GROWTH * math.sqrt(position)
    )


def make_generation_dict(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> dict:
    from harness.generation import generate_rollout_tokens, rollout_tokens_to_generation_dict

    record = generate_rollout_tokens(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens,
    )
    return rollout_tokens_to_generation_dict(record)


def run_grail_fidelity(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    randomness: str,
    rollouts: int = 1,
) -> list[GrailCheckResult]:
    from harness.grail_build import build_grail_commit

    results: list[GrailCheckResult] = []
    for _ in range(rollouts):
        generation = make_generation_dict(model, tokenizer, prompt)
        commit = build_grail_commit(model, tokenizer, generation, randomness)
        results.append(check_grail_commit(commit, model, randomness, tokenizer=tokenizer))
    return results

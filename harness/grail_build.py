from __future__ import annotations

from typing import Any


def build_grail_commit(
    model: Any,
    tokenizer: Any,
    generation: dict,
    randomness: str,
) -> dict:
    """Build a GRAIL commit dict matching the reference miner layout."""
    import torch

    from reliquary.constants import GRAIL_PROOF_VERSION, LAYER_INDEX
    from reliquary.protocol.grail_verifier import GRAILVerifier
    from reliquary.shared.forward import forward_single_layer
    from reliquary.shared.hf_compat import resolve_hidden_size

    all_tokens: list[int] = generation["tokens"]
    prompt_length: int = generation["prompt_length"]
    device = next(model.parameters()).device

    proof_input = torch.tensor([all_tokens], device=device)
    with torch.no_grad():
        hidden_states, logits = forward_single_layer(
            model, proof_input, None, LAYER_INDEX
        )

    hidden_states = hidden_states[0]
    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness)
    commitments = verifier.create_commitments_batch(hidden_states, r_vec)

    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
    token_logprobs: list[float] = []
    for i in range(prompt_length, len(all_tokens)):
        token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

    model_name: str = getattr(model, "name_or_path", "unknown")
    return {
        "tokens": all_tokens,
        "commitments": commitments,
        "proof_version": GRAIL_PROOF_VERSION,
        "model": {"name": model_name, "layer_index": LAYER_INDEX},
        "signature": "00" * 64,
        "beacon": {"randomness": randomness},
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": len(all_tokens) - prompt_length,
            "success": True,
            "total_reward": 0.0,
            "advantage": 0.0,
            "token_logprobs": token_logprobs,
        },
    }

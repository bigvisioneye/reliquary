from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from reliquary.constants import (
    M_ROLLOUTS,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)
from reliquary.protocol.tokens import encode_prompt
from reliquary.shared.modeling import first_eos_index, resolve_eos_token_ids


@dataclass(frozen=True)
class RolloutTokens:
    """One generated rollout with the exact token ids used for GRAIL binding."""

    tokens: list[int]
    prompt_length: int
    completion_token_ids: list[int]
    truncated: bool


GenerationBackend = Literal["hf", "vllm"]


def _hf_generate_rollouts(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    n_rollouts: int,
    max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    temperature: float = T_PROTO,
) -> list[RolloutTokens]:
    """Generate rollouts on HF and return raw token ids."""
    import torch

    prompt_tokens = encode_prompt(tokenizer, prompt)
    prompt_length = len(prompt_tokens)
    eos_ids = resolve_eos_token_ids(model, tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_ids:
        pad_token_id = min(eos_ids)

    device = getattr(model, "device", "cpu")
    input_tensor = torch.tensor([prompt_tokens] * n_rollouts, device=device)
    attention_mask = torch.ones_like(input_tensor)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": TOP_P_PROTO,
        "top_k": TOP_K_PROTO,
        "pad_token_id": pad_token_id,
        "attention_mask": attention_mask,
    }
    if eos_ids:
        generate_kwargs["eos_token_id"] = sorted(eos_ids)

    with torch.no_grad():
        outputs = model.generate(input_tensor, **generate_kwargs)

    records: list[RolloutTokens] = []
    for i in range(n_rollouts):
        seq = outputs[i].tolist()
        gen = seq[prompt_length:]
        truncated = False
        eos_idx = first_eos_index(gen, eos_ids)
        if eos_idx is not None:
            gen = gen[: eos_idx + 1]
        elif len(gen) >= max_new_tokens:
            truncated = True
        records.append(
            RolloutTokens(
                tokens=prompt_tokens + gen,
                prompt_length=prompt_length,
                completion_token_ids=gen,
                truncated=truncated,
            )
        )
    return records


def generate_rollout_tokens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    temperature: float = T_PROTO,
    gen_backend: GenerationBackend = "hf",
    vllm_engine: Any | None = None,
) -> RolloutTokens:
    """Generate one rollout and return raw token ids (no decode→re-encode)."""
    return generate_m_rollout_records(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        n_rollouts=1,
        gen_backend=gen_backend,
        vllm_engine=vllm_engine,
    )[0]


def generate_m_rollout_records(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    temperature: float = T_PROTO,
    n_rollouts: int = M_ROLLOUTS,
    gen_backend: GenerationBackend = "hf",
    vllm_engine: Any | None = None,
) -> list[RolloutTokens]:
    """Generate ``n_rollouts`` and return raw token records."""
    if gen_backend == "vllm":
        if vllm_engine is None:
            raise ValueError("vLLM backend requested but no vllm_engine provided")
        from harness.vllm_backend import vllm_generate_rollouts

        return vllm_generate_rollouts(
            vllm_engine,
            tokenizer,
            prompt,
            n=n_rollouts,
            temperature=temperature,
            top_p=TOP_P_PROTO,
            top_k=TOP_K_PROTO,
            max_new_tokens=max_new_tokens,
        )
    return _hf_generate_rollouts(
        model,
        tokenizer,
        prompt,
        n_rollouts=n_rollouts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def rollout_tokens_to_generation_dict(record: RolloutTokens) -> dict:
    return {
        "tokens": record.tokens,
        "prompt_length": record.prompt_length,
    }


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    temperature: float = T_PROTO,
) -> str:
    record = generate_rollout_tokens(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    return tokenizer.decode(record.completion_token_ids)


def generate_m_rollouts(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    temperature: float = T_PROTO,
    n_rollouts: int = M_ROLLOUTS,
    gen_backend: GenerationBackend = "hf",
    vllm_engine: Any | None = None,
) -> list[str]:
    """Generate ``n_rollouts`` independent completions in one batched call."""
    records = generate_m_rollout_records(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        n_rollouts=n_rollouts,
        gen_backend=gen_backend,
        vllm_engine=vllm_engine,
    )
    return [tokenizer.decode(r.completion_token_ids) for r in records]


def generate_m_rollout_dicts(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    temperature: float = T_PROTO,
    n_rollouts: int = M_ROLLOUTS,
    gen_backend: GenerationBackend = "hf",
    vllm_engine: Any | None = None,
) -> list[dict]:
    """Generate ``n_rollouts`` token dicts in one batched call (no decode/re-encode)."""
    records = generate_m_rollout_records(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        n_rollouts=n_rollouts,
        gen_backend=gen_backend,
        vllm_engine=vllm_engine,
    )
    return [rollout_tokens_to_generation_dict(r) for r in records]

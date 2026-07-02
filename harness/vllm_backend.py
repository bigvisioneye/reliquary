from __future__ import annotations

from typing import Any

from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

from harness.generation import RolloutTokens

_VLLM_CACHE: dict[tuple, Any] = {}


def load_vllm_generator(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    seed: int | None = None,
) -> Any:
    """Load/cache a vLLM LLM instance lazily."""
    cache_key = (model_path, dtype, gpu_memory_utilization, max_model_len, seed)
    if cache_key in _VLLM_CACHE:
        return _VLLM_CACHE[cache_key]

    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requested but vllm is not installed. "
            "Install with `pip install vllm` or use --gen-backend hf."
        ) from exc

    llm = LLM(
        model=model_path,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    _VLLM_CACHE[cache_key] = llm
    return llm


def vllm_generate_rollouts(
    llm: Any,
    tokenizer: Any,
    prompt_text: str,
    *,
    n: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    seed: int | None = None,
) -> list[RolloutTokens]:
    """Generate rollout token ids via vLLM without decode/re-encode drift."""
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requested but vllm is not installed. "
            "Install with `pip install vllm` or use --gen-backend hf."
        ) from exc

    prompt_ids = llm.get_tokenizer().encode(prompt_text, add_special_tokens=False)
    params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    outputs = llm.generate(prompt_text, params, use_tqdm=False)
    if not outputs:
        return []

    result = outputs[0]
    records: list[RolloutTokens] = []
    for out in result.outputs:
        completion_ids = list(out.token_ids)
        # vLLM naturally stops on EOS; only mark truncated if cap was hit.
        truncated = len(completion_ids) >= max_new_tokens and (
            getattr(out, "finish_reason", "") == "length"
        )
        records.append(
            RolloutTokens(
                tokens=prompt_ids + completion_ids,
                prompt_length=len(prompt_ids),
                completion_token_ids=completion_ids,
                truncated=truncated,
            )
        )
    return records

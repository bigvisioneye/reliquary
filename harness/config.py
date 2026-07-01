from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from typing import Any, Literal

import httpx

from reliquary import constants as C


@dataclass(frozen=True)
class ProtocolConfig:
    sigma_min: float = C.SIGMA_MIN
    bootstrap_sigma_min: float = C.BOOTSTRAP_SIGMA_MIN
    bootstrap_windows: int = C.BOOTSTRAP_WINDOWS
    m_rollouts: int = C.M_ROLLOUTS
    t_proto: float = C.T_PROTO
    top_p_proto: float = C.TOP_P_PROTO
    top_k_proto: int = C.TOP_K_PROTO
    max_new_tokens_protocol_cap: int = C.MAX_NEW_TOKENS_PROTOCOL_CAP
    prompt_range_size: int = C.PROMPT_RANGE_SIZE
    b_batch: int = C.B_BATCH
    sparse_valid_idle_seal_seconds: float = C.SPARSE_VALID_IDLE_SEAL_SECONDS
    sparse_valid_idle_min_distinct_prompts: int = C.SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS
    sparse_valid_max_window_seconds: float = C.SPARSE_VALID_MAX_WINDOW_SECONDS
    max_submissions_per_prompt: int = C.MAX_SUBMISSIONS_PER_PROMPT
    max_submissions_per_hotkey_per_window: int = C.MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
    ema_alpha: float = C.EMA_ALPHA
    checkpoint_publish_interval_windows: int = C.CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
    batch_prompt_cooldown_windows: int = C.BATCH_PROMPT_COOLDOWN_WINDOWS
    hash_dedup_retention_windows: int = C.HASH_DEDUP_RETENTION_WINDOWS
    attn_implementation: str = C.ATTN_IMPLEMENTATION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    attn: Literal["eager", "sdpa", "flash_attention_2"]
    dtype: str


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_repo_id: str
    checkpoint_revision: str


def resolve_runtime_config(device: str, attn: str, dtype: str) -> RuntimeConfig:
    valid_attn = {"eager", "sdpa", "flash_attention_2"}
    if attn not in valid_attn:
        raise ValueError(f"Unsupported --attn value '{attn}'. Expected one of: {sorted(valid_attn)}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Unsupported --device value. Expected 'cpu' or 'cuda'.")
    if device == "cuda":
        if importlib.util.find_spec("torch") is None:
            raise ValueError("Requested --device cuda but torch is unavailable.")
        import torch

        if not torch.cuda.is_available():
            raise ValueError("Requested --device cuda but CUDA is unavailable.")
    return RuntimeConfig(device=device, attn=attn, dtype=dtype)


def require_flash_attention_for_grail(runtime: RuntimeConfig) -> None:
    if runtime.attn != "flash_attention_2":
        raise ValueError("Step 5 requires --attn flash_attention_2.")
    if runtime.device != "cuda":
        raise ValueError("Step 5 requires --device cuda.")


def resolve_checkpoint_ref(
    checkpoint_repo_id: str | None,
    checkpoint_revision: str | None,
    validator_state_url: str | None = None,
    timeout_seconds: float = 15.0,
) -> CheckpointRef:
    if checkpoint_repo_id and checkpoint_revision:
        return CheckpointRef(
            checkpoint_repo_id=checkpoint_repo_id,
            checkpoint_revision=checkpoint_revision,
        )
    if not validator_state_url:
        raise ValueError(
            "Provide --checkpoint-repo-id and --checkpoint-revision, or set --state-url."
        )

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(validator_state_url)
        response.raise_for_status()
        payload = response.json()

    repo_id = payload.get("checkpoint_repo_id")
    revision = payload.get("checkpoint_revision")
    if not repo_id or not revision:
        raise ValueError(
            "Validator state did not provide checkpoint_repo_id/checkpoint_revision."
        )
    return CheckpointRef(checkpoint_repo_id=str(repo_id), checkpoint_revision=str(revision))


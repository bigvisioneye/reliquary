from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from reliquary.shared.modeling import (
    MODEL_SNAPSHOT_ALLOW_PATTERNS,
    load_text_generation_model,
    load_tokenizer,
)


@dataclass(frozen=True)
class LoadedCheckpoint:
    local_path: str
    tokenizer: Any
    model: Any


def snapshot_checkpoint(
    checkpoint_repo_id: str,
    checkpoint_revision: str,
    cache_dir: str | None = None,
) -> str:
    # First try local cache only to keep repeated setup free.
    try:
        return snapshot_download(
            repo_id=checkpoint_repo_id,
            revision=checkpoint_revision,
            allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception:
        return snapshot_download(
            repo_id=checkpoint_repo_id,
            revision=checkpoint_revision,
            allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
            cache_dir=cache_dir,
            local_files_only=False,
        )


def resolve_dtype(dtype: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported --dtype '{dtype}'.")
    return mapping[key]


def load_checkpoint(
    checkpoint_repo_id: str,
    checkpoint_revision: str,
    *,
    cache_dir: str | None = None,
    device: str = "cpu",
    attn: str = "eager",
    dtype: str = "bfloat16",
) -> LoadedCheckpoint:
    local_path = snapshot_checkpoint(
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        cache_dir=cache_dir,
    )
    tokenizer = load_tokenizer(local_path)
    torch_dtype = resolve_dtype(dtype)
    model = load_text_generation_model(
        local_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn,
    ).to(device).eval()
    return LoadedCheckpoint(local_path=local_path, tokenizer=tokenizer, model=model)


def ensure_cache_dir(path: str | None) -> str | None:
    if path is None:
        return None
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


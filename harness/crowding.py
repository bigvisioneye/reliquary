from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CrowdingEstimate:
    prompt_idx: int
    k_p: int
    expected_slot_share: float


class CrowdingEstimator(Protocol):
    def estimate(self, prompt_idx: int) -> CrowdingEstimate:
        """Return expected submitter count K_p for a prompt index."""
        ...


class StubCrowdingEstimator:
    """Placeholder seam for live-miner crowding signals (always uncrowded)."""

    def estimate(self, prompt_idx: int) -> CrowdingEstimate:
        return CrowdingEstimate(
            prompt_idx=prompt_idx,
            k_p=1,
            expected_slot_share=1.0,
        )

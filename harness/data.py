from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment


@dataclass
class OpenMathData:
    env: OpenMathInstructEnvironment

    @classmethod
    def load(cls) -> "OpenMathData":
        return cls(env=OpenMathInstructEnvironment())

    def __len__(self) -> int:
        return len(self.env)

    def get_problem(self, idx: int) -> dict[str, Any]:
        return self.env.get_problem(idx)


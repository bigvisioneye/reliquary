"""Harness tests — isolated from tests/unit/conftest validator fixtures."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _stub_heavy_ml_imports() -> None:
    if "transformers" not in sys.modules:
        tf = MagicMock()
        tf.PretrainedConfig = type("PretrainedConfig", (), {})
        tf.PreTrainedModel = type("PreTrainedModel", (), {})
        sys.modules["transformers"] = tf
    if "torch" not in sys.modules:
        sys.modules["torch"] = MagicMock()


_stub_heavy_ml_imports()

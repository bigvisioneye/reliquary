from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from harness.latency import measure_latency


def test_measure_latency_proof_loop_does_not_regenerate() -> None:
    generation = {"tokens": [1, 2, 3], "prompt_length": 1}

    def _boom(*_args, **_kwargs):
        raise AssertionError("make_generation_dict must not run inside proof timing")

    with (
        patch("harness.latency.generate_m_rollouts", return_value=["a"] * 8),
        patch("harness.latency.generate_rollout_tokens") as mock_rollout,
        patch("harness.latency.rollout_tokens_to_generation_dict", return_value=generation),
        patch("harness.latency.build_grail_commit", return_value=MagicMock()),
        patch("harness.latency.check_grail_commit", return_value=MagicMock(passed=True)),
        patch("harness.grail_check.make_generation_dict", side_effect=_boom),
    ):
        mock_rollout.return_value = MagicMock()
        report = measure_latency(
            model=MagicMock(),
            tokenizer=MagicMock(),
            prompt="test",
            randomness="aa" * 16,
            checkpoint_revision="rev",
            generation_runs=1,
            proof_runs=2,
        )

    assert mock_rollout.call_count == 1
    assert report.proof_per_rollout.samples == 2
    assert report.generation_is_batched is True
    assert report.proofs_batched is False


def test_measure_latency_proofs_batched_uses_batch_dicts() -> None:
    batch = [{"tokens": [1, 2], "prompt_length": 1}] * 8
    with (
        patch("harness.latency.generate_m_rollouts", return_value=["a"] * 8),
        patch("harness.latency.generate_m_rollout_dicts", return_value=batch) as mock_batch,
        patch("harness.latency.generate_rollout_tokens") as mock_single,
        patch("harness.latency.build_grail_commit", return_value=MagicMock()),
        patch("harness.latency.check_grail_commit", return_value=MagicMock(passed=True)),
    ):
        report = measure_latency(
            model=MagicMock(),
            tokenizer=MagicMock(),
            prompt="test",
            randomness="bb" * 16,
            checkpoint_revision="rev",
            generation_runs=1,
            proof_runs=1,
            proofs_batched=True,
        )

    mock_batch.assert_called_once()
    mock_single.assert_not_called()
    assert report.proofs_batched is True
    assert report.proof_batch is not None
    assert report.proof_batch.samples == 1

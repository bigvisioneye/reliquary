from __future__ import annotations

from unittest.mock import MagicMock

from harness.generation import RolloutTokens, generate_m_rollouts
from harness.label import true_label


def test_generate_m_rollouts_routes_to_vllm_backend(monkeypatch) -> None:
    called = {"vllm": 0}

    def _fake_vllm(*_args, **_kwargs):
        called["vllm"] += 1
        return [
            RolloutTokens(
                tokens=[1, 2, 3],
                prompt_length=1,
                completion_token_ids=[2, 3],
                truncated=False,
            )
        ]

    monkeypatch.setattr("harness.vllm_backend.vllm_generate_rollouts", _fake_vllm)
    out = generate_m_rollouts(
        model=MagicMock(),
        tokenizer=MagicMock(decode=lambda ids: "x"),
        prompt="p",
        n_rollouts=1,
        gen_backend="vllm",
        vllm_engine=object(),
    )
    assert out == ["x"]
    assert called["vllm"] == 1


def test_true_label_uses_max_label_tokens(monkeypatch) -> None:
    seen = {"max_new_tokens": None}

    def _fake_records(*_args, **kwargs):
        seen["max_new_tokens"] = kwargs["max_new_tokens"]
        return [
            RolloutTokens(
                tokens=[1, 2, 3],
                prompt_length=1,
                completion_token_ids=[2, 3],
                truncated=False,
            )
            for _ in range(8)
        ]

    monkeypatch.setattr("harness.label.generate_m_rollout_records", _fake_records)
    fake_env = MagicMock()
    fake_env.get_problem.return_value = {"prompt": "p", "ground_truth": "42"}
    tok = MagicMock()
    tok.decode.return_value = "\\boxed{42}"

    true_label(
        model=MagicMock(),
        tokenizer=tok,
        env=fake_env,
        prompt_idx=1,
        checkpoint_repo_id="repo",
        checkpoint_revision="rev",
        max_label_tokens=777,
    )
    assert seen["max_new_tokens"] == 777

import pytest

pytest.importorskip("torch")

from harness.scoring import compute_sigma, in_zone


def test_sigma_k4_in_zone() -> None:
    rewards = [1, 1, 1, 1, 0, 0, 0, 0]
    sigma = compute_sigma(rewards)
    assert sigma == 0.5
    assert in_zone(sigma, bootstrap=False) is True


def test_sigma_k1_out_of_zone() -> None:
    rewards = [1, 0, 0, 0, 0, 0, 0, 0]
    sigma = compute_sigma(rewards)
    assert round(sigma, 6) == round((7 / 64) ** 0.5, 6)
    assert in_zone(sigma, bootstrap=False) is False


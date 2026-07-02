from harness.sampling import effective_sample_mode, resolve_calibration_indices


class _FakeEnv:
    def __len__(self) -> int:
        return 1000


def test_effective_sample_mode_maps_enforce_slice() -> None:
    assert effective_sample_mode("uniform", True) == "slice"
    assert effective_sample_mode("slice", False) == "slice"


def test_resolve_uniform_samples_full_universe() -> None:
    env = _FakeEnv()
    indices = resolve_calibration_indices(
        env, count=5, seed=0, randomness=None, sample_mode="uniform",
    )
    assert len(indices) == 5
    assert all(0 <= idx < 1000 for idx in indices)


def test_resolve_slice_requires_randomness() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires --randomness"):
        resolve_calibration_indices(
            _FakeEnv(), count=3, seed=0, randomness=None, sample_mode="slice",
        )

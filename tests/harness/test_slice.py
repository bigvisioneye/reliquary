from harness.slice import (
    filter_prompt_indices,
    in_slice,
    slice_for_window,
)


def test_slice_for_window_deterministic() -> None:
    a = slice_for_window("abc123", "openmathinstruct", 100_000, size=5000)
    b = slice_for_window("abc123", "openmathinstruct", 100_000, size=5000)
    assert a == b
    lo, hi = a
    assert 0 <= lo < hi <= 100_000
    assert hi - lo == 5000


def test_filter_prompt_indices_enforce() -> None:
    bounds = (100, 200)
    indices = [50, 150, 250]
    assert filter_prompt_indices(indices, bounds, enforce=False) == indices
    assert filter_prompt_indices(indices, bounds, enforce=True) == [150]


def test_in_slice() -> None:
    assert in_slice(150, (100, 200)) is True
    assert in_slice(200, (100, 200)) is False

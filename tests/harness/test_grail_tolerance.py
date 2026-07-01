from harness.grail_check import tolerance_at_position

from reliquary.constants import PROOF_SKETCH_TOLERANCE_BASE, PROOF_SKETCH_TOLERANCE_GROWTH


def test_tolerance_at_position_matches_formula() -> None:
    pos = 100
    expected = int(
        PROOF_SKETCH_TOLERANCE_BASE + PROOF_SKETCH_TOLERANCE_GROWTH * (pos ** 0.5)
    )
    assert tolerance_at_position(pos) == expected

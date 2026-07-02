from harness.probe_logic import (
    ProbeConfig,
    binary_in_zone_band,
    predicts_in_zone,
    probe_step,
)
from harness.scoring import classify_probe_outcome


def test_probe_step_in_zone_band_on_mixed_outcomes() -> None:
    cfg = ProbeConfig(max_samples=6, extreme_same_threshold=3)
    decision, p_hat, confidence = probe_step(
        ["success", "failure"], config=cfg,
    )
    assert decision == "in_zone_band"
    assert p_hat == 0.5
    assert confidence == "high"


def test_probe_step_too_hard_after_streak() -> None:
    cfg = ProbeConfig(max_samples=6, extreme_same_threshold=3)
    decision, p_hat, confidence = probe_step(
        ["failure", "failure", "failure"], config=cfg,
    )
    assert decision == "too_hard"
    assert p_hat == 0.0
    assert confidence == "high"


def test_probe_step_too_easy_after_streak() -> None:
    cfg = ProbeConfig(max_samples=6, extreme_same_threshold=3)
    decision, p_hat, confidence = probe_step(
        ["success", "success", "success"], config=cfg,
    )
    assert decision == "too_easy"
    assert p_hat == 1.0
    assert confidence == "high"


def test_probe_step_budget_exhausted() -> None:
    cfg = ProbeConfig(max_samples=4, extreme_same_threshold=10)
    decision, _, confidence = probe_step(
        ["success", "success", "success", "success"], config=cfg,
    )
    assert decision == "budget_exhausted"
    assert confidence == "medium"


def test_probe_step_all_unknown_budget_exhausted_not_extreme() -> None:
    cfg = ProbeConfig(max_samples=4, extreme_same_threshold=3)
    decision, p_hat, confidence = probe_step(
        ["unknown", "unknown", "unknown", "unknown"], config=cfg,
    )
    assert decision == "budget_exhausted"
    assert decision not in ("too_easy", "too_hard")
    assert p_hat == 0.0
    assert confidence == "medium"


def test_probe_unknowns_do_not_increment_extreme_streak() -> None:
    cfg = ProbeConfig(max_samples=6, extreme_same_threshold=3)
    decision, _, _ = probe_step(
        ["unknown", "failure", "unknown", "failure", "unknown"], config=cfg,
    )
    assert decision == "continue"


def test_probe_unknowns_do_not_count_as_failures() -> None:
    cfg = ProbeConfig(max_samples=6, extreme_same_threshold=3)
    outcomes = ["unknown", "unknown", "unknown", "success", "failure"]
    decision, p_hat, confidence = probe_step(outcomes, config=cfg)
    assert decision == "in_zone_band"
    assert p_hat == 0.5
    assert confidence == "high"


def test_classify_truncated_as_unknown() -> None:
    problem = {"ground_truth": "42"}
    assert classify_probe_outcome(
        problem, "still thinking...", truncated=True,
    ) == "unknown"


def test_classify_natural_eos_without_boxed_as_failure() -> None:
    problem = {"ground_truth": "42"}
    assert classify_probe_outcome(
        problem, "no answer here", truncated=False,
    ) == "failure"


def test_binary_in_zone_band_steady_state() -> None:
    lo, hi = binary_in_zone_band(bootstrap=False)
    assert lo == 0.25
    assert hi == 0.75


def test_predicts_in_zone_middle_band() -> None:
    assert predicts_in_zone(0.5, bootstrap=False) is True
    assert predicts_in_zone(0.125, bootstrap=False) is False

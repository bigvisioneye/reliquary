from harness.calibrate_metrics import (
    CalibrationRow,
    build_pr_curve,
    classification_metrics,
    compute_saved_per_in_zone,
    recommend_jitter_band,
    summarize_jitter,
)


def test_classification_metrics_perfect_prediction() -> None:
    predicted = [True, False, True, False]
    actual = [True, False, True, False]
    m = classification_metrics(predicted, actual)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.true_positives == 2
    assert m.true_negatives == 2


def test_build_pr_curve_has_points() -> None:
    p_hats = [0.5, 0.125, 0.75, 0.375]
    actual = [True, False, True, True]
    curve = build_pr_curve(p_hats, actual, half_widths=[0.1, 0.2, 0.3])
    assert len(curve) == 3
    assert all(0.0 <= p.f1 <= 1.0 for p in curve)


def test_compute_saved_per_in_zone() -> None:
    rows = [
        CalibrationRow(
            idx=1, k=4, true_sigma=0.5, in_zone=True,
            p_hat=0.5, probe_samples_used=2, decision="in_zone_band",
            predicted_in_zone=True,
        ),
        CalibrationRow(
            idx=2, k=1, true_sigma=0.33, in_zone=True,
            p_hat=0.0, probe_samples_used=3, decision="too_hard",
            predicted_in_zone=False,
        ),
    ]
    assert compute_saved_per_in_zone(rows) == 1.5


def test_recommend_jitter_band_favors_middle() -> None:
    lo, hi = recommend_jitter_band([3, 4, 5], margin=1)
    assert lo == 0.25
    assert hi == 0.75


def test_summarize_jitter() -> None:
    summary = summarize_jitter(42, [3, 4, 5], [True, True, True], repeats=3)
    assert summary.prompt_idx == 42
    assert summary.k_mean == 4.0
    assert summary.in_zone_rate == 1.0

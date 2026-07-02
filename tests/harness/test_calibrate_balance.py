import pytest

from harness.calibrate_metrics import build_calibration_balance, CalibrationRow
from harness.probe_logic import ProbeConfig


def test_build_calibration_balance_counts_k_and_unknowns() -> None:
    rows = [
        CalibrationRow(
            idx=1, k=8, true_sigma=1.0, in_zone=False,
            p_hat=1.0, probe_samples_used=3, decision="too_easy",
            predicted_in_zone=False,
        ),
        CalibrationRow(
            idx=2, k=4, true_sigma=0.5, in_zone=True,
            p_hat=0.5, probe_samples_used=4, decision="in_zone_band",
            predicted_in_zone=True,
        ),
    ]
    balance = build_calibration_balance(
        rows, probe_samples_total=7, probe_unknowns_total=3,
    )
    assert balance.n_in_zone_true == 1
    assert balance.n_out_of_zone_true == 1
    assert balance.k_histogram["8"] == 1
    assert balance.k_histogram["4"] == 1
    assert balance.unknown_rate == pytest.approx(3 / 7)


def test_probe_config_default_max_probe_tokens() -> None:
    assert ProbeConfig().max_probe_tokens == 1536

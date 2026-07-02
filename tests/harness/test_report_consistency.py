from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.calibrate_metrics import (
    CalibrationBalance,
    CalibrationReport,
    CalibrationRow,
    ClassificationMetrics,
    calibration_report_dict,
    validate_calibration_report,
    write_calibration_report,
)
from harness.report import FullHarnessReport, write_report_artifacts


def _sample_report(*, n_prompts: int = 3) -> CalibrationReport:
    rows = [
        CalibrationRow(
            idx=i,
            k=4,
            true_sigma=0.5,
            in_zone=True,
            p_hat=0.5,
            probe_samples_used=2,
            decision="in_zone_band",
            predicted_in_zone=True,
        )
        for i in range(n_prompts)
    ]
    metrics = ClassificationMetrics(
        precision=1.0, recall=1.0, f1=1.0,
        true_positives=n_prompts, false_positives=0,
        true_negatives=0, false_negatives=0,
    )
    balance = CalibrationBalance(
        n_in_zone_true=n_prompts,
        n_out_of_zone_true=0,
        k_histogram={"4": n_prompts},
        unknown_rate=0.1,
        probe_samples_total=n_prompts * 2,
        probe_unknowns_total=n_prompts,
        label_rollouts_total=n_prompts * 8,
        label_truncated_unscorable_total=0,
        label_truncation_rate=0.0,
    )
    return CalibrationReport(
        checkpoint_repo_id="repo",
        checkpoint_revision="rev",
        n_prompts=n_prompts,
        requested_prompts=n_prompts,
        sample_mode="slice",
        metrics=metrics,
        pr_curve=[],
        compute_saved_per_in_zone=2.0,
        balance=balance,
        jitter=[],
        rows=rows,
    )


def test_validate_calibration_report_rejects_empty_when_requested() -> None:
    report = _sample_report(n_prompts=0)
    report.requested_prompts = 10
    with pytest.raises(ValueError, match="0 prompts"):
        validate_calibration_report(report)


def test_standalone_and_full_reports_share_calibration_fields(tmp_path: Path) -> None:
    calibration = _sample_report(n_prompts=4)
    full = FullHarnessReport(
        checkpoint_repo_id="repo",
        checkpoint_revision="rev",
        randomness="cc" * 16,
        enforce_slice=True,
        sample_mode="slice",
        prompt_indices=[r.idx for r in calibration.rows],
        slice_bounds=(0, 100),
        calibration=calibration,
        grail_checks=[],
        grail_all_passed=True,
        latency=None,
        jitter_band_recommendation=None,
        suggested_workers=2,
    )
    report_json = tmp_path / "full_report.json"
    cal_json = tmp_path / "calibration_report.json"
    csv_path = tmp_path / "calibration.csv"

    write_report_artifacts(
        full,
        report_json=report_json,
        calibration_csv=csv_path,
        calibration_report_json=cal_json,
    )

    full_payload = json.loads(report_json.read_text())
    cal_payload = json.loads(cal_json.read_text())
    assert full_payload["calibration"]["n_prompts"] == 4
    assert cal_payload["n_prompts"] == 4
    assert full_payload["calibration"]["metrics"] == cal_payload["metrics"]
    assert full_payload["calibration"]["balance"] == cal_payload["balance"]
    assert calibration_report_dict(calibration) == cal_payload


def test_write_calibration_report_matches_dict(tmp_path: Path) -> None:
    report = _sample_report(n_prompts=2)
    out = tmp_path / "calibration_report.json"
    write_calibration_report(out, report)
    assert json.loads(out.read_text()) == calibration_report_dict(report)

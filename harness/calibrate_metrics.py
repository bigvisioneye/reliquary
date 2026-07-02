from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from reliquary.constants import M_ROLLOUTS

from harness.probe_logic import predicts_in_zone


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass
class PRCurvePoint:
    band_half_width: float
    p_hat_lo: float
    p_hat_hi: float
    precision: float
    recall: float
    f1: float


@dataclass
class CalibrationRow:
    idx: int
    k: int
    true_sigma: float
    in_zone: bool
    p_hat: float
    probe_samples_used: int
    decision: str
    predicted_in_zone: bool


@dataclass
class JitterSummary:
    prompt_idx: int
    repeats: int
    k_values: list[int]
    k_min: int
    k_max: int
    k_mean: float
    in_zone_rate: float
    recommended_band: tuple[float, float]


@dataclass
class CalibrationBalance:
    n_in_zone_true: int
    n_out_of_zone_true: int
    k_histogram: dict[str, int]
    unknown_rate: float
    probe_samples_total: int
    probe_unknowns_total: int


@dataclass
class CalibrationReport:
    checkpoint_repo_id: str
    checkpoint_revision: str
    n_prompts: int
    requested_prompts: int
    sample_mode: str
    metrics: ClassificationMetrics
    pr_curve: list[PRCurvePoint]
    compute_saved_per_in_zone: float
    balance: CalibrationBalance
    jitter: list[JitterSummary]
    rows: list[CalibrationRow]


def classification_metrics(
    predicted: Iterable[bool],
    actual: Iterable[bool],
) -> ClassificationMetrics:
    tp = fp = tn = fn = 0
    for pred, truth in zip(predicted, actual, strict=True):
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def band_from_half_width(half_width: float) -> tuple[float, float]:
    center = 0.5
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def build_pr_curve(
    p_hats: list[float],
    actual_in_zone: list[bool],
    *,
    half_widths: list[float] | None = None,
) -> list[PRCurvePoint]:
    if half_widths is None:
        half_widths = [round(x * 0.05, 2) for x in range(1, 11)]
    points: list[PRCurvePoint] = []
    for hw in half_widths:
        band = band_from_half_width(hw)
        predicted = [predicts_in_zone(p, band=band) for p in p_hats]
        m = classification_metrics(predicted, actual_in_zone)
        lo, hi = band
        points.append(
            PRCurvePoint(
                band_half_width=hw,
                p_hat_lo=lo,
                p_hat_hi=hi,
                precision=m.precision,
                recall=m.recall,
                f1=m.f1,
            )
        )
    return points


def compute_saved_per_in_zone(
    rows: list[CalibrationRow],
    *,
    full_rollouts: int = M_ROLLOUTS,
) -> float:
    in_zone_rows = [r for r in rows if r.in_zone]
    if not in_zone_rows:
        return 0.0

    blind_cost = len(in_zone_rows) * full_rollouts
    probe_cost = 0.0
    for row in in_zone_rows:
        probe_cost += row.probe_samples_used
        if row.predicted_in_zone:
            probe_cost += full_rollouts
    return (blind_cost - probe_cost) / len(in_zone_rows)


def recommend_jitter_band(k_values: list[int], *, margin: int = 1) -> tuple[float, float]:
    if not k_values:
        return (0.375, 0.625)
    k_min = min(k_values)
    k_max = max(k_values)
    lo_k = max(0, k_min - margin)
    hi_k = min(M_ROLLOUTS, k_max + margin)
    return (lo_k / M_ROLLOUTS, hi_k / M_ROLLOUTS)


def summarize_jitter(
    prompt_idx: int,
    k_values: list[int],
    in_zone_flags: list[bool],
    *,
    repeats: int,
) -> JitterSummary:
    return JitterSummary(
        prompt_idx=prompt_idx,
        repeats=repeats,
        k_values=k_values,
        k_min=min(k_values) if k_values else 0,
        k_max=max(k_values) if k_values else 0,
        k_mean=sum(k_values) / len(k_values) if k_values else 0.0,
        in_zone_rate=sum(in_zone_flags) / len(in_zone_flags) if in_zone_flags else 0.0,
        recommended_band=recommend_jitter_band(k_values),
    )


def build_calibration_balance(
    rows: list[CalibrationRow],
    *,
    probe_samples_total: int,
    probe_unknowns_total: int,
) -> CalibrationBalance:
    k_hist: dict[str, int] = {str(k): 0 for k in range(M_ROLLOUTS + 1)}
    for row in rows:
        k_hist[str(row.k)] = k_hist.get(str(row.k), 0) + 1
    n_in_zone = sum(1 for r in rows if r.in_zone)
    unknown_rate = (
        probe_unknowns_total / probe_samples_total if probe_samples_total else 0.0
    )
    return CalibrationBalance(
        n_in_zone_true=n_in_zone,
        n_out_of_zone_true=len(rows) - n_in_zone,
        k_histogram=k_hist,
        unknown_rate=unknown_rate,
        probe_samples_total=probe_samples_total,
        probe_unknowns_total=probe_unknowns_total,
    )


def validate_calibration_report(report: CalibrationReport) -> None:
    if report.requested_prompts > 0 and report.n_prompts == 0:
        raise ValueError(
            f"calibration produced 0 prompts (requested {report.requested_prompts}); "
            "check --sample-mode, --randomness, explicit --prompts, and slice bounds"
        )


def calibration_report_dict(report: CalibrationReport) -> dict:
    return {
        "checkpoint_repo_id": report.checkpoint_repo_id,
        "checkpoint_revision": report.checkpoint_revision,
        "n_prompts": report.n_prompts,
        "requested_prompts": report.requested_prompts,
        "sample_mode": report.sample_mode,
        "metrics": asdict(report.metrics),
        "balance": asdict(report.balance),
        "pr_curve": [asdict(p) for p in report.pr_curve],
        "compute_saved_per_in_zone": report.compute_saved_per_in_zone,
        "jitter": [asdict(j) for j in report.jitter],
    }


def write_calibration_csv(path: str | Path, rows: list[CalibrationRow]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "idx", "k", "true_sigma", "in_zone", "p_hat",
        "probe_samples_used", "decision", "predicted_in_zone",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_calibration_report(path: str | Path, report: CalibrationReport) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(calibration_report_dict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )

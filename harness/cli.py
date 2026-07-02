from __future__ import annotations

import json
import random
from dataclasses import asdict

import typer

from harness.config import (
    ProtocolConfig,
    require_flash_attention_for_grail,
    resolve_checkpoint_ref,
    resolve_runtime_config,
)

app = typer.Typer(help="Reliquary offline calibration harness.")


def _parse_prompt_list(prompts: str, count: int, seed: int) -> list[int]:
    if prompts.strip():
        return [int(p.strip()) for p in prompts.split(",") if p.strip()]
    rng = random.Random(seed)
    return [rng.randrange(0, 10_000) for _ in range(count)]


def _resolve_indices(
    env,
    *,
    prompts: str,
    count: int,
    seed: int,
    randomness: str,
    sample_mode: str,
    enforce_slice: bool,
):
    from harness.sampling import resolve_calibration_indices

    return resolve_calibration_indices(
        env,
        count=count,
        seed=seed,
        randomness=randomness or None,
        sample_mode=sample_mode,  # type: ignore[arg-type]
        enforce_slice=enforce_slice,
        explicit_prompts=prompts,
    )


def _load_model_and_data(
    checkpoint_repo_id: str,
    checkpoint_revision: str,
    device: str,
    attn: str,
    dtype: str,
    cache_dir: str,
):
    from harness.data import OpenMathData
    from harness.model import ensure_cache_dir, load_checkpoint

    resolve_runtime_config(device=device, attn=attn, dtype=dtype)
    data = OpenMathData.load()
    loaded = load_checkpoint(
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        cache_dir=ensure_cache_dir(cache_dir or None),
        device=device,
        attn=attn,
        dtype=dtype,
    )
    return data, loaded


def _maybe_load_vllm_engine(
    *,
    loaded,
    gen_backend: str,
    gpu_mem_util: float,
    max_model_len: int | None,
):
    if gen_backend != "vllm":
        return None
    from harness.vllm_backend import load_vllm_generator

    model_path = getattr(loaded.model, "name_or_path", None)
    if not model_path:
        raise ValueError("could not resolve model path for vLLM backend")
    return load_vllm_generator(
        model_path,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
    )


@app.command("resolve")
def resolve_cmd(
    checkpoint_repo_id: str = typer.Option("", help="HF checkpoint repo id."),
    checkpoint_revision: str = typer.Option("", help="HF checkpoint revision."),
    state_url: str = typer.Option("", help="Validator /state URL to resolve checkpoint from."),
    device: str = typer.Option("cpu", help="cpu or cuda."),
    attn: str = typer.Option("eager", help="eager/sdpa/flash_attention_2."),
    dtype: str = typer.Option("bfloat16", help="bf16/fp16/fp32 style dtype."),
    require_grail_mode: bool = typer.Option(False, help="Enforce Step-5 runtime requirements."),
) -> None:
    runtime = resolve_runtime_config(device=device, attn=attn, dtype=dtype)
    if require_grail_mode:
        require_flash_attention_for_grail(runtime)
    ckpt = resolve_checkpoint_ref(
        checkpoint_repo_id=checkpoint_repo_id or None,
        checkpoint_revision=checkpoint_revision or None,
        validator_state_url=state_url or None,
    )
    payload = {
        "checkpoint": asdict(ckpt),
        "runtime": asdict(runtime),
        "protocol": ProtocolConfig().to_dict(),
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("label")
def label_cmd(
    prompt_idx: int = typer.Option(..., help="Prompt index to label."),
    checkpoint_repo_id: str = typer.Option(..., help="HF checkpoint repo id."),
    checkpoint_revision: str = typer.Option(..., help="HF checkpoint revision."),
    device: str = typer.Option("cpu", help="cpu or cuda."),
    attn: str = typer.Option("eager", help="eager/sdpa/flash_attention_2."),
    dtype: str = typer.Option("bfloat16", help="bf16/fp16/fp32 style dtype."),
    cache_dir: str = typer.Option("", help="Optional HF cache dir."),
    bootstrap: bool = typer.Option(False, help="Use bootstrap in-zone threshold."),
    max_label_tokens: int = typer.Option(2048, help="Cap tokens for offline true_label generation."),
    gen_backend: str = typer.Option("hf", help="Generation backend: hf or vllm."),
    gpu_mem_util: float = typer.Option(0.85, help="vLLM GPU memory utilization."),
    max_model_len: int = typer.Option(0, help="Optional vLLM max model length."),
) -> None:
    from harness.label import true_label

    data, loaded = _load_model_and_data(
        checkpoint_repo_id, checkpoint_revision, device, attn, dtype, cache_dir,
    )
    vllm_engine = _maybe_load_vllm_engine(
        loaded=loaded,
        gen_backend=gen_backend,
        gpu_mem_util=gpu_mem_util,
        max_model_len=max_model_len or None,
    )
    result = true_label(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        env=data.env,
        prompt_idx=prompt_idx,
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        bootstrap=bootstrap,
        max_label_tokens=max_label_tokens,
        gen_backend=gen_backend,
        vllm_engine=vllm_engine,
    )
    typer.echo(json.dumps(asdict(result), indent=2, sort_keys=True))


@app.command("probe")
def probe_cmd(
    prompt_idx: int = typer.Option(..., help="Prompt index to probe."),
    checkpoint_repo_id: str = typer.Option(..., help="HF checkpoint repo id."),
    checkpoint_revision: str = typer.Option(..., help="HF checkpoint revision."),
    device: str = typer.Option("cpu", help="cpu or cuda."),
    attn: str = typer.Option("eager", help="eager/sdpa/flash_attention_2."),
    dtype: str = typer.Option("bfloat16", help="bf16/fp16/fp32 style dtype."),
    cache_dir: str = typer.Option("", help="Optional HF cache dir."),
    max_samples: int = typer.Option(6, help="Max sequential probe samples."),
    max_probe_tokens: int = typer.Option(1536, help="Cap tokens per probe sample (raise if unknown_rate is high)."),
    temperature: float = typer.Option(-1.0, help="Override probe temperature; default T_PROTO."),
    bootstrap: bool = typer.Option(False, help="Use bootstrap in-zone band."),
    gen_backend: str = typer.Option("hf", help="Generation backend: hf or vllm."),
    gpu_mem_util: float = typer.Option(0.85, help="vLLM GPU memory utilization."),
    max_model_len: int = typer.Option(0, help="Optional vLLM max model length."),
) -> None:
    from harness.probe import run_probe
    from harness.probe_logic import ProbeConfig
    from reliquary.constants import T_PROTO

    data, loaded = _load_model_and_data(
        checkpoint_repo_id, checkpoint_revision, device, attn, dtype, cache_dir,
    )
    vllm_engine = _maybe_load_vllm_engine(
        loaded=loaded,
        gen_backend=gen_backend,
        gpu_mem_util=gpu_mem_util,
        max_model_len=max_model_len or None,
    )
    cfg = ProbeConfig(
        max_samples=max_samples,
        max_probe_tokens=max_probe_tokens,
        temperature=T_PROTO if temperature < 0 else temperature,
    )
    problem = data.env.get_problem(prompt_idx)
    result = run_probe(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        problem=problem,
        prompt_idx=prompt_idx,
        config=cfg,
        bootstrap=bootstrap,
        gen_backend=gen_backend,
        vllm_engine=vllm_engine,
    )
    typer.echo(json.dumps(asdict(result), indent=2, sort_keys=True))


@app.command("calibrate")
def calibrate_cmd(
    checkpoint_repo_id: str = typer.Option(..., help="HF checkpoint repo id."),
    checkpoint_revision: str = typer.Option(..., help="HF checkpoint revision."),
    prompts: str = typer.Option("", help="Comma-separated prompt indices."),
    count: int = typer.Option(20, help="Random prompt count when --prompts omitted (use 300+ for stable metrics)."),
    seed: int = typer.Option(0, help="RNG seed for random prompt selection."),
    device: str = typer.Option("cpu", help="cpu or cuda."),
    attn: str = typer.Option("eager", help="eager/sdpa/flash_attention_2."),
    dtype: str = typer.Option("bfloat16", help="bf16/fp16/fp32 style dtype."),
    cache_dir: str = typer.Option("", help="Optional HF cache dir."),
    max_samples: int = typer.Option(6, help="Max sequential probe samples."),
    max_probe_tokens: int = typer.Option(1536, help="Cap tokens per probe sample (raise if unknown_rate is high)."),
    max_label_tokens: int = typer.Option(2048, help="Cap tokens for offline true_label generation."),
    jitter_repeats: int = typer.Option(0, help="Re-run true_label R times per prompt."),
    bootstrap: bool = typer.Option(False, help="Use bootstrap in-zone threshold."),
    randomness: str = typer.Option("", help="Window randomness for slice filtering."),
    enforce_slice: bool = typer.Option(False, help="Alias for --sample-mode slice when randomness is set."),
    sample_mode: str = typer.Option(
        "uniform",
        help="uniform | slice | prefilter — slice/prefilter need --randomness.",
    ),
    gen_backend: str = typer.Option("hf", help="Generation backend: hf or vllm."),
    gpu_mem_util: float = typer.Option(0.85, help="vLLM GPU memory utilization."),
    max_model_len: int = typer.Option(0, help="Optional vLLM max model length."),
    csv_out: str = typer.Option("harness_out/calibration.csv", help="Per-prompt CSV path."),
    report_out: str = typer.Option("harness_out/calibration_report.json", help="Summary JSON path."),
) -> None:
    from harness.calibrate import run_calibration
    from harness.calibrate_metrics import write_calibration_csv, write_calibration_report
    from harness.probe_logic import ProbeConfig

    data, loaded = _load_model_and_data(
        checkpoint_repo_id, checkpoint_revision, device, attn, dtype, cache_dir,
    )
    vllm_engine = _maybe_load_vllm_engine(
        loaded=loaded,
        gen_backend=gen_backend,
        gpu_mem_util=gpu_mem_util,
        max_model_len=max_model_len or None,
    )
    indices = _resolve_indices(
        data.env,
        prompts=prompts,
        count=count,
        seed=seed,
        randomness=randomness,
        sample_mode=sample_mode,
        enforce_slice=enforce_slice,
    )
    report = run_calibration(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        env=data.env,
        prompt_indices=indices,
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        probe_config=ProbeConfig(max_samples=max_samples, max_probe_tokens=max_probe_tokens),
        bootstrap=bootstrap,
        jitter_repeats=jitter_repeats,
        randomness=randomness or None,
        enforce_slice=enforce_slice,
        sample_mode=sample_mode,  # type: ignore[arg-type]
        requested_prompts=count,
        max_label_tokens=max_label_tokens,
        gen_backend=gen_backend,
        vllm_engine=vllm_engine,
    )
    write_calibration_csv(csv_out, report.rows)
    write_calibration_report(report_out, report)
    typer.echo(json.dumps({
        "checkpoint_revision": checkpoint_revision,
        "n_prompts": report.n_prompts,
        "sample_mode": report.sample_mode,
        "n_in_zone_true": report.balance.n_in_zone_true,
        "unknown_rate": report.balance.unknown_rate,
        "label_truncation_rate": report.balance.label_truncation_rate,
        "metrics": asdict(report.metrics),
        "compute_saved_per_in_zone": report.compute_saved_per_in_zone,
        "csv_out": csv_out,
        "report_out": report_out,
    }, indent=2, sort_keys=True))


@app.command("grail-check")
def grail_check_cmd(
    prompt_idx: int = typer.Option(..., help="Prompt index for GRAIL fidelity check."),
    checkpoint_repo_id: str = typer.Option(..., help="HF checkpoint repo id."),
    checkpoint_revision: str = typer.Option(..., help="HF checkpoint revision."),
    randomness: str = typer.Option(..., help="Window randomness hex for GRAIL challenge."),
    device: str = typer.Option("cuda", help="Must be cuda for Step 5."),
    attn: str = typer.Option("flash_attention_2", help="Must be flash_attention_2 for Step 5."),
    dtype: str = typer.Option("bfloat16", help="bf16/fp16/fp32 style dtype."),
    cache_dir: str = typer.Option("", help="Optional HF cache dir."),
    rollouts: int = typer.Option(1, help="Number of rollouts to verify."),
) -> None:
    from harness.grail_check import run_grail_fidelity

    runtime = resolve_runtime_config(device=device, attn=attn, dtype=dtype)
    require_flash_attention_for_grail(runtime)
    data, loaded = _load_model_and_data(
        checkpoint_repo_id, checkpoint_revision, device, attn, dtype, cache_dir,
    )
    problem = data.env.get_problem(prompt_idx)
    results = run_grail_fidelity(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompt=problem["prompt"],
        randomness=randomness,
        rollouts=rollouts,
    )
    typer.echo(json.dumps({
        "checkpoint_revision": checkpoint_revision,
        "randomness": randomness,
        "all_passed": all(r.passed for r in results),
        "results": [asdict(r) for r in results],
    }, indent=2, sort_keys=True))


@app.command("latency")
def latency_cmd(
    prompt_idx: int = typer.Option(..., help="Prompt index for latency measurement."),
    checkpoint_repo_id: str = typer.Option(..., help="HF checkpoint repo id."),
    checkpoint_revision: str = typer.Option(..., help="HF checkpoint revision."),
    randomness: str = typer.Option(..., help="Window randomness hex for proof path."),
    device: str = typer.Option("cuda", help="Must be cuda for Step 5."),
    attn: str = typer.Option("flash_attention_2", help="Must be flash_attention_2 for Step 5."),
    dtype: str = typer.Option("bfloat16", help="bf16/fp16/fp32 style dtype."),
    cache_dir: str = typer.Option("", help="Optional HF cache dir."),
    generation_runs: int = typer.Option(3, help="Timed 8-rollout generation runs."),
    proof_runs: int = typer.Option(3, help="Timed per-rollout proof runs."),
    proofs_batched: bool = typer.Option(False, help="Model proof as one batched pass over 8 rollouts."),
    gen_backend: str = typer.Option("hf", help="Generation backend: hf or vllm."),
    gpu_mem_util: float = typer.Option(0.85, help="vLLM GPU memory utilization."),
    max_model_len: int = typer.Option(0, help="Optional vLLM max model length."),
    window_seconds: float = typer.Option(45.0, help="Target window duration for worker sizing."),
) -> None:
    from harness.latency import latency_report_dict, measure_latency

    runtime = resolve_runtime_config(device=device, attn=attn, dtype=dtype)
    require_flash_attention_for_grail(runtime)
    data, loaded = _load_model_and_data(
        checkpoint_repo_id, checkpoint_revision, device, attn, dtype, cache_dir,
    )
    vllm_engine = _maybe_load_vllm_engine(
        loaded=loaded,
        gen_backend=gen_backend,
        gpu_mem_util=gpu_mem_util,
        max_model_len=max_model_len or None,
    )
    problem = data.env.get_problem(prompt_idx)
    report = measure_latency(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompt=problem["prompt"],
        randomness=randomness,
        checkpoint_revision=checkpoint_revision,
        generation_runs=generation_runs,
        proof_runs=proof_runs,
        window_seconds=window_seconds,
        proofs_batched=proofs_batched,
        gen_backend=gen_backend,
        vllm_engine=vllm_engine,
    )
    typer.echo(json.dumps(latency_report_dict(report), indent=2, sort_keys=True))


@app.command("report")
def report_cmd(
    checkpoint_repo_id: str = typer.Option(..., help="HF checkpoint repo id."),
    checkpoint_revision: str = typer.Option(..., help="HF checkpoint revision."),
    randomness: str = typer.Option(..., help="Window randomness hex."),
    prompts: str = typer.Option("", help="Comma-separated prompt indices."),
    count: int = typer.Option(10, help="Random prompt count when --prompts omitted (use 300+ for stable metrics)."),
    seed: int = typer.Option(0, help="RNG seed for random prompt selection."),
    device: str = typer.Option("cuda", help="cpu for partial dev; cuda for full report."),
    attn: str = typer.Option("flash_attention_2", help="eager on CPU dev; flash_attention_2 on H200."),
    dtype: str = typer.Option("bfloat16", help="bf16/fp16/fp32 style dtype."),
    cache_dir: str = typer.Option("", help="Optional HF cache dir."),
    max_samples: int = typer.Option(6, help="Max sequential probe samples."),
    max_probe_tokens: int = typer.Option(1536, help="Cap tokens per probe sample (raise if unknown_rate is high)."),
    max_label_tokens: int = typer.Option(2048, help="Cap tokens for offline true_label generation."),
    jitter_repeats: int = typer.Option(2, help="Re-run true_label R times per prompt."),
    bootstrap: bool = typer.Option(False, help="Use bootstrap in-zone threshold."),
    enforce_slice: bool = typer.Option(False, help="Alias for --sample-mode slice."),
    sample_mode: str = typer.Option(
        "slice",
        help="uniform | slice | prefilter — slice/prefilter restrict to window.",
    ),
    gen_backend: str = typer.Option("hf", help="Generation backend: hf or vllm."),
    gpu_mem_util: float = typer.Option(0.85, help="vLLM GPU memory utilization."),
    max_model_len: int = typer.Option(0, help="Optional vLLM max model length."),
    skip_grail: bool = typer.Option(False, help="Skip Step 5 GRAIL/latency (CPU dev only)."),
    grail_rollouts: int = typer.Option(1, help="GRAIL fidelity rollouts to verify."),
    latency_prompt_idx: int = typer.Option(-1, help="Prompt for latency; default first sampled."),
    generation_runs: int = typer.Option(3, help="Timed 8-rollout generation runs."),
    proof_runs: int = typer.Option(3, help="Timed per-rollout proof runs."),
    proofs_batched: bool = typer.Option(False, help="Model proof as one batched pass over 8 rollouts."),
    window_seconds: float = typer.Option(45.0, help="Target window duration for worker sizing."),
    report_out: str = typer.Option("harness_out/full_report.json", help="Summary JSON path."),
    csv_out: str = typer.Option("harness_out/calibration.csv", help="Per-prompt CSV path."),
    calibration_report_out: str = typer.Option(
        "",
        help="Standalone calibration JSON (default: harness_out/calibration_report.json).",
    ),
) -> None:
    from harness.probe_logic import ProbeConfig
    from harness.report import run_full_report, write_report_artifacts

    runtime = resolve_runtime_config(device=device, attn=attn, dtype=dtype)
    run_grail = not skip_grail
    run_latency = not skip_grail
    if run_grail:
        require_flash_attention_for_grail(runtime)

    data, loaded = _load_model_and_data(
        checkpoint_repo_id, checkpoint_revision, device, attn, dtype, cache_dir,
    )
    vllm_engine = _maybe_load_vllm_engine(
        loaded=loaded,
        gen_backend=gen_backend,
        gpu_mem_util=gpu_mem_util,
        max_model_len=max_model_len or None,
    )
    indices = _resolve_indices(
        data.env,
        prompts=prompts,
        count=count,
        seed=seed,
        randomness=randomness,
        sample_mode=sample_mode,
        enforce_slice=enforce_slice,
    )
    lat_idx = latency_prompt_idx if latency_prompt_idx >= 0 else None

    full = run_full_report(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        env=data.env,
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_revision=checkpoint_revision,
        randomness=randomness,
        prompt_indices=indices,
        prompt_count=count,
        seed=seed,
        enforce_slice=enforce_slice,
        sample_mode=sample_mode,  # type: ignore[arg-type]
        probe_config=ProbeConfig(max_samples=max_samples, max_probe_tokens=max_probe_tokens),
        max_label_tokens=max_label_tokens,
        gen_backend=gen_backend,
        vllm_engine=vllm_engine,
        bootstrap=bootstrap,
        jitter_repeats=jitter_repeats,
        run_grail=run_grail,
        run_latency=run_latency,
        grail_rollouts=grail_rollouts,
        latency_prompt_idx=lat_idx,
        generation_runs=generation_runs,
        proof_runs=proof_runs,
        window_seconds=window_seconds,
        proofs_batched=proofs_batched,
    )
    cal_report_path = calibration_report_out or "harness_out/calibration_report.json"
    write_report_artifacts(
        full,
        report_json=report_out,
        calibration_csv=csv_out,
        calibration_report_json=cal_report_path,
    )
    typer.echo(json.dumps({
        "checkpoint_revision": checkpoint_revision,
        "randomness": randomness,
        "slice_bounds": list(full.slice_bounds),
        "sample_mode": full.sample_mode,
        "n_prompts": full.calibration.n_prompts,
        "n_in_zone_true": full.calibration.balance.n_in_zone_true,
        "unknown_rate": full.calibration.balance.unknown_rate,
        "label_truncation_rate": full.calibration.balance.label_truncation_rate,
        "probe_precision": full.calibration.metrics.precision,
        "probe_recall": full.calibration.metrics.recall,
        "probe_f1": full.calibration.metrics.f1,
        "compute_saved_per_in_zone": full.calibration.compute_saved_per_in_zone,
        "jitter_band_recommendation": full.jitter_band_recommendation,
        "grail_all_passed": full.grail_all_passed,
        "latency_p90_cycle_seconds": (
            full.latency.cycle_p90_seconds if full.latency else None
        ),
        "suggested_workers": full.suggested_workers,
        "report_out": report_out,
        "calibration_report_out": cal_report_path,
        "csv_out": csv_out,
    }, indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()

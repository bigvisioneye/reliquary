# Reliquary Offline Harness

Offline calibration and validation harness for Reliquary miner strategy work.

## Hardware expectations

| Step | Command(s) | Hardware |
|------|------------|----------|
| 1 | `resolve` | CPU |
| 2 | model/data load | CPU or GPU (`--device cpu` for dev) |
| 3 | `label` | CPU or GPU (`--attn eager` on CPU) |
| 4 | `probe`, `calibrate` | CPU or GPU (`--max-probe-tokens 128` for cheap CPU runs) |
| 5 | `grail-check`, `latency` | **H200 + `flash_attention_2`** (enforced by CLI) |
| 6 | slice (`--randomness`, `--sample-mode slice`) | CPU logic; generation still needs model |
| 7 | `report` | Full run on H200; use `--skip-grail` for CPU-only dev |

Default dev flags: `--device cpu --attn eager`.

For fast GPU generation, install vLLM and opt in:

```bash
pip install vllm
```

Then pass `--gen-backend vllm` (proof/GRAIL stays on HF).

## Commands

```bash
# Step 1
python3 -m harness resolve --checkpoint-repo-id <repo> --checkpoint-revision <rev>

# Step 3
python3 -m harness label --prompt-idx 42 --checkpoint-repo-id <repo> --checkpoint-revision <rev> \
  --device cpu --attn eager

# Step 4
python3 -m harness calibrate --checkpoint-repo-id <repo> --checkpoint-revision <rev> \
  --count 20 --device cpu --attn eager --max-probe-tokens 128 --max-label-tokens 512 \
  --randomness <window-randomness-hex> --sample-mode slice

# Step 5 (H200 only)
python3 -m harness grail-check --prompt-idx 42 --checkpoint-repo-id <repo> \
  --checkpoint-revision <rev> --randomness <hex> --device cuda --attn flash_attention_2

python3 -m harness latency --prompt-idx 42 --checkpoint-repo-id <repo> \
  --checkpoint-revision <rev> --randomness <hex> --device cuda --attn flash_attention_2

# Step 7 one-shot report (H200 rental artifact) — recommended production run
python3 -m harness report --checkpoint-repo-id <repo> --checkpoint-revision <rev> \
  --randomness <hex> --count 300 --sample-mode prefilter --max-probe-tokens 1536 \
  --max-label-tokens 2048 --gen-backend vllm --gpu-mem-util 0.85 \
  --device cuda --attn flash_attention_2 \
  --report-out harness_out/full_report.json --csv-out harness_out/calibration.csv

# CPU dev partial report (skips GRAIL/latency)
python3 -m harness report ... --device cpu --attn eager --skip-grail --max-probe-tokens 128
```

## Design notes

- Protocol constants imported from `reliquary.constants` only.
- GRAIL verification reuses `verify_commitment_proofs` and miner-style `build_grail_commit`.
- Prompt slice uses `reliquary.shared.prompt_range.window_prompt_range`.
- `harness/crowding.py` is a stub seam for future live-miner crowding weighting.
- Latency `proof_per_rollout` times build+verify only on a pre-generated rollout; generation is measured separately as one batched 8-rollout call.
- Generation backend can be `hf` (default, CPU-safe) or `vllm` (GPU generation only).
- GRAIL build/verify always uses the HF model path; vLLM is generation-only.
- `full_report.json` and `calibration_report.json` share the same calibration object (`n_prompts`, `metrics`, `balance`, `jitter`).

## Sampling modes

- `uniform`: sample across the full prompt universe (can skew easy at trained checkpoints).
- `slice`: sample only from the current window slice (matches live miner visibility).
- `prefilter`: draw a larger candidate pool from the slice, run a cheap 2-sample probe, and keep a balanced mix before expensive `true_label` runs.

Use `--count 300` or more for stable precision/recall. Check `unknown_rate` in the report; if high, raise `--max-probe-tokens` (try 1024–2048).
Check `label_truncation_rate` in the report; if high, raise `--max-label-tokens`.

## First H200 boot checklist

Before trusting Step 5 numbers, run once on the rented GPU:

```bash
python3 -m harness grail-check --prompt-idx 42 --checkpoint-repo-id <repo> \
  --checkpoint-revision <rev> --randomness <hex> --device cuda --attn flash_attention_2
```

If this fails on import (e.g. `forward_single_layer`, `indices_from_root`), it is a
signature drift in repo internals — fix imports, not probe logic. Check
`margin_is_fallback` on the result: when `false`, per-position margins are authoritative.

## Probe note (truncation)

Short probe budgets (`--max-probe-tokens`) may cut thinking-model completions before
`\\boxed{}` appears. Truncated samples are classified as **unknown**, not failures, so
`p_hat` is computed only from resolved (natural-EOS) samples. This avoids downward bias
in calibration. The report includes `unknown_rate` to diagnose insufficient probe budget.

## Tests (CPU-safe, no model)

```bash
pytest --confcutdir=tests/harness -q tests/harness/
```

## Progress logging (long runs / nohup)

`report` and `calibrate` emit timestamped progress logs by default (`INFO`).
Use `--log-level DEBUG` for per-probe-sample detail.

For `nohup`, use unbuffered stdout so lines appear immediately:

```bash
PYTHONUNBUFFERED=1 nohup python3 -u -m harness report ... > harness_out/report.log 2>&1 &
tail -f harness_out/report.log
```

Example log stages: model load → prefilter candidates → per-prompt probe/label → GRAIL → latency → write artifacts.

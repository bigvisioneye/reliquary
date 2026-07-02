#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/reliquary"
cd "$ROOT"

# Load wallet / validator / env settings
set -a
source "$ROOT/scripts/.env"
set +a

# Torch CUDA libs (needed for flash-attn on some boxes)
export LD_LIBRARY_PATH="$("$ROOT/venv/bin/python" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):${LD_LIBRARY_PATH:-}"

exec "$ROOT/venv/bin/python" -u -m reliquary.cli.main mine \
  --network "${BT_NETWORK}" \
  --netuid "${NETUID}" \
  --wallet-name "${BT_WALLET_NAME}" \
  --hotkey "${BT_HOTKEY}" \
  --checkpoint "${RELIQUARY_CHECKPOINT:-Qwen/Qwen3.5-4B}" \
  --environments "${RELIQUARY_ENVIRONMENTS:-openmathinstruct}" \
  --log-level INFO \
  ${BT_WALLET_PATH:+--wallet-path "$BT_WALLET_PATH"} \
  ${RELIQUARY_VALIDATOR_URL:+--validator-url "$RELIQUARY_VALIDATOR_URL"}
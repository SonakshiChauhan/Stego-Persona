#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME"

out="${1:-/workspace/runs/stego_qwen38_thinking_v1}"
candidates="${2:-30000}"
if [[ ! -f "$out/cases.jsonl" ]]; then
    python -m stego prepare --out "$out" --candidates "$candidates"
fi
python -m stego run --out "$out"

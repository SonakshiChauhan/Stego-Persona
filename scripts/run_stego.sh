#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 2 || "$1" != "--count" || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash scripts/run_stego.sh --count GROUPS_PER_SCHEME" >&2
    exit 2
fi
if [[ ! -d .venv ]]; then
    python3 -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -q -r data_generation/requirements.txt
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME"
python -m stego validate
split="${STEGO_SPLIT:-train}"
out="${STEGO_OUT:-/workspace/runs/qwen38-27b-$split}"
candidates="${STEGO_CANDIDATES:-30000}"
if [[ ! -f "$out/cases.jsonl" ]]; then
    python -m stego prepare --out "$out" --split "$split" --candidates "$candidates"
fi
python -m stego run --out "$out" --per-scheme "$2"

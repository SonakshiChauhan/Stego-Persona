#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
mkdir -p "$HF_HOME"

python -m pip install -r data_generation/requirements.txt
python -m spacy download en_core_web_sm

python - <<'PY'
import os
import torch
assert torch.cuda.is_available(), "No CUDA GPU is visible"
assert torch.cuda.is_bf16_supported(), "The installed PyTorch/GPU does not support BF16"
assert torch.cuda.get_device_properties(0).total_memory >= 75 * 1024**3, "Qwen3.8-27B BF16 needs an 80 GB-class GPU"
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
print("HF_NAME set:", bool(os.getenv("HF_NAME")))
print("HF_TOKEN set:", bool(os.getenv("HF_TOKEN")))
PY

python -m stego validate

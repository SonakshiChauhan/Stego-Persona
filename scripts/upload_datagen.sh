#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

if [[ $# -ne 3 ]]; then
    echo "Usage: bash scripts/upload_datagen.sh TRAIN_RUN VAL_RUN TEST_RUN" >&2
    exit 2
fi
: "${HF_NAME:?Set HF_NAME to your Hugging Face account or organization}"
repo="${HF_REPO:-$HF_NAME/qwen3-14b}"
dest="$(mktemp -d)"
trap 'rm -r "$dest"' EXIT

for out in "$@"; do
    python -m stego package --out "$out" --dest "$dest"
done
hf repo create "$repo" --repo-type dataset --private --exist-ok
hf upload "$repo" "$dest" . --repo-type dataset

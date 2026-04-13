#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export THESIS_SHARED_ROOT="${THESIS_SHARED_ROOT:-/nfsdata/ganbei/chenwei_thesis}"
export TORCH_HOME="${TORCH_HOME:-$THESIS_SHARED_ROOT/torch_cache}"
export HF_HOME="${HF_HOME:-$THESIS_SHARED_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$THESIS_SHARED_ROOT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$THESIS_SHARED_ROOT/hf_cache}"

mkdir -p \
  "$THESIS_SHARED_ROOT/data/cifar10" \
  "$THESIS_SHARED_ROOT/torch_cache" \
  "$THESIS_SHARED_ROOT/hf_cache" \
  "$THESIS_SHARED_ROOT/checkpoints" \
  "$THESIS_SHARED_ROOT/outputs" \
  "$THESIS_SHARED_ROOT/logs"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -f ".venv/bin/activate" ]]; then
  # Reuse a local virtualenv when available on the debug node.
  source .venv/bin/activate
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" scripts/prepare_cifar10.py --data-dir "$THESIS_SHARED_ROOT/data/cifar10"
"$PYTHON_BIN" scripts/train_cifar_vit.py --config configs/current/b0_standard_paper_cifar500ep_bs128_a800.yaml

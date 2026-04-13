#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export THESIS_SHARED_ROOT="${THESIS_SHARED_ROOT:-/nfsdata/ganbei/chenwei_thesis}"
export TORCH_HOME="${TORCH_HOME:-$THESIS_SHARED_ROOT/torch_cache}"
export HF_HOME="${HF_HOME:-$THESIS_SHARED_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$THESIS_SHARED_ROOT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$THESIS_SHARED_ROOT/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

mkdir -p \
  "$THESIS_SHARED_ROOT/data/cifar10" \
  "$THESIS_SHARED_ROOT/torch_cache" \
  "$THESIS_SHARED_ROOT/hf_cache" \
  "$THESIS_SHARED_ROOT/checkpoints" \
  "$THESIS_SHARED_ROOT/outputs" \
  "$THESIS_SHARED_ROOT/logs"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" scripts/prepare_cifar10.py --data-dir "$THESIS_SHARED_ROOT/data/cifar10"
"$PYTHON_BIN" scripts/train_cifar_vit_method3.py --config configs/method3/b0_band_ratio_regularization_500ep_bs128_a800_seed44.yaml

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_cifar10.py --data-dir data/cifar10
python scripts/train_cifar_vit_method2.py --config configs/method2/b0_forward_structural_prior_20ep.yaml

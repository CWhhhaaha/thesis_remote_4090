#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate
python scripts/train_cifar_vit_method3.py --config configs/flowers102/m3_ratio_vit_base_patch16_224_30ep.yaml

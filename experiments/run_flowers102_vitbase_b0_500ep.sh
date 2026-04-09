#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate
python scripts/train_cifar_vit.py --config configs/flowers102/b0_vit_base_patch16_224_500ep.yaml

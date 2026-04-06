#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ../../.venv/bin/activate
python -m pip install -r requirements.txt

python analyze_qk_geometry.py \
  --inventory qk_interaction_vision_inventory.json \
  --split primary \
  --output-dir outputs/qk_interaction_vision_subset \
  --device cpu \
  --cache-dir hf_cache \
  --local-files-only \
  --skip-existing

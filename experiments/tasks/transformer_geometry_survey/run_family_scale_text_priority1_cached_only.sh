#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ../../.venv/bin/activate
python -m pip install -r requirements.txt

python analyze_qk_geometry.py \
  --inventory family_scale_text_inventory.json \
  --split primary \
  --max-priority 1 \
  --output-dir outputs/family_scale_text_subset_priority1 \
  --device cpu \
  --cache-dir hf_cache \
  --local-files-only \
  --skip-existing

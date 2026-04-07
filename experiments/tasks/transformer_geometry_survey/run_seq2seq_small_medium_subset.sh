#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ../../.venv/bin/activate
python -m pip install -r requirements.txt

python analyze_qk_geometry.py \
  --inventory seq2seq_small_medium_inventory.json \
  --split primary \
  --output-dir outputs/seq2seq_small_medium_subset \
  --device cpu \
  --cache-dir hf_cache \
  --download-retries 5 \
  --retry-sleep-sec 5 \
  --skip-existing

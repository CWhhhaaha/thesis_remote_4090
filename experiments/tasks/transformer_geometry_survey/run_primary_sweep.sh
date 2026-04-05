#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "../../.venv" ]]; then
  (
    cd ../..
    python -m venv .venv
  )
fi

source ../../.venv/bin/activate
pip install -r requirements.txt

python analyze_qk_geometry.py \
  --split primary \
  --output-dir outputs/primary_sweep

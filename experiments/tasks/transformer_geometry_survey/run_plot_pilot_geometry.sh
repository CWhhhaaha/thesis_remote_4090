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
python -m pip install matplotlib

python plot_pilot_geometry.py \
  --input-dir outputs/pilot_subset \
  --fig-dir outputs/pilot_subset_figures

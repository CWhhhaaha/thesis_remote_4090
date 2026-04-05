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
  --local-files-only \
  --cache-dir hf_cache \
  --skip-existing \
  --models \
    google/vit-base-patch16-224 \
    facebook/dinov2-base \
    google-bert/bert-base-uncased \
    microsoft/deberta-v3-base \
    openai-community/gpt2 \
    facebook/opt-350m \
    t5-base \
    facebook/bart-base \
    facebook/wav2vec2-base \
    openai/whisper-small \
    openai/clip-vit-base-patch32 \
    dandelin/vilt-b32-mlm \
  --output-dir outputs/pilot_subset

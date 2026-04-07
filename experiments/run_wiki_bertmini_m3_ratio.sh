#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate
python scripts/train_wiki_bert_mlm.py --config configs/language_mlm/m3_ratio_wiki_bertmini_200k.yaml

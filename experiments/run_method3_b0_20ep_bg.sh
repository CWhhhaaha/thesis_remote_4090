#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/logs

nohup bash run_method3_b0_20ep.sh > outputs/logs/method3_b0_20ep.log 2>&1 &
echo "Started Method3 B0 20-epoch run. Log: outputs/logs/method3_b0_20ep.log"

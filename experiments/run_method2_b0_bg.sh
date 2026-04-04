#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/logs
LOG_PATH="outputs/logs/method2_b0_$(date +%Y%m%d_%H%M%S).log"

nohup bash run_method2_b0.sh > "$LOG_PATH" 2>&1 &
echo "Started Method2 B0 in background"
echo "Log: $LOG_PATH"

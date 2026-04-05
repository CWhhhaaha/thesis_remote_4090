#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/method3_b0_target_ratio_20ep_x10.log"

nohup bash run_method3_b0_target_ratio_20ep_x10.sh > "$LOG_FILE" 2>&1 &
echo "Started method3 target-ratio x10 B0 20-epoch run in background."
echo "Log: $LOG_FILE"

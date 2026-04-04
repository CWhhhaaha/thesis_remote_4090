#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/method2_b0_500ep.log"

nohup bash run_method2_b0_500ep.sh > "$LOG_FILE" 2>&1 &
echo "Started method2 B0 500-epoch run in background."
echo "Log: $LOG_FILE"

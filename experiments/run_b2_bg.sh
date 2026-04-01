#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/logs
LOG_PATH="outputs/logs/b2_$(date +%Y%m%d_%H%M%S).log"

nohup bash run_b2.sh > "$LOG_PATH" 2>&1 &
echo "Started B2 in background"
echo "Log: $LOG_PATH"

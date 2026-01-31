#!/usr/bin/env bash 
# 
# gpu_listen.sh  Andrew Belles  Jan 31st, 2026 
# 
# Listens to GPU id on peak vram and utilization 
#
#

set -euo pipefail

DEV=0
INTERVAL=1

while getopts ":d:i:" opt; do
  case "$opt" in
    d) DEV="$OPTARG" ;;
    i) INTERVAL="$OPTARG" ;;
    *) echo "Usage: $0 [-d device] [-i interval_seconds]" >&2; exit 1 ;;
  esac
done

echo "Monitoring GPU ${DEV} (interval=${INTERVAL}s). Ctrl+C to stop."

max_util=0
max_mem=0

while true; do
  line=$(nvidia-smi -i "$DEV" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null) || {
      echo "nvidia-smi failed (device $DEV?)" >&2
      exit 1
    }

  util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1); print $1}')
  mem=$(echo "$line"  | awk -F',' '{gsub(/ /,"",$2); print $2}')

  if [[ "$util" -gt "$max_util" ]]; then max_util="$util"; fi
  if [[ "$mem" -gt "$max_mem" ]]; then max_mem="$mem"; fi

  ts=$(date +"%Y-%m-%d %H:%M:%S")
  printf "[%s] GPU %s | util %3s%% (max %3s%%) | mem %5s MiB (max %5s MiB)\n" \
    "$ts" "$DEV" "$util" "$max_util" "$mem" "$max_mem"

  sleep "$INTERVAL"
done

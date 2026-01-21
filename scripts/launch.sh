#!/usr/bin/env bash 
# 
# launch(.sh)  Andrew Belles  Jan 20th, 2026 
# 
# wrapper call that ensures executables respect resources on shared computing server 
# 
# 

set -euo pipefail 

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_GPUS=0 
JOBS=1 
CMD=()

export TOPG_PACK_CACHE_MB=4096 
export TOPG_PACK_CACHE_ITEMS=400

usage() {
  cat << 'EOF'
usage: 
  launch [-g N] [-j K] <command...> 

notes: 
  -g, --num-gpus N number of GPUs to expose (0 = CPU only)
  -j, --jobs     K number of CPU cores/threads to use (-1 = greedy)
EOF
}

launch() {
  if [ $# -eq 0 ]; then
    usage 
    exit 2 
  fi 

  exec "$@"
}

while [[ $# -gt 0 ]]; do 
  case "$1" in 
    -g|--num-gpus) NUM_GPUS="$2"; shift 2 ;; 
    -j|--jobs) JOBS="$2"; shift 2 ;; 
    --) shift; CMD=("$@"); break ;; 
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;; 
  esac 
done 

if [ ${#CMD[@]} -eq 0 ]; then 
  usage 
  exit 2 
fi 

if [ "$NUM_GPUS" -le 0 ]; then 
  export CUDA_VISIBLE_DEVICES=""
elif [ "$NUM_GPUS" -eq 1 ]; then 
  export CUDA_VISIBLE_DEVICES="0"
else 
  CUDA_VISIBLE_DEVICES="$(seq $((NUM_GPUS - 1)) | paste -sd, -)"
  export CUDA_VISIBLE_DEVICES 
fi 

export TOPG_JOBS="$JOBS"
if [ "$JOBS" -gt 0 ]; then 
  export OMP_NUM_THREADS="$JOBS"
else 
  unset OMP_NUM_THREADS
fi 

launch "${CMD[@]}" 

wait 

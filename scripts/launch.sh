#!/usr/bin/env bash 
# 
# launch(.sh)  Andrew Belles  Jan 20th, 2026 
# 
# wrapper call that ensures executables respect resources on shared computing server 
# 
# 

set -euo pipefail 

###########################################################
# Vars  
###########################################################

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PREPROCESS_RAM=""

NUM_GPUS=0 
JOBS=1 
DEVICE=""
CMD=()

export TOPG_PACK_CACHE_MB=4096 
export TOPG_PACK_CACHE_ITEMS=1200

###########################################################
# Usage information   
###########################################################

usage() {
  cat << 'EOF'
usage: 
  launch [-g N] [-j K] [-d GPU ID] [-m PREPROCESS_RAM] <command...> 

notes: 
  -g, --num-gpus N number of GPUs to expose (0 = CPU only)
  -j, --jobs     K number of CPU cores/threads to use (-1 = greedy)
  -d, --device   GPU id provided by caller (overrides -g's selection) 
  -m, --ram-mb   soft RAM gap for preprocessing/tensors.py (MB)
EOF
}

###########################################################
# Launched executable command + args  
###########################################################

launch() {
  if [ $# -eq 0 ]; then
    usage 
    exit 2 
  fi 

  exec "$@"
}

###########################################################
# Selects block of available GPUs from nvidia-smi given -g N
###########################################################

select_gpu_block() {
  local count="$1"
  local candidates 

  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then 
    IFS=',' read -ra candidates <<< "$CUDA_VISIBLE_DEVICES"
  else 
    if ! command -v nvidia-smi >/dev/null 2>&1; then 
      echo "nvidia-smi not found and CUDA_VISIBLE_DEVICES is unset" >&2 
      return 1 
    fi 
    mapfile -t candidates < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  fi 

  candidates=($(printf "%s\n" "${candidates[@]}" | sed 's/ //g' | sort -n))
  if [ "${#candidates[@]}" -lt "$count" ]; then 
    echo "not enought GPUs available for block size $count" >&2 
    return 1 
  fi 

  for ((i=0; i<=${#candidates[@]}-count; i++)); do 
    local start="${candidates[$i]}"
    local ok=1 

    for ((j=1; j<count; j++)); do 
      if [ $((start + j)) -ne "${candidates[$((i + j))]}" ]; then 
        ok=0 
        break 
      fi 
    done 
    if [ "$ok" -eq 1 ]; then 
      seq "$start" $((start + count - 1)) | paste -sd, - 
      return 0 
    fi 
  done 

  printf "%s\n" "${candidates[@]:0:$count}" | paste -sd, - 
  return 0 
}

###########################################################
# Arg parsing, running  
###########################################################

while [[ $# -gt 0 ]]; do 
  case "$1" in 
    -g|--num-gpus) NUM_GPUS="$2"; shift 2 ;; 
    -j|--jobs) JOBS="$2"; shift 2 ;; 
    -d|--device) DEVICE="$2"; shift 2 ;;
    -m|--ram-mb) RAM_MB="$2"; shift 2 ;; 
    --) shift; CMD=("$@"); break ;; 
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;; 
  esac 
done 

if [ ${#CMD[@]} -eq 0 ]; then 
  usage 
  exit 2 
fi 

if [ -n "${DEVICE}" ]; then
  if [ "$DEVICE" -lt 0 ]; then
    echo "invalid device: $DEVICE" >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$DEVICE"
else
  if [ "$NUM_GPUS" -le 0 ]; then
    export CUDA_VISIBLE_DEVICES=""
  elif [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES="0"
  else
    block="$(select_gpu_block "$NUM_GPUS")" || exit 2 
    export CUDA_VISIBLE_DEVICES="$block"
  fi
fi

export TOPG_JOBS="$JOBS"
if [ "$JOBS" -gt 0 ]; then 
  export OMP_NUM_THREADS="$JOBS"
else 
  unset OMP_NUM_THREADS
fi 

if [ -n "${PREPROCESS_RAM}" ]; then 
  export TOPG_TENSOR_RAM_MB="$RAM_MB"
fi 

launch "${CMD[@]}" 

wait 

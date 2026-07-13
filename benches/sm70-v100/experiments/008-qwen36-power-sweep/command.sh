#!/usr/bin/env bash
set -euo pipefail

gpu=${V100_GPU_1_UUID}
model=${LLAMA_CPP_ROOT}/build/bin/models/Qwen_Qwen3.6-27B-Q2_K.gguf

restore_power() {
    sudo -n nvidia-smi -i "$gpu" -pl 150 >/dev/null
}
trap restore_power EXIT

for watts in 150 225 300; do
    sudo -n nvidia-smi -i "$gpu" -pl "$watts" >/dev/null
    CUDA_VISIBLE_DEVICES="$gpu" ./build/bin/llama-bench \
        -m "$model" -p 512 -n 128 -r 3 -ngl 99 -sm none -mg 0 -fa auto \
        -b 2048 -ub 512 -o json
done

#!/usr/bin/env bash

set -u -o pipefail

ROOT=$(git rev-parse --show-toplevel)
ORIGIN_WORKTREE=${ORIGIN_WORKTREE:-${UPSTREAM_WORKTREE}}
GPU_UUID=${V100_GPU_0_UUID}
EXPECTED_BRANCH=e33e5bf79b8aaab5017882fed46a3ab7a7552169
EXPECTED_ORIGIN=e8f19cc0ad70a243c8012bf17b4be601abfc8ea2
OUT_ROOT="$ROOT/benches/sm70-v100/comparisons/001-origin-master-e8f19cc0a"

MODE=${1:-full}
case "$MODE" in
    smoke)
        ROUNDS=1
        P_VALUES=32
        N_VALUES=2
        ;;
    full)
        ROUNDS=${ROUNDS:-6}
        P_VALUES=512,4096,32768
        N_VALUES=128,1024
        ;;
    *)
        echo "usage: $0 [smoke|full]" >&2
        exit 2
        ;;
esac

OUT="$OUT_ROOT/$MODE"
mkdir -p "$OUT/raw" "$OUT/logs" "$OUT/telemetry" "$OUT/commands"

TELEMETRY_PID=

cleanup() {
    if [[ -n "$TELEMETRY_PID" ]]; then
        kill "$TELEMETRY_PID" 2>/dev/null || true
        wait "$TELEMETRY_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

BRANCH_BIN="$ROOT/build/bin/llama-bench"
ORIGIN_BIN="$ORIGIN_WORKTREE/build/bin/llama-bench"
MODEL_ROOT="$ROOT/build/bin/models"

MODEL_IDS=(
    qwen35-0.8b-q4_0
    qwen35-0.8b-q8_0
    qwen35-9b-q4_k_m
    gemma4-12b-q4_k_xl
    qwen36-27b-q2_k
    gemma4-31b-q4_k_xl
)

declare -A MODEL_FILES=(
    [qwen35-0.8b-q4_0]=Qwen3.5-0.8B-Q4_0.gguf
    [qwen35-0.8b-q8_0]=Qwen3.5-0.8B-Q8_0.gguf
    [qwen35-9b-q4_k_m]=Qwen3.5-9B-Q4_K_M.gguf
    [gemma4-12b-q4_k_xl]=gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
    [qwen36-27b-q2_k]=Qwen_Qwen3.6-27B-Q2_K.gguf
    [gemma4-31b-q4_k_xl]=gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
)

declare -A MODEL_NGL=(
    [qwen35-0.8b-q4_0]=99
    [qwen35-0.8b-q8_0]=99
    [qwen35-9b-q4_k_m]=99
    [gemma4-12b-q4_k_xl]=99
    [qwen36-27b-q2_k]=99
    [gemma4-31b-q4_k_xl]=42
)

declare -A BINS=(
    [origin]="$ORIGIN_BIN"
    [branch]="$BRANCH_BIN"
)

declare -A EXPECTED_COMMITS=(
    [origin]=${EXPECTED_ORIGIN:0:9}
    [branch]=${EXPECTED_BRANCH:0:9}
)

fail() {
    echo "error: $*" >&2
    exit 1
}

trim() {
    local value=$1
    value=${value#"${value%%[![:space:]]*}"}
    value=${value%"${value##*[![:space:]]}"}
    printf '%s' "$value"
}

validate_environment() {
    [[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_BRANCH" ]] || fail "unexpected branch revision"
    [[ $(git -C "$ORIGIN_WORKTREE" rev-parse HEAD) == "$EXPECTED_ORIGIN" ]] || fail "unexpected origin revision"
    [[ -x "$BRANCH_BIN" ]] || fail "missing branch llama-bench"
    [[ -x "$ORIGIN_BIN" ]] || fail "missing origin llama-bench"

    local id
    for id in "${MODEL_IDS[@]}"; do
        [[ -r "$MODEL_ROOT/${MODEL_FILES[$id]}" ]] || fail "missing model ${MODEL_FILES[$id]}"
    done

    local state power graphics memory
    state=$(nvidia-smi -i "$GPU_UUID" \
        --query-gpu=power.limit,clocks.current.graphics,clocks.current.memory \
        --format=csv,noheader,nounits) || fail "cannot query V100 state"
    IFS=, read -r power graphics memory <<< "$state"
    power=$(trim "$power")
    graphics=$(trim "$graphics")
    memory=$(trim "$memory")
    [[ "$power" == 300.00 ]] || fail "V100 power limit is $power W, expected 300.00 W"
    [[ "$graphics" == 1192 ]] || fail "V100 graphics clock is $graphics MHz, expected 1192 MHz"
    [[ "$memory" == 877 ]] || fail "V100 memory clock is $memory MHz, expected 877 MHz"
}

wait_cool() {
    local samples=0
    local waited=0
    local state temp util

    while (( samples < 3 )); do
        state=$(nvidia-smi -i "$GPU_UUID" \
            --query-gpu=temperature.gpu,utilization.gpu \
            --format=csv,noheader,nounits) || fail "cannot query V100 temperature"
        IFS=, read -r temp util <<< "$state"
        temp=$(trim "$temp")
        util=$(trim "$util")
        if (( temp <= 70 && util == 0 )); then
            ((samples++))
        else
            samples=0
        fi
        if (( samples < 3 )); then
            if (( waited % 10 == 0 )); then
                echo "cooldown: ${temp} C, ${util}% GPU utilization"
            fi
            sleep 1
            ((waited++))
        fi
    done
}

valid_result() {
    local path=$1
    local expected=$2
    local expected_ngl=$3
    local count

    [[ -s "$path" ]] || return 1
    count=$(jq 'length' "$path" 2>/dev/null) || return 1
    if [[ "$MODE" == full ]]; then
        [[ "$count" == 5 ]] || return 1
    else
        [[ "$count" == 2 ]] || return 1
    fi
    jq -e --arg expected "$expected" --argjson expected_ngl "$expected_ngl" \
        'all(.[]; .build_commit == $expected and .n_gpu_layers == $expected_ngl)' \
        "$path" >/dev/null
}

run_one() {
    local round=$1
    local model_id=$2
    local label=$3
    local stem
    stem=$(printf 'r%02d-%s-%s' "$round" "$model_id" "$label")

    local json="$OUT/raw/$stem.json"
    local tmp="$json.tmp"
    local log="$OUT/logs/$stem.log"
    local telemetry="$OUT/telemetry/$stem.csv"
    local command="$OUT/commands/$stem.txt"
    local model="$MODEL_ROOT/${MODEL_FILES[$model_id]}"
    local bin=${BINS[$label]}
    local expected=${EXPECTED_COMMITS[$label]}
    local ngl=${MODEL_NGL[$model_id]}

    local -a args=(
        "$bin"
        -m "$model"
        -p "$P_VALUES"
        -n "$N_VALUES"
        -r 1
        -ngl "$ngl"
        -sm none
        -mg 0
        -fa auto
        -b 2048
        -ub 512
        -t 8
        -o json
    )

    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_UUID" > "$command"
    printf '%q' "${args[0]}" >> "$command"
    local arg
    for arg in "${args[@]:1}"; do
        printf ' %q' "$arg" >> "$command"
    done
    printf '\n' >> "$command"

    printf '%s\n' \
        'timestamp,graphics_clock_mhz,memory_clock_mhz,power_w,temperature_c,gpu_util_pct,memory_used_mib,sw_power_cap,sw_thermal_slowdown,hw_thermal_slowdown' \
        > "$telemetry"
    nvidia-smi -i "$GPU_UUID" \
        --query-gpu=timestamp,clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown \
        --format=csv,noheader,nounits -lms 50 >> "$telemetry" &
    TELEMETRY_PID=$!

    echo "start: round=$round model=$model_id revision=$label ngl=$ngl"
    local start end status
    start=$(date +%s)
    CUDA_VISIBLE_DEVICES="$GPU_UUID" "${args[@]}" > "$tmp" 2> "$log"
    status=$?
    end=$(date +%s)

    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=

    if (( status != 0 )); then
        rm -f "$tmp"
        fail "$stem exited with status $status; see $log"
    fi
    if ! valid_result "$tmp" "$expected" "$ngl"; then
        fail "$stem produced invalid JSON or revision metadata"
    fi
    mv "$tmp" "$json"

    echo "done: $stem in $((end - start)) s"
    jq -r '.[] | "  " + (if .n_prompt > 0 then "pp" + (.n_prompt|tostring) else "tg" + (.n_gen|tostring) end) + " = " + (.avg_ts|tostring) + " tok/s"' "$json"
}

pair_complete() {
    local round=$1
    local model_id=$2
    local origin branch
    origin=$(printf '%s/raw/r%02d-%s-origin.json' "$OUT" "$round" "$model_id")
    branch=$(printf '%s/raw/r%02d-%s-branch.json' "$OUT" "$round" "$model_id")
    local ngl=${MODEL_NGL[$model_id]}
    valid_result "$origin" "${EXPECTED_COMMITS[origin]}" "$ngl" && \
        valid_result "$branch" "${EXPECTED_COMMITS[branch]}" "$ngl"
}

run_pair() {
    local round=$1
    local model_id=$2
    local -a order

    if pair_complete "$round" "$model_id"; then
        echo "skip: completed round=$round model=$model_id"
        return
    fi

    rm -f \
        "$OUT/raw/$(printf 'r%02d-%s-' "$round" "$model_id")"*.json \
        "$OUT/raw/$(printf 'r%02d-%s-' "$round" "$model_id")"*.json.tmp \
        "$OUT/logs/$(printf 'r%02d-%s-' "$round" "$model_id")"*.log \
        "$OUT/telemetry/$(printf 'r%02d-%s-' "$round" "$model_id")"*.csv \
        "$OUT/commands/$(printf 'r%02d-%s-' "$round" "$model_id")"*.txt

    wait_cool
    if (( round % 2 == 1 )); then
        order=(origin branch)
    else
        order=(branch origin)
    fi
    run_one "$round" "$model_id" "${order[0]}"
    run_one "$round" "$model_id" "${order[1]}"
}

validate_environment

echo "mode=$MODE rounds=$ROUNDS pp=$P_VALUES tg=$N_VALUES"
for ((round = 1; round <= ROUNDS; round++)); do
    for model_id in "${MODEL_IDS[@]}"; do
        run_pair "$round" "$model_id"
    done
done

echo "benchmark matrix complete: $OUT"

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
        WORKLOAD_IDS=(pp32 tg2)
        ;;
    full)
        ROUNDS=${ROUNDS:-6}
        WORKLOAD_IDS=(pp512 pp4096 pp32768 tg128 tg1024)
        ;;
    *)
        echo "usage: $0 [smoke|full]" >&2
        exit 2
        ;;
esac

LOCKED_GRAPHICS=1192
LOCKED_MEMORY=877
COOLDOWN_TEMP=50
BUSY_UTIL=1

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
)

declare -A MODEL_FILES=(
    [qwen35-0.8b-q4_0]=Qwen3.5-0.8B-Q4_0.gguf
    [qwen35-0.8b-q8_0]=Qwen3.5-0.8B-Q8_0.gguf
    [qwen35-9b-q4_k_m]=Qwen3.5-9B-Q4_K_M.gguf
    [gemma4-12b-q4_k_xl]=gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
    [qwen36-27b-q2_k]=Qwen_Qwen3.6-27B-Q2_K.gguf
)

declare -A MODEL_NGL=(
    [qwen35-0.8b-q4_0]=99
    [qwen35-0.8b-q8_0]=99
    [qwen35-9b-q4_k_m]=99
    [gemma4-12b-q4_k_xl]=99
    [qwen36-27b-q2_k]=99
)

declare -A WORKLOAD_P=(
    [pp32]=32
    [tg2]=0
    [pp512]=512
    [pp4096]=4096
    [pp32768]=32768
    [tg128]=0
    [tg1024]=0
)

declare -A WORKLOAD_N=(
    [pp32]=0
    [tg2]=2
    [pp512]=0
    [pp4096]=0
    [pp32768]=0
    [tg128]=128
    [tg1024]=1024
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
    git -C "$ROOT" diff --quiet "$EXPECTED_BRANCH" HEAD -- . \
        ':(exclude)benches/sm70-v100/comparisons/001-origin-master-e8f19cc0a' || \
        fail "source changes after the candidate revision would make its binary stale"
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
    [[ "$graphics" == "$LOCKED_GRAPHICS" ]] || fail "V100 graphics clock is $graphics MHz, expected $LOCKED_GRAPHICS MHz"
    [[ "$memory" == "$LOCKED_MEMORY" ]] || fail "V100 memory clock is $memory MHz, expected $LOCKED_MEMORY MHz"
}

wait_cool() {
    local samples=0
    local waited=0
    local state gpu_temp memory_temp util

    while (( samples < 3 )); do
        state=$(nvidia-smi -i "$GPU_UUID" \
            --query-gpu=temperature.gpu,temperature.memory,utilization.gpu \
            --format=csv,noheader,nounits) || fail "cannot query V100 temperature"
        IFS=, read -r gpu_temp memory_temp util <<< "$state"
        gpu_temp=$(trim "$gpu_temp")
        memory_temp=$(trim "$memory_temp")
        util=$(trim "$util")
        if (( gpu_temp <= COOLDOWN_TEMP && memory_temp <= COOLDOWN_TEMP && util == 0 )); then
            ((samples++))
        else
            samples=0
        fi
        if (( samples < 3 )); then
            if (( waited % 10 == 0 )); then
                echo "cooldown: GPU ${gpu_temp} C, HBM ${memory_temp} C, ${util}% GPU utilization"
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
    local workload=$4
    local count

    [[ -s "$path" ]] || return 1
    count=$(jq 'length' "$path" 2>/dev/null) || return 1
    [[ "$count" == 1 ]] || return 1
    jq -e --arg expected "$expected" \
        --argjson expected_ngl "$expected_ngl" \
        --argjson expected_p "${WORKLOAD_P[$workload]}" \
        --argjson expected_n "${WORKLOAD_N[$workload]}" \
        'all(.[];
            .build_commit == $expected and
            .n_gpu_layers == $expected_ngl and
            .n_prompt == $expected_p and
            .n_gen == $expected_n)' \
        "$path" >/dev/null
}

valid_telemetry() {
    local path=$1

    [[ -s "$path" ]] || return 1
    awk -F, -v graphics="$LOCKED_GRAPHICS" -v memory="$LOCKED_MEMORY" -v busy_util="$BUSY_UTIL" '
        NR > 1 {
            for (i = 2; i <= 11; ++i) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
            }
            if ($7 + 0 >= busy_util) {
                ++busy
                if ($2 + 0 != graphics || $3 + 0 != memory ||
                    $9 != "Not Active" || $10 != "Not Active" ||
                    $11 != "Not Active") {
                    ++invalid
                }
            }
        }
        END { exit !(busy > 0 && invalid == 0) }
    ' "$path"
}

run_one() {
    local round=$1
    local model_id=$2
    local workload=$3
    local label=$4
    local stem
    stem=$(printf 'r%02d-%s-%s-%s' "$round" "$model_id" "$workload" "$label")

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
        -p "${WORKLOAD_P[$workload]}"
        -n "${WORKLOAD_N[$workload]}"
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
        'timestamp,graphics_clock_mhz,memory_clock_mhz,power_w,gpu_temperature_c,memory_temperature_c,gpu_util_pct,memory_used_mib,sw_power_cap,sw_thermal_slowdown,hw_thermal_slowdown' \
        > "$telemetry"
    wait_cool
    nvidia-smi -i "$GPU_UUID" \
        --query-gpu=timestamp,clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,temperature.memory,utilization.gpu,memory.used,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown \
        --format=csv,noheader,nounits -lms 50 >> "$telemetry" &
    TELEMETRY_PID=$!

    echo "start: round=$round model=$model_id workload=$workload revision=$label ngl=$ngl"
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
    if ! valid_result "$tmp" "$expected" "$ngl" "$workload"; then
        rm -f "$tmp"
        fail "$stem produced invalid JSON or revision metadata"
    fi
    if ! valid_telemetry "$telemetry"; then
        rm -f "$tmp"
        fail "$stem departed from the locked clocks or reported a cap"
    fi
    mv "$tmp" "$json"

    echo "done: $stem in $((end - start)) s"
    jq -r '.[] | "  " + (if .n_prompt > 0 then "pp" + (.n_prompt|tostring) else "tg" + (.n_gen|tostring) end) + " = " + (.avg_ts|tostring) + " tok/s"' "$json"
}

pair_complete() {
    local round=$1
    local model_id=$2
    local workload=$3
    local origin branch
    origin=$(printf '%s/raw/r%02d-%s-%s-origin.json' "$OUT" "$round" "$model_id" "$workload")
    branch=$(printf '%s/raw/r%02d-%s-%s-branch.json' "$OUT" "$round" "$model_id" "$workload")
    local ngl=${MODEL_NGL[$model_id]}
    local origin_telemetry branch_telemetry
    origin_telemetry=$(printf '%s/telemetry/r%02d-%s-%s-origin.csv' "$OUT" "$round" "$model_id" "$workload")
    branch_telemetry=$(printf '%s/telemetry/r%02d-%s-%s-branch.csv' "$OUT" "$round" "$model_id" "$workload")
    valid_result "$origin" "${EXPECTED_COMMITS[origin]}" "$ngl" "$workload" && \
        valid_result "$branch" "${EXPECTED_COMMITS[branch]}" "$ngl" "$workload" && \
        valid_telemetry "$origin_telemetry" && \
        valid_telemetry "$branch_telemetry"
}

run_pair() {
    local round=$1
    local model_id=$2
    local workload=$3
    local -a order

    if pair_complete "$round" "$model_id" "$workload"; then
        echo "skip: completed round=$round model=$model_id workload=$workload"
        return
    fi

    local prefix
    prefix=$(printf 'r%02d-%s-%s-' "$round" "$model_id" "$workload")
    rm -f \
        "$OUT/raw/$prefix"*.json \
        "$OUT/raw/$prefix"*.json.tmp \
        "$OUT/logs/$prefix"*.log \
        "$OUT/telemetry/$prefix"*.csv \
        "$OUT/commands/$prefix"*.txt

    if (( round % 2 == 1 )); then
        order=(origin branch)
    else
        order=(branch origin)
    fi
    run_one "$round" "$model_id" "$workload" "${order[0]}"
    run_one "$round" "$model_id" "$workload" "${order[1]}"
}

validate_environment

echo "mode=$MODE rounds=$ROUNDS workloads=${WORKLOAD_IDS[*]}"
for ((round = 1; round <= ROUNDS; round++)); do
    for model_id in "${MODEL_IDS[@]}"; do
        for workload in "${WORKLOAD_IDS[@]}"; do
            run_pair "$round" "$model_id" "$workload"
        done
    done
done

echo "benchmark matrix complete: $OUT"

#!/bin/bash

### ─── USER CONFIG (edit these) ──────────────────────────────────────────

VLLM_PORT=8001
SITES="shopping"
MODELS=(
    "gpt-oss-20b"
)

# ─── Environment ───────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMMON_SH="$SCRIPT_DIR/common.sh"

if [ ! -f "$COMMON_SH" ]; then
    echo "ERROR: Shared shell helpers not found at $COMMON_SH"
    exit 1
fi

source "$COMMON_SH"

cd "$PROJECT_DIR"


# ── Paths to change for your setup ────────────────────────────────────
export AGENTLAB_EXP_ROOT="$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export VLLM_BIN="${VLLM_BIN:-$(conda run -n vllm which vllm 2>/dev/null || echo vllm)}"
PYTHON="${PYTHON:-$(conda run -n agentlab which python 2>/dev/null || echo python)}"
# ──────────────────────────────────────────────────────────────────────

export WA_SHOPPING="http://localhost:7770"
export WA_SHOPPING_ADMIN="http://localhost:7780/admin"
export WA_REDDIT="http://localhost:9999"
export WA_GITLAB="http://localhost:8023"
export WA_WIKIPEDIA="http://localhost:7770"
export WA_MAP="http://localhost:3030"
export WA_HOMEPAGE="http://localhost:4399"

export VLLM_API_URL="http://localhost:$VLLM_PORT/v1"
export VLLM_API_KEY="dummy"

RESULTS_DIR="$AGENTLAB_EXP_ROOT/results"
ARCHIVE_DIR="$RESULTS_DIR/archive"
VLLM_PID=""

# ─── Model configs ──────────────────────────────────────────────────────
declare -A MODEL_SERVE=(
    [gpt-oss-20b]="webarena/serve_gpt_oss_20b.sh"
    [gpt-oss-120b]="webarena/serve_gpt_oss_120b.sh"
    [agenttrek-32b]="webarena/serve_agenttrek_32b.sh"
    [bu-30b]="webarena/serve_bu_30b.sh"
)
declare -A MODEL_WAIT=(
    [gpt-oss-20b]=300
    [gpt-oss-120b]=600
    [agenttrek-32b]=300
    [bu-30b]=600
)

# ─── Helper functions ──────────────────────────────────────────────────

log() {
    echo ""
    echo "======================================"
    echo "  $1"
    echo "  $(date)"
    echo "======================================"
    echo ""
}

reset_dockers() {
    log "Resetting Docker containers"
    reset_webarena_verified_env "$PROJECT_DIR"
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID $VLLM_PID)..."
        kill -- -"$VLLM_PID" 2>/dev/null || kill "$VLLM_PID" 2>/dev/null || true
        for i in $(seq 1 15); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            kill -9 -- -"$VLLM_PID" 2>/dev/null || kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    local stray_pid
    stray_pid=$(lsof -ti :"$VLLM_PORT" 2>/dev/null || true)
    if [ -n "$stray_pid" ]; then
        kill -9 $stray_pid 2>/dev/null || true
    fi
    sleep 5
    echo "GPU memory after vLLM shutdown:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

trap stop_vllm EXIT

# ─── Run models ────────────────────────────────────────────────────────

TOTAL=${#MODELS[@]}
IDX=0

for MODEL in "${MODELS[@]}"; do
    IDX=$((IDX + 1))

    log "EXPERIMENT $IDX/$TOTAL: $MODEL"

    # Kick off Docker reset immediately so it can overlap with vLLM startup.
    reset_dockers &
    DOCKER_PID=$!

    # Log files: one for vLLM server, one for eval client
    VLLM_LOG="$RESULTS_DIR/${MODEL}_vllm.log"
    EVAL_LOG="$RESULTS_DIR/${MODEL}_eval.log"

    # Start vLLM while Docker is still resetting so the two waits overlap.
    if [[ -n "${MODEL_SERVE[$MODEL]:-}" ]]; then
        GPU=$(find_free_gpus 1)
        log "Starting vLLM for $MODEL on GPU $GPU"
        log "  vLLM log: $VLLM_LOG"
        log "  Eval log: $EVAL_LOG"
        bash "$SCRIPT_DIR/${MODEL_SERVE[$MODEL]}" "$GPU" "$VLLM_PORT" > "$VLLM_LOG" 2>&1 &
        VLLM_PID=$!
    else
        log "No vLLM needed for $MODEL (Azure/API model)"
        log "  Eval log: $EVAL_LOG"
        VLLM_PID=""
    fi

    # Both dependencies must be ready before eval starts.
    wait "$DOCKER_PID" || {
        echo "Docker reset FAILED"
        stop_vllm
        exit 1
    }
    if [[ -n "${MODEL_SERVE[$MODEL]:-}" ]]; then
        wait_for_vllm "$VLLM_PORT" "${MODEL_WAIT[$MODEL]}" || {
            echo "vLLM startup FAILED"
            exit 1
        }
    fi

    log "Running $MODEL"
    $PYTHON main_webarena_verified.py --model "$MODEL" --sites "$SITES" 2>&1 | tee "$EVAL_LOG"

    stop_vllm

    # Kill any leftover vLLM/Ray GPU processes from this model run
    echo "Cleaning up stale GPU processes..."
    for pid in $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null | grep -i "vllm\|ray" | cut -d',' -f1 | xargs); do
        echo "  Killing leftover GPU process: $pid"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5

    log "$MODEL COMPLETE"
done

log "ALL $TOTAL EXPERIMENTS COMPLETE"
echo "Results in: $RESULTS_DIR/"
ls -d "$RESULTS_DIR"/GenericAgent-* 2>/dev/null

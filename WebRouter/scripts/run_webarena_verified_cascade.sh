#!/bin/bash

### ===================================================================
###  USER CONFIG (edit these)
### ===================================================================

# ── Models ──
SMALL_MODEL_KEY="gpt-oss-20b"                      # key into VLLM_MODELS / MODEL_SERVE
LARGE_MODEL_KEY="azure-gpt-5-mini"                   # Azure model key

# ── BERT thresholds for switching ──
STUCK_THRESHOLD=0.1
MILESTONE_THRESHOLD=0.1

# ── Task range & sites ──
START=0
END=812
SITES="shopping"
N_JOBS=4

# ── vLLM ──
VLLM_PORT=8001

### ===================================================================
###  ENVIRONMENT PATHS (edit for your setup)
### ===================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMMON_SH="$SCRIPT_DIR/common.sh"

export AGENTLAB_EXP_ROOT="$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export VLLM_BIN="${VLLM_BIN:-$(conda run -n vllm which vllm 2>/dev/null || echo vllm)}"

PYTHON="${PYTHON:-$(conda run -n agentlab which python 2>/dev/null || echo python)}"
RESULTS_DIR="$PROJECT_DIR/results"

# WebArena URLs
export WA_SHOPPING="http://localhost:7770"
export WA_SHOPPING_ADMIN="http://localhost:7780/admin"
export WA_REDDIT="http://localhost:9999"
export WA_GITLAB="http://localhost:8023"
export WA_WIKIPEDIA="http://localhost:7770"
export WA_MAP="http://localhost:3030"
export WA_HOMEPAGE="http://localhost:4399"

export VLLM_API_URL="http://localhost:$VLLM_PORT/v1"
export VLLM_API_KEY="dummy"

### ===================================================================
###  DERIVED PARAMS (do not edit below this line)
### ===================================================================

if [ ! -f "$COMMON_SH" ]; then
    echo "ERROR: common.sh not found at $COMMON_SH"
    exit 1
fi
source "$COMMON_SH"
cd "$PROJECT_DIR"

# ── Model configs (mirrors main_webarena_verified.py + serve scripts) ──
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

# ── Resolve study directory names ──
# Run main_webarena_verified.py in dry-run mode to get the exact study dir name
get_study_dir() {
    local model_key="$1"
    $PYTHON -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('AGENTLAB_EXP_ROOT', '$PROJECT_DIR')
from main_webarena_verified import build_agent
from pathlib import Path

agent, suffix = build_agent('$model_key')
vllm_suffix = getattr(agent, '_vllm_params_suffix', '')
study_dir = f'{suffix}_webarena-verified{vllm_suffix}_${START}-${END}'
print(study_dir)
"
}

SMALL_STUDY_NAME="$(get_study_dir "$SMALL_MODEL_KEY")"
SMALL_STUDY_DIR="$RESULTS_DIR/$SMALL_STUDY_NAME"

# Large model: Azure models don't have vllm suffix
LARGE_STUDY_NAME="$(get_study_dir "$LARGE_MODEL_KEY")"
LARGE_STUDY_DIR="$RESULTS_DIR/$LARGE_STUDY_NAME"

# ── Data tag: compact pair name for BERT data/models ──
# e.g. "AgentTrek-32B+gpt-5-mini" or "gpt-oss-20b+gpt-5-mini"
SMALL_SHORT="${SMALL_MODEL_KEY}"
LARGE_SHORT="${LARGE_MODEL_KEY#azure-}"   # strip "azure-" prefix
DATA_TAG="${SMALL_SHORT}+${LARGE_SHORT}_webarena-verified_${START}-${END}"

DATA_DIR="$RESULTS_DIR/data/$DATA_TAG"
BERT_DIR="$RESULTS_DIR/berts/$DATA_TAG"
THRESHOLD_TAG="${STUCK_THRESHOLD}-${MILESTONE_THRESHOLD}"

# ── BERTSwitch study dir ──
SWITCH_STUDY_NAME="BERTSwitch-${THRESHOLD_TAG}-${SMALL_SHORT}+${LARGE_SHORT}_webarena-verified_${START}-${END}"
SWITCH_STUDY_DIR="$RESULTS_DIR/$SWITCH_STUDY_NAME"

# ── Resolve small model's full HF name and token limits for BERTSwitch ──
read -r SMALL_MODEL_NAME SMALL_MAX_TOTAL_TOKENS SMALL_MAX_NEW_TOKENS <<< "$($PYTHON -c "
from main_webarena_verified import VLLM_MODELS
cfg = VLLM_MODELS['$SMALL_MODEL_KEY']
print(cfg['model_name'], cfg['max_total_tokens'], cfg['max_new_tokens'])
")"

LARGE_MODEL_NAME="${LARGE_MODEL_KEY#azure-}"

VLLM_PID=""

### ===================================================================
###  HELPERS
### ===================================================================

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
}

cleanup_gpu() {
    echo "Cleaning up stale GPU processes..."
    for pid in $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null | grep -i "vllm\|ray" | cut -d',' -f1 | xargs); do
        echo "  Killing leftover GPU process: $pid"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
}

# Check if a study directory has completed (all tasks have summary_info.json)
study_is_complete() {
    local study_dir="$1"
    if [ ! -d "$study_dir" ]; then
        return 1
    fi
    local n_dirs n_done
    n_dirs=$(find "$study_dir" -maxdepth 1 -mindepth 1 -type d | wc -l)
    n_done=$(find "$study_dir" -maxdepth 2 -name "summary_info.json" | wc -l)
    if [ "$n_dirs" -eq 0 ]; then
        return 1
    fi
    # Consider complete if >95% have summaries (some may error)
    local pct=$((n_done * 100 / n_dirs))
    [ "$pct" -ge 95 ]
}

find_bert_gpu() {
    # Find a GPU with at least 4GB free VRAM for BERT training/inference
    $PYTHON -c "
import subprocess, sys
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
    capture_output=True, text=True
)
for line in result.stdout.strip().split('\n'):
    idx, free = line.split(',')
    if int(free.strip()) >= 4000:
        print(idx.strip())
        sys.exit(0)
print('0')  # fallback
"
}

trap 'stop_vllm' EXIT

### ===================================================================
###  PRINT PLAN
### ===================================================================

log "BERT CASCADE PIPELINE"
echo "Small model:     $SMALL_MODEL_KEY ($SMALL_MODEL_NAME)"
echo "Large model:     $LARGE_MODEL_KEY ($LARGE_MODEL_NAME)"
echo "Task range:      $START-$END"
echo "Sites:           $SITES"
echo "Thresholds:      stuck=$STUCK_THRESHOLD, milestone=$MILESTONE_THRESHOLD"
echo ""
echo "Directories:"
echo "  Small study:   $SMALL_STUDY_DIR"
echo "  Large study:   $LARGE_STUDY_DIR"
echo "  Data (annot):  $DATA_DIR"
echo "  BERTs:         $BERT_DIR"
echo "  BERTSwitch:    $SWITCH_STUDY_DIR"

### ===================================================================
###  STAGE 1: Run small model (vLLM-served)
### ===================================================================

if study_is_complete "$SMALL_STUDY_DIR"; then
    log "STAGE 1: SKIP - Small model results exist at $SMALL_STUDY_DIR"
else
    log "STAGE 1: Run small model ($SMALL_MODEL_KEY)"

    reset_dockers &
    DOCKER_PID=$!

    VLLM_LOG="$RESULTS_DIR/${SMALL_MODEL_KEY}_vllm.log"
    EVAL_LOG="$RESULTS_DIR/${SMALL_MODEL_KEY}_eval.log"

    if [[ -n "${MODEL_SERVE[$SMALL_MODEL_KEY]:-}" ]]; then
        GPU=$(find_free_gpus 1)
        log "Starting vLLM for $SMALL_MODEL_KEY on GPU $GPU"
        bash "$SCRIPT_DIR/${MODEL_SERVE[$SMALL_MODEL_KEY]}" "$GPU" "$VLLM_PORT" > "$VLLM_LOG" 2>&1 &
        VLLM_PID=$!
    fi

    wait "$DOCKER_PID" || { echo "Docker reset FAILED"; stop_vllm; exit 1; }

    if [[ -n "${MODEL_SERVE[$SMALL_MODEL_KEY]:-}" ]]; then
        wait_for_vllm "$VLLM_PORT" "${MODEL_WAIT[$SMALL_MODEL_KEY]}" || { echo "vLLM startup FAILED"; exit 1; }
    fi

    log "Running $SMALL_MODEL_KEY evaluation"
    $PYTHON main_webarena_verified.py --model "$SMALL_MODEL_KEY" --sites "$SITES" \
        --start "$START" --end "$END" --n-jobs "$N_JOBS" 2>&1 | tee "$EVAL_LOG"

    stop_vllm
    cleanup_gpu
    log "STAGE 1 COMPLETE: Small model ($SMALL_MODEL_KEY)"
fi

### ===================================================================
###  STAGE 2: Run large model (Azure API)
### ===================================================================

if study_is_complete "$LARGE_STUDY_DIR"; then
    log "STAGE 2: SKIP - Large model results exist at $LARGE_STUDY_DIR"
else
    log "STAGE 2: Run large model ($LARGE_MODEL_KEY)"

    reset_dockers

    EVAL_LOG="$RESULTS_DIR/${LARGE_MODEL_KEY}_eval.log"

    $PYTHON main_webarena_verified.py --model "$LARGE_MODEL_KEY" --sites "$SITES" \
        --start "$START" --end "$END" --n-jobs "$N_JOBS" 2>&1 | tee "$EVAL_LOG"

    log "STAGE 2 COMPLETE: Large model ($LARGE_MODEL_KEY)"
fi

### ===================================================================
###  STAGE 3: GPT annotation (stuck + milestone) on small model trajectories
### ===================================================================

if [ -d "$DATA_DIR/stuck_analysis" ] && [ -d "$DATA_DIR/milestone_analysis" ] && \
   [ "$(find "$DATA_DIR/stuck_analysis" -name '*.json' | wc -l)" -gt 100 ]; then
    log "STAGE 3: SKIP - Annotations exist at $DATA_DIR"
else
    log "STAGE 3: GPT annotation on $SMALL_MODEL_KEY trajectories"

    mkdir -p "$DATA_DIR/stuck_analysis" "$DATA_DIR/milestone_analysis"

    echo "-- Stuck analysis --"
    $PYTHON scripts/bert/annotate_trajectories.py \
        --results-dir "$SMALL_STUDY_DIR" \
        --analysis-type stuck \
        --output-dir "$DATA_DIR/stuck_analysis" \
        --all --skip-existing

    echo ""
    echo "-- Milestone analysis --"
    $PYTHON scripts/bert/annotate_trajectories.py \
        --results-dir "$SMALL_STUDY_DIR" \
        --analysis-type milestones \
        --output-dir "$DATA_DIR/milestone_analysis" \
        --all --skip-existing

    log "STAGE 3 COMPLETE: Annotations saved to $DATA_DIR"
fi

### ===================================================================
###  STAGE 4: Build training datasets
### ===================================================================

if [ -f "$DATA_DIR/stuck_training_dataset.json" ] && [ -f "$DATA_DIR/milestone_training_dataset.json" ]; then
    log "STAGE 4: SKIP - Training datasets exist at $DATA_DIR"
else
    log "STAGE 4: Build training datasets"

    echo "-- Stuck dataset --"
    $PYTHON scripts/bert/build_stuck_dataset.py \
        --results-dir "$SMALL_STUDY_DIR" \
        --analysis-dir "$DATA_DIR/stuck_analysis" \
        --split-mode task \
        --output "$DATA_DIR/stuck_training_dataset.json" \
        --output-stats "$DATA_DIR/stuck_training_dataset_stats.json"

    echo ""
    echo "-- Milestone dataset --"
    $PYTHON scripts/bert/build_milestone_dataset.py \
        --results-dir "$SMALL_STUDY_DIR" \
        --analysis-dir "$DATA_DIR/milestone_analysis" \
        --split-mode task \
        --output "$DATA_DIR/milestone_training_dataset.json" \
        --output-stats "$DATA_DIR/milestone_training_dataset_stats.json"

    log "STAGE 4 COMPLETE: Datasets saved to $DATA_DIR"
fi

### ===================================================================
###  STAGE 5: Train BERTs
### ===================================================================

if [ -f "$BERT_DIR/stuck-detector/config.json" ] && [ -f "$BERT_DIR/milestone-detector/config.json" ]; then
    log "STAGE 5: SKIP - Trained BERTs exist at $BERT_DIR"
else
    log "STAGE 5: Train stuck + milestone BERTs"

    BERT_GPU="$(find_bert_gpu)"
    echo "Using GPU $BERT_GPU for BERT training"
    export CUDA_VISIBLE_DEVICES="$BERT_GPU"

    mkdir -p "$BERT_DIR"

    echo "-- Train stuck detector --"
    $PYTHON scripts/bert/train_router.py \
        --dataset "$DATA_DIR/stuck_training_dataset.json" \
        --output-dir "$BERT_DIR/stuck-detector" \
        --max-length 1024 \
        --num-epochs 5 \
        --learning-rate 2e-5 \
        --train-batch-size 8 \
        --eval-batch-size 16 \
        --use-class-weights \
        --weight-strategy balanced

    echo ""
    echo "-- Train milestone detector --"
    $PYTHON scripts/bert/train_router.py \
        --dataset "$DATA_DIR/milestone_training_dataset.json" \
        --output-dir "$BERT_DIR/milestone-detector" \
        --max-length 1024 \
        --num-epochs 5 \
        --learning-rate 2e-5 \
        --train-batch-size 8 \
        --eval-batch-size 16 \
        --use-class-weights \
        --weight-strategy balanced

    unset CUDA_VISIBLE_DEVICES
    log "STAGE 5 COMPLETE: BERTs saved to $BERT_DIR"
fi

### ===================================================================
###  STAGE 6: BERTSwitch inference
### ===================================================================

if study_is_complete "$SWITCH_STUDY_DIR"; then
    log "STAGE 6: SKIP - BERTSwitch results exist at $SWITCH_STUDY_DIR"
else
    log "STAGE 6: BERTSwitch inference ($SMALL_MODEL_KEY + $LARGE_MODEL_KEY)"

    reset_dockers &
    DOCKER_PID=$!

    VLLM_LOG="$RESULTS_DIR/${SWITCH_STUDY_NAME}_vllm.log"
    EVAL_LOG="$RESULTS_DIR/${SWITCH_STUDY_NAME}_eval.log"

    if [[ -n "${MODEL_SERVE[$SMALL_MODEL_KEY]:-}" ]]; then
        GPU=$(find_free_gpus 1)
        log "Starting vLLM for $SMALL_MODEL_KEY on GPU $GPU"
        bash "$SCRIPT_DIR/${MODEL_SERVE[$SMALL_MODEL_KEY]}" "$GPU" "$VLLM_PORT" > "$VLLM_LOG" 2>&1 &
        VLLM_PID=$!
    fi

    wait "$DOCKER_PID" || { echo "Docker reset FAILED"; stop_vllm; exit 1; }

    if [[ -n "${MODEL_SERVE[$SMALL_MODEL_KEY]:-}" ]]; then
        wait_for_vllm "$VLLM_PORT" "${MODEL_WAIT[$SMALL_MODEL_KEY]}" || { echo "vLLM startup FAILED"; exit 1; }
    fi

    log "Running BERTSwitch evaluation"
    $PYTHON scripts/bert/run_with_routing.py \
        --mode builtin \
        --stuck-bert-dir "$BERT_DIR/stuck-detector" \
        --milestone-bert-dir "$BERT_DIR/milestone-detector" \
        --sites "$SITES" \
        --n-jobs "$N_JOBS" \
        --start "$START" \
        --end "$END" \
        --backend ray \
        --small-model-name "$SMALL_MODEL_NAME" \
        --small-model-url "http://localhost:$VLLM_PORT/v1" \
        --small-max-total-tokens "$SMALL_MAX_TOTAL_TOKENS" \
        --small-max-new-tokens "$SMALL_MAX_NEW_TOKENS" \
        --large-model-name "$LARGE_MODEL_NAME" \
        --stuck-threshold "$STUCK_THRESHOLD" \
        --milestone-threshold "$MILESTONE_THRESHOLD" \
        --study-dir "$SWITCH_STUDY_NAME" \
        2>&1 | tee "$EVAL_LOG"

    stop_vllm
    cleanup_gpu
    log "STAGE 6 COMPLETE: BERTSwitch results at $SWITCH_STUDY_DIR"
fi

### ===================================================================
###  DONE
### ===================================================================

log "ALL STAGES COMPLETE"
echo "Results:"
echo "  Small model:  $SMALL_STUDY_DIR"
echo "  Large model:  $LARGE_STUDY_DIR"
echo "  Annotations:  $DATA_DIR"
echo "  BERTs:        $BERT_DIR"
echo "  BERTSwitch:   $SWITCH_STUDY_DIR"

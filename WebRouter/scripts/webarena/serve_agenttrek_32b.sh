#!/bin/bash
# Serve xlangai/AgentTrek-1.0-32B using vLLM. Single GPU.

GPU=${1:-4}
PORT=${2:-8001}
GPU_MEMORY_UTILIZATION=${3:-${GPU_MEMORY_UTILIZATION:-0.95}}
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_PYTHON="${VLLM_BIN%/*}/python"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

echo "Serving xlangai/AgentTrek-1.0-32B on GPU $GPU, port $PORT"
echo "GPU memory utilization target: $GPU_MEMORY_UTILIZATION"
echo "Using vLLM from: $VLLM_BIN"
$VLLM_PYTHON -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

CUDA_VISIBLE_DEVICES=$GPU $VLLM_BIN serve xlangai/AgentTrek-1.0-32B \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 10}' \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --port $PORT

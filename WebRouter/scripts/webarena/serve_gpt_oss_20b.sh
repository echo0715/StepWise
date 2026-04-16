#!/bin/bash
# Serve gpt-oss-20b using vLLM. Single GPU.

GPU=${1:-3}
PORT=${2:-8001}
GPU_MEMORY_UTILIZATION=${3:-${GPU_MEMORY_UTILIZATION:-0.95}}
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_PYTHON="${VLLM_BIN%/*}/python"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

echo "Serving openai/gpt-oss-20b on GPU $GPU, port $PORT"
echo "GPU memory utilization target: $GPU_MEMORY_UTILIZATION"
echo "Using vLLM from: $VLLM_BIN"
$VLLM_PYTHON -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

CUDA_VISIBLE_DEVICES=$GPU $VLLM_BIN serve openai/gpt-oss-20b \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --limit-mm-per-prompt '{"image": 10}' \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --port $PORT

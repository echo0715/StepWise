#!/bin/bash
# Serve gpt-oss-120b using vLLM. Single GPU (mxfp4 quantized).

GPU=${1:-4}
TP_SIZE=1
PORT=${2:-8001}
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_PYTHON="${VLLM_BIN%/*}/python"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"

echo "Serving openai/gpt-oss-120b on GPU $GPU (mxfp4, TP=$TP_SIZE), port $PORT"
echo "Using vLLM from: $VLLM_BIN"
$VLLM_PYTHON -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

CUDA_VISIBLE_DEVICES=$GPU $VLLM_BIN serve openai/gpt-oss-120b \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len 131072 \
    --limit-mm-per-prompt '{"image": 10}' \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --port $PORT

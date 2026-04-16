# WebRouter

## Setup

### Background
This repo is built on https://github.com/ServiceNow/AgentLab

### Installation

#### `agentlab` env — running the benchmark suite and all BERT scripts
```bash
conda create -n agentlab python=3.12 -y
conda activate agentlab
pip install -e ".[transformers]"
pip install --upgrade playwright && playwright install

# Webarena-verified
mkdir -p third_party && cd third_party
pip install git+https://github.com/ServiceNow/webarena-verified
pip install browsergym-webarena-verified
pip install --upgrade playwright && playwright install

# Webarena docker containers
cd third_party/webarena-verified
docker compose up -d
```

#### `vllm` env — serving open-source models
```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
```

#### API keys (for Azure models only)
```bash
# Set API key
export AZURE_OPENAI_API_KEY=<your-key>
# Set endpoint URL
export AZURE_OPENAI_ENDPOINT=<your-endpoint>
```

### Environment paths
All scripts read paths from a single block at the top. Change these to match your setup:
```
export AGENTLAB_EXP_ROOT=/path/to/AgentLab               # repo root
export HF_HOME=/path/to/cache/huggingface                 # HuggingFace cache
export VLLM_CACHE_ROOT=/path/to/cache/vllm                # vLLM cache
export XDG_CACHE_HOME=/path/to/cache                      # general cache
export PLAYWRIGHT_BROWSERS_PATH=/path/to/cache/ms-playwright
export UV_CACHE_DIR=/path/to/cache/uv
export VLLM_BIN=$(conda run -n vllm which vllm)           # vLLM binary
PYTHON=$(conda run -n agentlab which python)               # agentlab python
```
Edit this block in whichever script you run:
- `scripts/run_webarena_verified.sh` — benchmark 1 model
- `scripts/run_webarena_verified_cascade.sh` — full cascade pipeline

The vLLM serve scripts (`scripts/webarena/serve_*.sh`) inherit `VLLM_BIN`, `HF_HOME`, `VLLM_CACHE_ROOT`, `XDG_CACHE_HOME` from the parent script automatically.

## Running

### Single model evaluation
Run k models sequentially on webarena-verified (edit MODELS array in script):
```
bash scripts/run_webarena_verified.sh
```

### BERT cascade pipeline

One script runs the full cascade: baseline evals → GPT annotation → BERT training → BERTSwitch inference.
```
bash scripts/run_webarena_verified_cascade.sh
```

Edit the user config block at the top:
```bash
SMALL_MODEL_KEY="gpt-oss-20b"       # vLLM-served model (key from main_webarena_verified.py)
LARGE_MODEL_KEY="azure-gpt-5-mini"    # Azure API model
STUCK_THRESHOLD=0.1                   # BERT switching thresholds
MILESTONE_THRESHOLD=0.1
START=0; END=812                      # task range
SITES="shopping"                      # task
```

#### Stages
1. **Run small model** — finds free GPU, starts vLLM, runs WebArena eval
2. **Run large model** — Azure API, no GPU needed
3. **GPT annotation** — gpt-5-mini labels each step as stuck/milestone on small model trajectories
4. **Build datasets** — converts annotations to BERT training format
5. **Train BERTs** — stuck-detector + milestone-detector
6. **BERTSwitch inference** — starts vLLM, runs switching agent

Each stage **skips if outputs already exist**. Delete the output dir to force re-run.

#### Output directories
```
results/data/<small>+<large>_webarena-verified_<start>-<end>/     # annotations + datasets
results/berts/<small>+<large>_webarena-verified_<start>-<end>/    # trained BERTs
results/BERTSwitch-<thresholds>-<small>+<large>_webarena-verified_<start>-<end>/  # switch results
```

#### Running individual stages

To re-run a specific stage (e.g. retrain BERTs with different params, or run switching with different thresholds):

**GPT annotation**
```bash
# Annotate stuck steps
python scripts/bert/annotate_trajectories.py \
    --results-dir results/<small-model-study-dir> \
    --analysis-type stuck \
    --output-dir results/data/<tag>/stuck_analysis \
    --all --skip-existing

# Annotate milestone steps
python scripts/bert/annotate_trajectories.py \
    --results-dir results/<small-model-study-dir> \
    --analysis-type milestones \
    --output-dir results/data/<tag>/milestone_analysis \
    --all --skip-existing
```

**Build datasets**
```bash
# Build stuck training dataset
python scripts/bert/build_stuck_dataset.py \
    --results-dir results/<small-model-study-dir> \
    --analysis-dir results/data/<tag>/stuck_analysis \
    --output results/data/<tag>/stuck_training_dataset.json

# Build milestone training dataset
python scripts/bert/build_milestone_dataset.py \
    --results-dir results/<small-model-study-dir> \
    --analysis-dir results/data/<tag>/milestone_analysis \
    --output results/data/<tag>/milestone_training_dataset.json
```

**Train BERTs**
```bash
# Train stuck-detector
CUDA_VISIBLE_DEVICES=0 python scripts/bert/train_router.py \
    --dataset results/data/<tag>/stuck_training_dataset.json \
    --output-dir results/berts/<tag>/stuck-detector \
    --max-length 1024 --num-epochs 5 --learning-rate 2e-5 \
    --train-batch-size 8 --eval-batch-size 16 \
    --use-class-weights --weight-strategy balanced

# Train milestone-detector
CUDA_VISIBLE_DEVICES=0 python scripts/bert/train_router.py \
    --dataset results/data/<tag>/milestone_training_dataset.json \
    --output-dir results/berts/<tag>/milestone-detector \
    --max-length 1024 --num-epochs 5 --learning-rate 2e-5 \
    --train-batch-size 8 --eval-batch-size 16 \
    --use-class-weights --weight-strategy balanced
```

**BERTSwitch inference**
```bash
python scripts/bert/run_with_routing.py --mode builtin \
    --stuck-bert-dir results/berts/<tag>/stuck-detector \
    --milestone-bert-dir results/berts/<tag>/milestone-detector \
    --small-model-name xlangai/AgentTrek-1.0-32B \
    --small-model-url http://localhost:8001/v1 \
    --small-max-total-tokens 32768 --small-max-new-tokens 512 \
    --large-model-name gpt-5-mini \
    --stuck-threshold 0.1 --milestone-threshold 0.1 \
    --sites shopping \
    --start 0 --end 812 --n-jobs 4
```

## Results
`scripts/summary.py` generates `results/metrics.txt` that summarizes all runs concisely.

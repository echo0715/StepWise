# Computer Router

A cascade agent for [OSWorld](https://github.com/xlang-ai/OSWorld): a small GUI agent runs the trajectory and a ModernBERT classifier decides when to hand off to a larger model (stuck detection + milestone verification).

Built on top of the OSWorld benchmark — see `OSWORLD_UPSTREAM_README.md` for upstream setup (AWS / Docker / VMware providers, task data, etc.).

## Setup

### `osworld` env — running the benchmark and hybrid agents
```bash
conda create -n osworld python=3.10 -y
conda activate osworld
pip install -r requirements.txt
```

### `bert` env — training / inferencing ModernBERT detectors
```bash
conda create -n bert python=3.10 -y
conda activate bert
pip install "torch==2.4.1" tensorboard scikit-learn datasets==3.1.0 accelerate==1.2.1 hf-transfer==0.1.8 huggingface_hub
pip install "git+https://github.com/huggingface/transformers.git@6e0515e99c39444caae39472ee1b2fd76ece32f1"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### AWS / providers
Follow `PUBLIC_EVALUATION_GUIDELINE.md` for AWS credentials and the host/client setup. Minimum env:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=...
export AWS_SECURITY_GROUP_ID=...
export AWS_SUBNET_ID=...
```

### Model API keys (depending on which runner you use)
```bash
export OPENAI_API_KEY=...           # Azure / self-hosted vLLM endpoint
export OPENAI_BASE_URL=...
export ANTHROPIC_API_KEY=...        # Claude + MilestoneJudge
export KIMI_API_KEY=...              # Kimi K2.5
```

## Running

### Single-model baseline
Each baseline runner corresponds to a small- or large-model slot in the hybrid cascade:

```bash
# EvoCUA (small model for run_multienv_hybrid.py)
python run_multienv_evocua.py --headless --provider_name aws \
    --evocua_model EvoCUA-S2 \
    --result_dir results/evocua_results \
    --test_all_meta_path evaluation_examples/test_nogdrive.json \
    --max_steps 50 --num_envs 10

# Qwen3-VL (small model for run_multienv_hybrid_qwen3_kimi.py)
python run_multienv_qwen3vl.py --headless --provider_name aws \
    --qwen3_model qwen3-vl-8b --qwen3_enable_thinking \
    --result_dir results/qwen3_results ...

# Claude (large model for run_multienv_hybrid.py)
python run_multienv_claude.py --headless --provider_name aws \
    --model claude-sonnet-4-5-20250929 \
    --result_dir results/claude_results ...

# Kimi K2.5 (large model for run_multienv_hybrid_kimi*.py)
python run_multienv_kimi_k25.py --headless --provider_name aws \
    --kimi_model kimi-k2.5 --kimi_thinking \
    --result_dir results/kimi_results ...
```

### Hybrid cascade (small → BERT → large)

Runs a small model, watches for stuck / failed-milestone signals with ModernBERT, and switches to a large model when triggered.

```bash
# EvoCUA → Claude
python run_multienv_hybrid.py --headless --provider_name aws \
    --evocua_model EvoCUA-S2 \
    --claude_model claude-sonnet-4-5-20250929 \
    --stuck_detector_path /path/to/modernbert-stuck-detector \
    --stuck_threshold 0.5 --min_steps_to_check 3 \
    --result_dir results/hybrid_results \
    --test_all_meta_path evaluation_examples/test_nogdrive.json \
    --max_steps 50 --num_envs 10

# Qwen3-VL → Kimi K2.5 (with milestone verification)
python run_multienv_hybrid_qwen3_kimi.py --headless --provider_name aws \
    --qwen3_model qwen3-vl-8b --qwen3_enable_thinking \
    --kimi_model kimi-k2.5 --kimi_thinking \
    --stuck_detector_path /path/to/modernbert-stuck-detector \
    --milestone_detector_path /path/to/modernbert-milestone-detector \
    --result_dir results/hybrid_qwen3_kimi_results ...
```

Other hybrid variants: `run_multienv_hybrid_kimi.py`, `run_multienv_hybrid_kimi_bounce.py`, `run_multienv_periodic_verify.py`.

## BERT training pipeline

Scripts live under `bert/`. The pipeline:

1. **Run a small model** to produce trajectories in `results/<study>/`.
2. **Annotate trajectories** with GPT (stuck / milestone labels).
3. **Build training datasets** from the annotations.
4. **Fine-tune ModernBERT** on each dataset.
5. **Plug the checkpoints** into a hybrid runner via `--stuck_detector_path` / `--milestone_detector_path`.

### 1. Annotate
```bash
# Stuck analysis (single run)
python bert/analyze_trajectories.py \
    --base-dir results/<study> \
    --analysis-type stuck \
    --output-dir results/<tag>/stuck_analysis \
    --all

# Milestone analysis (multiple runs for overlap voting)
python bert/analyze_trajectories.py \
    --base-dir results/<study> \
    --analysis-type milestones \
    --output-dir results/<tag>/milestone_analysis \
    --all --num-runs 4
```
Stuck analysis writes per-task JSONs to `results/<tag>/stuck_analysis/`.
Milestone analysis with `--num-runs 4` writes to `results/<tag>/milestone_analysis_run1/` … `_run4/`.

### 2. Build datasets
```bash
# Stuck (uses trajectory jsonl + stuck annotations)
python bert/build_stuck_dataset.py \
    --analysis-dir results/<tag>/stuck_analysis \
    --output results/stuck_training_dataset.json

# Milestone (aggregates per-task milestone_steps across N runs)
python bert/build_milestone_dataset.py \
    --milestone-dir results/<tag>/milestone_analysis \
    --num-runs 4 \
    --examples-dir evaluation_examples/examples \
    --output results/milestone_training_dataset.json
```
Outputs: `results/stuck_training_dataset.json`, `results/milestone_training_dataset.json`.

The milestone builder reads `<milestone-dir>_run1` … `<milestone-dir>_runN` and keeps steps
that appear in at least `--min-overlap` runs (default: `ceil(N * 0.75)`). With `--num-runs 1`
it reads the directory directly (no `_run1` suffix).

### 3. Fine-tune ModernBERT
```bash
# Stuck-detector
CUDA_VISIBLE_DEVICES=0 python bert/modernbert_finetune.py \
    --task stuck \
    --dataset_path results/stuck_training_dataset.json \
    --output_dir /path/to/modernbert-stuck-detector

# Milestone-detector
CUDA_VISIBLE_DEVICES=0 python bert/modernbert_finetune.py \
    --task milestone \
    --dataset_path results/milestone_training_dataset.json \
    --output_dir /path/to/modernbert-milestone-detector
```

## Project structure

```
ComputerRouter/
├── run_multienv_evocua.py          # small-model baseline (EvoCUA)
├── run_multienv_qwen3vl.py         # small-model baseline (Qwen3-VL)
├── run_multienv_claude.py          # large-model baseline (Claude)
├── run_multienv_kimi_k25.py        # large-model baseline (Kimi K2.5)
├── run_multienv_hybrid.py          # EvoCUA → Claude cascade
├── run_multienv_hybrid_kimi.py     # EvoCUA → Kimi cascade
├── run_multienv_hybrid_kimi_bounce.py
├── run_multienv_hybrid_qwen3_kimi.py   # Qwen3-VL → Kimi cascade
├── run_multienv_periodic_verify.py # periodic-verification variant
├── lib_run_single.py               # shared per-task execution loop
├── lib_results_logger.py
├── mm_agents/                  # agent implementations
│   ├── stuck_detector.py       # ModernBERT stuck classifier wrapper
│   ├── milestone_detector.py   # ModernBERT milestone classifier + judge
│   └── <agent>/                # one subdir per supported model family
├── desktop_env/                # OSWorld VM / AWS / Docker provider (upstream)
├── evaluation_examples/        # OSWorld task definitions (upstream)
├── bert/                       # BERT training + trajectory analysis scripts
│   ├── analyze_trajectories.py
│   ├── build_stuck_dataset.py
│   ├── build_milestone_dataset.py
│   └── modernbert_finetune.py
├── results/                    # all run outputs, datasets, summaries (gitignored)
├── summarize_results.py        # aggregate per-model success rates
└── show_result.py
```

## Results

`summarize_results.py` scans `results/<study-dir>/` and prints per-domain success rates:
```bash
python summarize_results.py --results-dir results/hybrid_qwen3_kimi_results
```

All run outputs land in `results/` (gitignored). Summary JSONs (`results_summary_*.json`, `qwen3_milestone_summary_*.json`, `trajectory_analysis.pdf`) are kept in git for reporting.

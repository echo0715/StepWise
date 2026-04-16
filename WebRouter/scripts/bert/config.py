"""
Shared configuration for BERT model routing pipeline.

This pipeline trains a ModernBERT classifier to route WebArena-verified tasks
Trains stuck/milestone BERTs and routes between gpt-oss-20b and gpt-5-mini.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

# ── Result directories for each model ─────────────────────────────────────────
# gpt-oss-20b: current run (647/678 completed, avg_reward=0.255)
OSS20B_RESULTS_DIR = (
    RESULTS_ROOT
    / "GenericAgent-openai_gpt-oss-20b_webarena-verified_1.0-1.0-None-4096-131072_0-812"
)

# gpt-5-mini: resp-max-out archive (562 completed, avg_reward=0.338)
GPT5MINI_RESULTS_DIR = (
    RESULTS_ROOT / "archive" / "resp-max-out"
    / "GenericAgent-gpt-5-mini_webarena-verified_0-812"
)

# gpt-5-mini fallback: initial archive (807 completed, avg_reward=0.324)
GPT5MINI_RESULTS_DIR_FALLBACK = (
    RESULTS_ROOT / "archive" / "initial"
    / "GenericAgent-gpt-5-mini_webarena-verified_0-812"
)

# ── Output directories ────────────────────────────────────────────────────────
# Shared data (routing, paired results) → results/data/
DATA_OUTPUT_ROOT = RESULTS_ROOT / "data"
PAIRED_RESULTS_CSV = DATA_OUTPUT_ROOT / "paired_results.csv"
ROUTING_DATASET_JSON = DATA_OUTPUT_ROOT / "routing_dataset.json"
ROUTING_DATASET_STATS_JSON = DATA_OUTPUT_ROOT / "routing_dataset_stats.json"

# Model artifacts (trained model, predictions, eval) → scripts/bert/output/
BERT_OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "bert" / "output"
TRAINED_MODEL_DIR = BERT_OUTPUT_ROOT / "modernbert-router"
PREDICTIONS_JSON = BERT_OUTPUT_ROOT / "routing_predictions.json"
EVAL_RESULTS_JSON = BERT_OUTPUT_ROOT / "eval_results.json"

# ── Stuck + Milestone analysis and datasets ──────────────────────────────────
def get_run_data_dir(results_dir: str | Path) -> Path:
    """Return the run-scoped data directory under results/data/."""
    return DATA_OUTPUT_ROOT / Path(results_dir).name


def get_stuck_analysis_dir(results_dir: str | Path) -> Path:
    """Return the default stuck-analysis output directory for a run."""
    return get_run_data_dir(results_dir) / "stuck_analysis"


def get_milestone_analysis_dir(results_dir: str | Path) -> Path:
    """Return the default milestone-analysis output directory for a run."""
    return get_run_data_dir(results_dir) / "milestone_analysis"


def get_stuck_dataset_json(results_dir: str | Path) -> Path:
    """Return the default stuck dataset path for a run."""
    return get_run_data_dir(results_dir) / "stuck_training_dataset.json"


def get_stuck_dataset_stats_json(results_dir: str | Path) -> Path:
    """Return the default stuck dataset stats path for a run."""
    return get_run_data_dir(results_dir) / "stuck_training_dataset_stats.json"


def get_milestone_dataset_json(results_dir: str | Path) -> Path:
    """Return the default milestone dataset path for a run."""
    return get_run_data_dir(results_dir) / "milestone_training_dataset.json"


def get_milestone_dataset_stats_json(results_dir: str | Path) -> Path:
    """Return the default milestone dataset stats path for a run."""
    return get_run_data_dir(results_dir) / "milestone_training_dataset_stats.json"


STUCK_ANALYSIS_DIR = get_stuck_analysis_dir(OSS20B_RESULTS_DIR)
MILESTONE_ANALYSIS_DIR = get_milestone_analysis_dir(OSS20B_RESULTS_DIR)
STUCK_DATASET_JSON = get_stuck_dataset_json(OSS20B_RESULTS_DIR)
STUCK_DATASET_STATS_JSON = get_stuck_dataset_stats_json(OSS20B_RESULTS_DIR)
MILESTONE_DATASET_JSON = get_milestone_dataset_json(OSS20B_RESULTS_DIR)
MILESTONE_DATASET_STATS_JSON = get_milestone_dataset_stats_json(OSS20B_RESULTS_DIR)
STUCK_MODEL_DIR = BERT_OUTPUT_ROOT / "modernbert-stuck-detector"
MILESTONE_MODEL_DIR = BERT_OUTPUT_ROOT / "modernbert-milestone-detector"

# ── Model routing labels ─────────────────────────────────────────────────────
LABEL_OSS20B = 0  # Prefer gpt-oss-20b
LABEL_GPT5MINI = 1  # Prefer gpt-5-mini
LABEL_NAMES = {LABEL_OSS20B: "gpt-oss-20b", LABEL_GPT5MINI: "gpt-5-mini"}

# ── Training hyperparameters (defaults) ──────────────────────────────────────
DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"
DEFAULT_MAX_LENGTH = 4096  # ModernBERT supports up to 8192; 4096 covers all samples
DEFAULT_NUM_EPOCHS = 5
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_WEIGHT_STRATEGY = "balanced"  # or "sqrt"

# ── WebArena environment URLs (for data collection) ──────────────────────────
WEBARENA_URLS = {
    "shopping": "http://localhost:7770",
    "shopping_admin": "http://localhost:7780/admin",
    "reddit": "http://localhost:9999",
    "gitlab": "http://localhost:8023",
}

# ── Model serving ────────────────────────────────────────────────────────────
VLLM_PORT = 8001
VLLM_API_URL = f"http://localhost:{VLLM_PORT}/v1"

# ── Utility ──────────────────────────────────────────────────────────────────

def get_task_name_from_dir(dirname: str) -> str | None:
    """Extract task_name from experiment directory name.

    Directory format: {timestamp}_{agent}_on_{task_name}_{seed}
    Task name format: webarena_verified.{template_id}.{task_id}.{revision}
    """
    # Find the "webarena_verified" part
    idx = dirname.find("webarena_verified.")
    if idx == -1:
        return None
    # Extract from webarena_verified. to the last underscore (which is the seed)
    remainder = dirname[idx:]
    # remainder looks like: webarena_verified.279.0.2_5
    # Split off the seed (last _N)
    parts = remainder.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return remainder


def ensure_output_dirs(results_dir: str | Path | None = None):
    """Create output directories if they don't exist."""
    DATA_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if results_dir is not None:
        get_run_data_dir(results_dir).mkdir(parents=True, exist_ok=True)
    BERT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

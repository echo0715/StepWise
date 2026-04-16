#!/usr/bin/env python3
"""Build a BERT training dataset for stuck detection from GPT analysis results.

Reads GPT-generated stuck analysis JSONs (one per task) and corresponding WebArena
experiment trajectories. For each non-terminal step, creates a training sample with
ONLY action + think context (NO goal/task description) and a binary label indicating
whether the agent is stuck at that step.

Train/test split is by TASK (not by step) to avoid data leakage.

Usage:
    python build_stuck_dataset.py
    python build_stuck_dataset.py --analysis-dir /path/to/stuck_analysis/
    python build_stuck_dataset.py --context-window 3 --test-fraction 0.3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import *  # noqa: E402, F401, F403

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ANALYSIS_DIR = get_stuck_analysis_dir(OSS20B_RESULTS_DIR)
DEFAULT_OUTPUT = get_stuck_dataset_json(OSS20B_RESULTS_DIR)
DEFAULT_OUTPUT_STATS = get_stuck_dataset_stats_json(OSS20B_RESULTS_DIR)
DEFAULT_CONTEXT_WINDOW = 5
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_SEED = 42
DEFAULT_SPLIT_MODE = "task"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_exp_dir(results_dir: Path, task_name: str) -> Path | None:
    """Find the experiment directory for a given task name.

    Iterates over subdirectories in results_dir and matches on the extracted
    task name. Returns the first match that has a summary_info.json file.

    Args:
        results_dir: Path to the gpt-oss-20b experiment results root.
        task_name: Task name to match (e.g. "webarena_verified.279.0.2").

    Returns:
        Path to the experiment directory, or None if not found.
    """
    for d in results_dir.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        tn = get_task_name_from_dir(d.name)
        if tn == task_name and (d / "summary_info.json").exists():
            return d
    return None


def load_steps(exp_dir: Path) -> list[dict]:
    """Load all step JSON files from an experiment directory.

    Returns a sorted list of step dicts, excluding the terminal step
    (where action is None).

    Args:
        exp_dir: Path to the experiment directory containing step_N.json files.

    Returns:
        Sorted list of step dictionaries.
    """
    steps = []
    for f in sorted(exp_dir.glob("step_*.json")):
        try:
            step_data = json.loads(f.read_text())
        except Exception:
            continue
        # Skip the terminal observation step (action=null)
        if step_data.get("action") is None:
            continue
        steps.append(step_data)
    steps.sort(key=lambda s: s.get("step", 0))
    return steps


def build_step_text(steps: list[dict], current_idx: int, context_window: int = 5) -> str:
    """Build BERT input text from a sliding window of action + think context.

    IMPORTANT: The stuck BERT does NOT use the user query/goal. Only step
    actions and thinking are included.

    Format:
        Step 3: [click('1502')] I need to inspect the order details...
        Step 4: [click('1618')] Looking at the order list...
        Step 5: [click('1502')] I need to inspect the order details...

    Args:
        steps: All step dicts for this task.
        current_idx: Index of the current step (inclusive).
        context_window: Number of previous steps to include.

    Returns:
        BERT input text with action + think context only.
    """
    parts = []

    # Context window: last context_window steps (including current)
    start_idx = max(0, current_idx - context_window + 1)
    for i in range(start_idx, current_idx + 1):
        step = steps[i]
        action = step.get("action", "")
        think = step.get("think", "")
        step_num = step.get("step", i)

        if think:
            parts.append(f"Step {step_num}: [{action}] {think}")
        else:
            parts.append(f"Step {step_num}: [{action}]")

    return "\n".join(parts)


def load_analysis_jsons(analysis_dir: Path) -> list[dict]:
    """Load all GPT analysis JSON files from the analysis directory.

    Args:
        analysis_dir: Directory containing one JSON file per task.

    Returns:
        List of analysis dicts.
    """
    analyses = []
    for f in sorted(analysis_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            analyses.append(data)
        except Exception as e:
            print(f"  WARNING: Failed to load {f.name}: {e}")
            continue
    return analyses


def compute_stats(dataset: list[dict], n_tasks_processed: int, n_tasks_skipped: int) -> dict:
    """Compute summary statistics for the dataset.

    Args:
        dataset: List of training sample dicts.
        n_tasks_processed: Number of tasks successfully processed.
        n_tasks_skipped: Number of tasks skipped.

    Returns:
        Dict with summary statistics.
    """
    total = len(dataset)
    if total == 0:
        return {
            "total_samples": 0,
            "total_tasks_processed": n_tasks_processed,
            "total_tasks_skipped": n_tasks_skipped,
        }

    labels = [d["label"] for d in dataset]
    splits = [d["split"] for d in dataset]
    task_names = set(d["task_name"] for d in dataset)

    # Step distribution
    steps_per_task = Counter(d["task_name"] for d in dataset)
    step_counts = list(steps_per_task.values())

    # Class distribution (by step)
    label_counts = Counter(labels)
    stuck_count = label_counts.get(1, 0)
    not_stuck_count = label_counts.get(0, 0)

    # Split distribution (by step)
    split_step_counts = Counter(splits)

    # Split distribution (by task). A task may appear in both splits in
    # sample-level split mode, so count unique task names per split directly.
    split_task_counts = Counter()
    for split_name in ["train", "test"]:
        split_task_counts[split_name] = len(
            {d["task_name"] for d in dataset if d["split"] == split_name}
        )

    # Per-split class balance
    split_class = {}
    for split_name in ["train", "test"]:
        split_samples = [d for d in dataset if d["split"] == split_name]
        if split_samples:
            split_labels = Counter(d["label"] for d in split_samples)
            split_total = len(split_samples)
            split_class[split_name] = {
                "total_samples": split_total,
                "stuck": split_labels.get(1, 0),
                "not_stuck": split_labels.get(0, 0),
                "stuck_fraction": split_labels.get(1, 0) / split_total if split_total > 0 else 0.0,
            }

    return {
        "total_samples": total,
        "total_tasks": len(task_names),
        "total_tasks_processed": n_tasks_processed,
        "total_tasks_skipped": n_tasks_skipped,
        "stuck_samples": stuck_count,
        "not_stuck_samples": not_stuck_count,
        "stuck_fraction": stuck_count / total if total > 0 else 0.0,
        "steps_per_task": {
            "mean": float(np.mean(step_counts)),
            "median": float(np.median(step_counts)),
            "min": int(min(step_counts)),
            "max": int(max(step_counts)),
        },
        "split_distribution_tasks": dict(sorted(split_task_counts.items())),
        "split_distribution_steps": dict(sorted(split_step_counts.items())),
        "per_split_class_balance": split_class,
    }


def print_summary(stats: dict) -> None:
    """Print a readable summary of dataset statistics.

    Args:
        stats: Stats dict from compute_stats.
    """
    print(f"\n{'=' * 70}")
    print("  Stuck Detection Training Dataset Summary")
    print(f"{'=' * 70}")

    print(f"\n  Total samples:           {stats['total_samples']}")
    print(f"  Total tasks:             {stats['total_tasks']}")
    print(f"  Tasks processed:         {stats['total_tasks_processed']}")
    print(f"  Tasks skipped:           {stats['total_tasks_skipped']}")

    if "steps_per_task" in stats:
        spt = stats["steps_per_task"]
        print(
            f"  Steps/task:  mean={spt['mean']:.1f}  median={spt['median']:.0f}"
            f"  min={spt['min']}  max={spt['max']}"
        )

    print(f"\n  Class distribution (by step):")
    stuck = stats.get("stuck_samples", 0)
    not_stuck = stats.get("not_stuck_samples", 0)
    total = stats["total_samples"]
    if total > 0:
        print(f"    not_stuck (0)    {not_stuck:6d}  ({not_stuck / total:.1%})")
        print(f"    stuck (1)        {stuck:6d}  ({stuck / total:.1%})")

    print(f"\n  Split distribution (by task):")
    for split_name, count in stats.get("split_distribution_tasks", {}).items():
        print(f"    {split_name:12s}  {count:4d}")

    print(f"\n  Split distribution (by step):")
    for split_name, count in stats.get("split_distribution_steps", {}).items():
        print(f"    {split_name:12s}  {count:5d}")

    print(f"\n  Per-split class balance:")
    for split_name, info in stats.get("per_split_class_balance", {}).items():
        print(
            f"    {split_name:6s}  total={info['total_samples']:5d}"
            f"  stuck={info['stuck']:5d}  not_stuck={info['not_stuck']:5d}"
            f"  stuck_frac={info['stuck_fraction']:.1%}"
        )

    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build BERT training dataset for stuck detection from GPT analysis results."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing GPT stuck analysis JSONs "
            "(default: results/data/<run-name>/stuck_analysis)"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=OSS20B_RESULTS_DIR,
        help="Path to gpt-oss-20b experiment results (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output dataset JSON path "
            "(default: results/data/<run-name>/stuck_training_dataset.json)"
        ),
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=None,
        help=(
            "Output stats JSON path "
            "(default: results/data/<run-name>/stuck_training_dataset_stats.json)"
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help="Number of previous context steps in sliding window (default: %(default)s)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Fraction of tasks for test split (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--split-mode",
        choices=["task", "sample"],
        default=DEFAULT_SPLIT_MODE,
        help=(
            "Split dataset by task (original behavior) or by individual samples. "
            "Use 'sample' to allow the same task in both train and test "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.analysis_dir is None:
        args.analysis_dir = get_stuck_analysis_dir(args.results_dir)
    if args.output is None:
        args.output = get_stuck_dataset_json(args.results_dir)
    if args.output_stats is None:
        args.output_stats = get_stuck_dataset_stats_json(args.results_dir)

    print(f"{'=' * 70}")
    print("  Building stuck detection training dataset")
    print(f"{'=' * 70}")
    print(f"\n  Analysis dir:    {args.analysis_dir}")
    print(f"  Results dir:     {args.results_dir}")
    print(f"  Output:          {args.output}")
    print(f"  Context window:  {args.context_window}")
    print(f"  Test fraction:   {args.test_fraction}")
    print(f"  Seed:            {args.seed}")
    print(f"  Split mode:      {args.split_mode}")

    # ── 1. Load GPT analysis JSONs ───────────────────────────────────────
    print(f"\nLoading analysis JSONs from {args.analysis_dir}")
    if not args.analysis_dir.exists():
        print(f"  ERROR: Analysis directory does not exist: {args.analysis_dir}")
        sys.exit(1)

    analyses = load_analysis_jsons(args.analysis_dir)
    print(f"  Loaded {len(analyses)} analysis files")

    if not analyses:
        print("  ERROR: No analysis files found. Exiting.")
        sys.exit(1)

    # ── 2. Build per-step dataset ────────────────────────────────────────
    print(f"\nBuilding per-step dataset from trajectories...")

    dataset = []
    tasks_processed = 0
    tasks_skipped = 0

    for analysis in analyses:
        task_name = analysis["task_name"]
        stuck_steps = set(analysis.get("stuck_steps", []))

        # Find the experiment directory
        exp_dir_from_analysis = analysis.get("exp_dir")
        exp_dir = None

        # Try the exp_dir from analysis first
        if exp_dir_from_analysis:
            exp_path = Path(exp_dir_from_analysis)
            if exp_path.exists() and (exp_path / "summary_info.json").exists():
                exp_dir = exp_path

        # Fallback: search in results dir
        if exp_dir is None:
            exp_dir = find_exp_dir(args.results_dir, task_name)

        if exp_dir is None:
            tasks_skipped += 1
            continue

        # Load step data
        steps = load_steps(exp_dir)
        if not steps:
            tasks_skipped += 1
            continue

        # Create one sample per step
        for step_idx in range(len(steps)):
            step_num = steps[step_idx].get("step", step_idx)
            text = build_step_text(steps, step_idx, args.context_window)
            label = 1 if step_num in stuck_steps else 0

            dataset.append(
                {
                    "text": text,
                    "label": label,
                    "task_name": task_name,
                    "step_num": step_num,
                }
            )

        tasks_processed += 1

    print(f"\n  Tasks processed: {tasks_processed}")
    print(f"  Tasks skipped:   {tasks_skipped}")
    print(f"  Total samples:   {len(dataset)}")

    if not dataset:
        print("\n  ERROR: No samples generated. Exiting.")
        sys.exit(1)

    # ── 3. Assign train/test split ───────────────────────────────────────
    if args.split_mode == "task":
        print(f"\nAssigning train/test split by task (test_fraction={args.test_fraction})...")

        all_task_names = sorted(set(d["task_name"] for d in dataset))
        n_test = max(1, int(len(all_task_names) * args.test_fraction))

        shuffled_tasks = list(all_task_names)
        random.shuffle(shuffled_tasks)
        test_tasks = set(shuffled_tasks[:n_test])
        train_tasks = set(shuffled_tasks[n_test:])

        for sample in dataset:
            sample["split"] = "test" if sample["task_name"] in test_tasks else "train"

        n_train = sum(1 for d in dataset if d["split"] == "train")
        n_test_samples = sum(1 for d in dataset if d["split"] == "test")
        print(f"  Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")
        print(f"  Train samples: {n_train}, Test samples: {n_test_samples}")
    else:
        print(f"\nAssigning train/test split by sample (test_fraction={args.test_fraction})...")
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        n_test = max(1, int(len(indices) * args.test_fraction))
        test_indices = set(indices[:n_test])

        for idx, sample in enumerate(dataset):
            sample["split"] = "test" if idx in test_indices else "train"

        n_train = sum(1 for d in dataset if d["split"] == "train")
        n_test_samples = sum(1 for d in dataset if d["split"] == "test")
        print("  Train/test can share task names in sample split mode")
        print(f"  Train samples: {n_train}, Test samples: {n_test_samples}")

    # ── 4. Save dataset ──────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nDataset saved to {args.output}")

    # ── 5. Compute and save stats ────────────────────────────────────────
    stats = compute_stats(dataset, tasks_processed, tasks_skipped)
    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {args.output_stats}")

    print_summary(stats)


if __name__ == "__main__":
    main()
